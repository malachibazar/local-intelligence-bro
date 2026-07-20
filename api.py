from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Generator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, StringConstraints, ValidationError

from libby import (
    HISTORY_DB_PATH,
    create_libby_agent,
    extract_response,
    warm_libby_model,
)
from speech import PocketSpeechService, SynthesizedAudio
from transcription import (
    SAMPLE_RATE,
    NoSpeechDetectedError,
    WhisperSpeechRecognitionService,
)
logger = logging.getLogger(__name__)
FACE_PATH = Path(__file__).resolve().parent / "static" / "index.html"

ThreadId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
Message = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20_000)]


class ChatRequest(BaseModel):
    thread_id: ThreadId = "default"
    message: Message


class ChatResponse(BaseModel):
    thread_id: str
    response: str


class HealthResponse(BaseModel):
    status: str

class TurnStart(BaseModel):
    type: Literal["turn.start"]
    thread_id: ThreadId = "default"
    message: Message
    output_audio: bool = False


class TurnCancel(BaseModel):
    type: Literal["turn.cancel"]
    turn_id: str

class AudioInputStart(BaseModel):
    type: Literal["audio.input.start"]
    thread_id: ThreadId = "default"
    output_audio: bool = True
    encoding: Literal["pcm_s16le"] = "pcm_s16le"
    sample_rate: Literal[16_000] = 16_000
    channels: Literal[1] = 1


class AudioInputStop(BaseModel):
    type: Literal["audio.input.stop"]
    input_id: str


@dataclass(slots=True)
class ActiveTurn:
    turn_id: str
    task: asyncio.Task[None]

@dataclass(slots=True)
class AudioInputSession:
    input_id: str
    thread_id: str
    output_audio: bool
    pcm_s16le: bytearray


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    HISTORY_DB_PATH.parent.mkdir(exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(HISTORY_DB_PATH)) as checkpointer:
        await checkpointer.setup()
        await asyncio.to_thread(warm_libby_model)
        warm_agent = create_libby_agent()
        await warm_agent.ainvoke(
            {"messages": [{"role": "user", "content": "Reply with exactly READY."}]}
        )
        app.state.agent = create_libby_agent(checkpointer=checkpointer)
        app.state.thread_locks = {}
        app.state.speech = PocketSpeechService()
        app.state.transcriber = WhisperSpeechRecognitionService()
        yield


app = FastAPI(title="Libby API", lifespan=lifespan)


@app.get("/", include_in_schema=False)
async def face() -> FileResponse:
    return FileResponse(FACE_PATH, headers={"Cache-Control": "no-store"})


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    locks: dict[str, asyncio.Lock] = request.app.state.thread_locks
    lock = locks.setdefault(payload.thread_id, asyncio.Lock())

    agent: Any = request.app.state.agent
    async with lock:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": payload.message}]},
            config={"configurable": {"thread_id": payload.thread_id}},
        )

    return ChatResponse(
        thread_id=payload.thread_id,
        response=extract_response(result["messages"]),
    )


async def _send_event(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    event: dict[str, object],
) -> bool:
    try:
        async with send_lock:
            await websocket.send_json(event)
        return True
    except (RuntimeError, WebSocketDisconnect):
        return False

async def _send_audio_chunk(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    chunk: bytes,
) -> bool:
    try:
        async with send_lock:
            await websocket.send_bytes(chunk)
        return True
    except (RuntimeError, WebSocketDisconnect):
        return False


async def _next_audio_chunk(
    stream: Generator[SynthesizedAudio, None, None],
) -> SynthesizedAudio | None:
    task = asyncio.create_task(asyncio.to_thread(next, stream, None))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.shield(task)
        raise


def _take_speech_phrases(
    text: str,
    *,
    flush: bool = False,
) -> tuple[list[str], str]:
    phrases: list[str] = []
    remaining = text.lstrip()
    while remaining:
        cut = None
        for index, character in enumerate(remaining):
            length = index + 1
            if character in ".!?\n" and length >= 24:
                cut = length
                break
            if character in ",;:" and length >= 40:
                cut = length
                break
            if length >= 72:
                cut = remaining.rfind(" ", 0, 73)
                if cut < 24:
                    cut = 72
                break

        if cut is None:
            break

        phrase = remaining[:cut].strip()
        if phrase:
            phrases.append(phrase)
        remaining = remaining[cut:].lstrip()

    if flush and remaining.strip():
        phrases.append(remaining.strip())
        remaining = ""
    return phrases, remaining


async def _stream_audio_phrases(
    websocket: WebSocket,
    turn_id: str,
    speech: PocketSpeechService,
    phrases: asyncio.Queue[str | None],
    send_lock: asyncio.Lock,
) -> bool:
    synthesizing = False
    started = False
    byte_count = 0

    while True:
        phrase = await phrases.get()
        if phrase is None:
            break

        if not synthesizing:
            synthesizing = True
            delivered = await _send_event(
                websocket,
                send_lock,
                {
                    "type": "audio.synthesizing",
                    "turn_id": turn_id,
                },
            )
            if not delivered:
                return False

        audio_stream = speech.stream(phrase)
        try:
            audio = await _next_audio_chunk(audio_stream)
            if audio is None:
                raise RuntimeError("Pocket TTS returned no audio")

            if not started:
                started = True
                delivered = await _send_event(
                    websocket,
                    send_lock,
                    {
                        "type": "audio.started",
                        "turn_id": turn_id,
                        "encoding": "pcm_s16le",
                        "sample_rate": audio.sample_rate,
                        "channels": 1,
                        "sample_width": 2,
                    },
                )
                if not delivered:
                    return False

            while audio is not None:
                delivered = await _send_audio_chunk(
                    websocket,
                    send_lock,
                    audio.pcm_s16le,
                )
                if not delivered:
                    return False

                byte_count += len(audio.pcm_s16le)
                audio = await _next_audio_chunk(audio_stream)
        finally:
            await asyncio.to_thread(audio_stream.close)

    if not started:
        return True
    return await _send_event(
        websocket,
        send_lock,
        {
            "type": "audio.completed",
            "turn_id": turn_id,
            "byte_count": byte_count,
        },
    )


async def _stream_turn(
    websocket: WebSocket,
    request: TurnStart,
    turn_id: str,
    agent: Any,
    speech: PocketSpeechService,
    thread_lock: asyncio.Lock,
    send_lock: asyncio.Lock,
) -> None:
    config = {"configurable": {"thread_id": request.thread_id}}
    phrase_queue: asyncio.Queue[str | None] | None = None
    audio_task: asyncio.Task[bool] | None = None
    speech_buffer = ""
    queued_speech = False

    if request.output_audio:
        phrase_queue = asyncio.Queue()
        audio_task = asyncio.create_task(
            _stream_audio_phrases(
                websocket,
                turn_id,
                speech,
                phrase_queue,
                send_lock,
            )
        )

    try:
        async with thread_lock:
            async for chunk in agent.astream(
                {"messages": [{"role": "user", "content": request.message}]},
                config=config,
                stream_mode="messages",
                subgraphs=True,
                version="v2",
            ):
                if chunk["type"] != "messages" or chunk["ns"]:
                    continue

                token, _metadata = chunk["data"]
                if (
                    not isinstance(token, AIMessageChunk)
                    or token.tool_call_chunks
                    or not isinstance(token.content, str)
                    or not token.content
                ):
                    continue

                delivered = await _send_event(
                    websocket,
                    send_lock,
                    {
                        "type": "assistant.delta",
                        "turn_id": turn_id,
                        "text": token.content,
                    },
                )
                if not delivered:
                    return

                if phrase_queue is not None:
                    speech_buffer += token.content
                    ready_phrases, speech_buffer = _take_speech_phrases(speech_buffer)
                    for phrase in ready_phrases:
                        queued_speech = True
                        phrase_queue.put_nowait(phrase)

            state = await agent.aget_state(config)
            response = extract_response(state.values["messages"])

            if phrase_queue is not None and audio_task is not None:
                ready_phrases, speech_buffer = _take_speech_phrases(
                    speech_buffer,
                    flush=True,
                )
                for phrase in ready_phrases:
                    queued_speech = True
                    phrase_queue.put_nowait(phrase)
                if not queued_speech:
                    phrase_queue.put_nowait(response)
                phrase_queue.put_nowait(None)

                delivered = await audio_task
                audio_task = None
                if not delivered:
                    return

            await _send_event(
                websocket,
                send_lock,
                {
                    "type": "turn.completed",
                    "turn_id": turn_id,
                    "response": response,
                },
            )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Libby WebSocket turn %s failed", turn_id)
        await _send_event(
            websocket,
            send_lock,
            {
                "type": "error",
                "turn_id": turn_id,
                "code": "turn_failed",
                "message": "Libby could not complete the turn.",
            },
        )
    finally:
        if audio_task is not None:
            audio_task.cancel()
            try:
                await audio_task
            except asyncio.CancelledError:
                pass


async def _transcribe_turn(
    websocket: WebSocket,
    audio_input: AudioInputSession,
    pcm_s16le: bytes,
    turn_id: str,
    agent: Any,
    speech: PocketSpeechService,
    transcriber: WhisperSpeechRecognitionService,
    thread_lock: asyncio.Lock,
    send_lock: asyncio.Lock,
) -> None:
    try:
        transcript = await asyncio.to_thread(transcriber.transcribe, pcm_s16le)
        delivered = await _send_event(
            websocket,
            send_lock,
            {
                "type": "transcript",
                "turn_id": turn_id,
                "input_id": audio_input.input_id,
                "thread_id": audio_input.thread_id,
                "text": transcript.text,
                "language": transcript.language,
                "language_probability": transcript.language_probability,
            },
        )
        if not delivered:
            return

        await _send_event(
            websocket,
            send_lock,
            {"type": "turn.started", "turn_id": turn_id},
        )
        await _stream_turn(
            websocket,
            TurnStart(
                type="turn.start",
                thread_id=audio_input.thread_id,
                message=transcript.text,
                output_audio=audio_input.output_audio,
            ),
            turn_id,
            agent,
            speech,
            thread_lock,
            send_lock,
        )
    except asyncio.CancelledError:
        raise
    except NoSpeechDetectedError:
        await _send_event(
            websocket,
            send_lock,
            {
                "type": "error",
                "turn_id": turn_id,
                "code": "no_speech",
                "message": "No speech was detected.",
            },
        )
    except Exception:
        logger.exception("Libby transcription turn %s failed", turn_id)
        await _send_event(
            websocket,
            send_lock,
            {
                "type": "error",
                "turn_id": turn_id,
                "code": "transcription_failed",
                "message": "Libby could not transcribe the audio.",
            },
        )


async def _cancel_active_turn(active: ActiveTurn | None) -> bool:
    if active is None or active.task.done():
        return False

    active.task.cancel()
    try:
        await active.task
    except asyncio.CancelledError:
        pass
    return True


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    await websocket.accept()
    send_lock = asyncio.Lock()
    active: ActiveTurn | None = None
    audio_input: AudioInputSession | None = None
    max_audio_bytes = SAMPLE_RATE * 2 * 30

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))

            binary = message.get("bytes")
            if binary is not None:
                if audio_input is None:
                    await _send_event(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "code": "unexpected_audio",
                            "message": "Send audio.input.start before binary audio.",
                        },
                    )
                    continue

                if len(audio_input.pcm_s16le) + len(binary) > max_audio_bytes:
                    audio_input = None
                    await _send_event(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "code": "audio_too_large",
                            "message": "Audio input cannot exceed 30 seconds.",
                        },
                    )
                    continue

                audio_input.pcm_s16le.extend(binary)
                continue

            raw_text = message.get("text")
            try:
                raw_event = json.loads(raw_text) if raw_text is not None else None
            except json.JSONDecodeError:
                raw_event = None
            event_type = raw_event.get("type") if isinstance(raw_event, dict) else None

            if event_type == "audio.input.start":
                try:
                    start_audio = AudioInputStart.model_validate(raw_event)
                except ValidationError:
                    await _send_event(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "code": "invalid_event",
                            "message": "Invalid audio.input.start event.",
                        },
                    )
                    continue

                if audio_input is not None:
                    await _send_event(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "code": "audio_input_in_progress",
                            "message": "An audio input stream is already active.",
                        },
                    )
                    continue

                if active is not None and not active.task.done():
                    await _send_event(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "code": "turn_in_progress",
                            "message": "Cancel the active turn before recording.",
                        },
                    )
                    continue

                audio_input = AudioInputSession(
                    input_id=str(uuid4()),
                    thread_id=start_audio.thread_id,
                    output_audio=start_audio.output_audio,
                    pcm_s16le=bytearray(),
                )
                await _send_event(
                    websocket,
                    send_lock,
                    {
                        "type": "audio.input.ready",
                        "input_id": audio_input.input_id,
                        "encoding": "pcm_s16le",
                        "sample_rate": SAMPLE_RATE,
                        "channels": 1,
                        "sample_width": 2,
                        "max_seconds": 30,
                    },
                )
                continue

            if event_type == "audio.input.stop":
                try:
                    stop_audio = AudioInputStop.model_validate(raw_event)
                except ValidationError:
                    await _send_event(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "code": "invalid_event",
                            "message": "Invalid audio.input.stop event.",
                        },
                    )
                    continue

                if audio_input is None or audio_input.input_id != stop_audio.input_id:
                    await _send_event(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "code": "unknown_audio_input",
                            "message": "The requested audio input stream is not active.",
                        },
                    )
                    continue

                completed_input = audio_input
                audio_input = None
                turn_id = str(uuid4())
                locks: dict[str, asyncio.Lock] = websocket.app.state.thread_locks
                thread_lock = locks.setdefault(completed_input.thread_id, asyncio.Lock())
                await _send_event(
                    websocket,
                    send_lock,
                    {
                        "type": "transcription.started",
                        "turn_id": turn_id,
                        "input_id": completed_input.input_id,
                    },
                )
                task = asyncio.create_task(
                    _transcribe_turn(
                        websocket,
                        completed_input,
                        bytes(completed_input.pcm_s16le),
                        turn_id,
                        websocket.app.state.agent,
                        websocket.app.state.speech,
                        websocket.app.state.transcriber,
                        thread_lock,
                        send_lock,
                    )
                )
                active = ActiveTurn(turn_id=turn_id, task=task)
                continue

            if event_type == "turn.start":
                try:
                    start = TurnStart.model_validate(raw_event)
                except ValidationError:
                    await _send_event(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "code": "invalid_event",
                            "message": "Invalid turn.start event.",
                        },
                    )
                    continue

                if audio_input is not None:
                    await _send_event(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "code": "audio_input_in_progress",
                            "message": "Stop the audio input before starting a text turn.",
                        },
                    )
                    continue

                if active is not None and not active.task.done():
                    await _send_event(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "code": "turn_in_progress",
                            "message": "Cancel the active turn before starting another.",
                        },
                    )
                    continue

                turn_id = str(uuid4())
                locks: dict[str, asyncio.Lock] = websocket.app.state.thread_locks
                thread_lock = locks.setdefault(start.thread_id, asyncio.Lock())
                await _send_event(
                    websocket,
                    send_lock,
                    {"type": "turn.started", "turn_id": turn_id},
                )
                task = asyncio.create_task(
                    _stream_turn(
                        websocket,
                        start,
                        turn_id,
                        websocket.app.state.agent,
                        websocket.app.state.speech,
                        thread_lock,
                        send_lock,
                    )
                )
                active = ActiveTurn(turn_id=turn_id, task=task)
                continue

            if event_type == "turn.cancel":
                try:
                    cancel = TurnCancel.model_validate(raw_event)
                except ValidationError:
                    await _send_event(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "code": "invalid_event",
                            "message": "Invalid turn.cancel event.",
                        },
                    )
                    continue

                if active is None or active.turn_id != cancel.turn_id:
                    await _send_event(
                        websocket,
                        send_lock,
                        {
                            "type": "error",
                            "code": "unknown_turn",
                            "message": "The requested turn is not active.",
                        },
                    )
                    continue

                if await _cancel_active_turn(active):
                    await _send_event(
                        websocket,
                        send_lock,
                        {"type": "turn.cancelled", "turn_id": active.turn_id},
                    )
                active = None
                continue

            await _send_event(
                websocket,
                send_lock,
                {
                    "type": "error",
                    "code": "invalid_event",
                    "message": (
                        "Event type must be audio.input.start, audio.input.stop, "
                        "turn.start, or turn.cancel."
                    ),
                },
            )
    except WebSocketDisconnect:
        await _cancel_active_turn(active)
