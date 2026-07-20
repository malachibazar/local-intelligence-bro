from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import soundfile as sf
from websockets.asyncio.client import ClientConnection, connect

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2
FRAME_BYTES = 3_200


async def receive_event(websocket: ClientConnection) -> dict[str, Any]:
    message = await websocket.recv()
    if isinstance(message, bytes):
        raise RuntimeError("Received unexpected binary audio")
    return json.loads(message)


async def send_wav(websocket: ClientConnection, path: Path) -> None:
    audio, sample_rate = sf.read(path, dtype="int16", always_2d=True)
    if sample_rate != SAMPLE_RATE or audio.shape[1] != CHANNELS:
        raise ValueError("Input WAV must be mono 16-bit PCM at 16 kHz")

    pcm = audio[:, 0].astype("<i2", copy=False).tobytes()
    for offset in range(0, len(pcm), FRAME_BYTES):
        await websocket.send(pcm[offset : offset + FRAME_BYTES])


async def stream_microphone(websocket: ClientConnection, seconds: float) -> None:
    process = await asyncio.create_subprocess_exec(
        "arecord",
        "-q",
        "-t",
        "raw",
        "-f",
        "S16_LE",
        "-r",
        str(SAMPLE_RATE),
        "-c",
        str(CHANNELS),
        stdout=asyncio.subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError("arecord did not provide an audio stream")

    print(f"Speak now for {seconds:g} seconds...")
    deadline = asyncio.get_running_loop().time() + seconds
    try:
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                frame = await asyncio.wait_for(
                    process.stdout.read(FRAME_BYTES),
                    timeout=remaining,
                )
            except TimeoutError:
                break
            if not frame:
                raise RuntimeError("arecord stopped before recording completed")
            await websocket.send(frame)
    finally:
        if process.returncode is None:
            process.terminate()
        await process.wait()


async def run(args: argparse.Namespace) -> None:
    async with connect(args.url, max_size=None) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "audio.input.start",
                    "thread_id": args.thread,
                    "output_audio": not args.text_only,
                    "encoding": "pcm_s16le",
                    "sample_rate": SAMPLE_RATE,
                    "channels": CHANNELS,
                }
            )
        )
        ready = await receive_event(websocket)
        if ready.get("type") != "audio.input.ready":
            raise RuntimeError(f"Server rejected audio input: {ready}")

        if args.wav is not None:
            await send_wav(websocket, args.wav)
        else:
            await stream_microphone(websocket, args.seconds)

        await websocket.send(
            json.dumps(
                {
                    "type": "audio.input.stop",
                    "input_id": ready["input_id"],
                }
            )
        )

        output_process: asyncio.subprocess.Process | None = None
        printed_delta = False
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    if output_process is None or output_process.stdin is None:
                        raise RuntimeError("Received audio before audio.started")
                    output_process.stdin.write(message)
                    await output_process.stdin.drain()
                    continue

                event = json.loads(message)
                event_type = event.get("type")
                if event_type == "transcript":
                    print(f"You: {event['text']}")
                elif event_type == "assistant.delta":
                    if not printed_delta:
                        print("Libby: ", end="", flush=True)
                        printed_delta = True
                    print(event["text"], end="", flush=True)
                elif event_type == "audio.started":
                    if printed_delta:
                        print()
                    output_process = await asyncio.create_subprocess_exec(
                        "aplay",
                        "-q",
                        "-t",
                        "raw",
                        "-f",
                        "S16_LE",
                        "-r",
                        str(event["sample_rate"]),
                        "-c",
                        str(event["channels"]),
                        stdin=asyncio.subprocess.PIPE,
                    )
                elif event_type == "audio.completed":
                    if output_process is not None:
                        if output_process.stdin is not None:
                            output_process.stdin.close()
                        await output_process.wait()
                        output_process = None
                elif event_type == "turn.completed":
                    if printed_delta and args.text_only:
                        print()
                    return
                elif event_type == "error":
                    raise RuntimeError(f"Server error: {event}")
        finally:
            if output_process is not None and output_process.returncode is None:
                output_process.terminate()
                await output_process.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a voice turn with Libby")
    parser.add_argument("--url", default="ws://127.0.0.1:8001/ws/chat")
    parser.add_argument("--thread", default="voice-client")
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--wav", type=Path, help="Stream a mono 16 kHz PCM WAV instead of the microphone")
    parser.add_argument("--text-only", action="store_true", help="Do not request spoken output")
    args = parser.parse_args()
    if args.seconds <= 0 or args.seconds > 30:
        parser.error("--seconds must be greater than 0 and at most 30")
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
