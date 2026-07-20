from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.request import Request as UrlRequest, urlopen

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_ollama import ChatOllama

MODEL_NAME = "gemma4:e4b-it-qat"
HISTORY_DB_PATH = Path(__file__).resolve().parent / "data" / "libby.sqlite3"
SYSTEM_PROMPT = """You are Libby, a local voice assistant and a girl.
Your name is always Libby; never identify yourself as Gemma or as the underlying model.
Use feminine self-reference when relevant, but do not announce your gender unless it matters.
Give concise, warm, natural, plain-text answers suitable for speaking aloud.
Use an available tool whenever the user explicitly asks you to use it.
"""


def warm_libby_model() -> None:
    request = UrlRequest(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(
            {
                "model": MODEL_NAME,
                "keep_alive": -1,
                "stream": False,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        response.read()


@tool
def get_libby_status() -> str:
    """Return Libby's harmless diagnostic status marker."""
    return "LIBBY_TOOL_OK"


def create_libby_agent(*, checkpointer=None):
    model = ChatOllama(
        model=MODEL_NAME,
        reasoning=False,
        temperature=0,
        num_ctx=16_384,
        keep_alive=-1,
    )
    return create_deep_agent(
        model=model,
        tools=[get_libby_status],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


@contextmanager
def open_libby_agent() -> Iterator[Any]:
    HISTORY_DB_PATH.parent.mkdir(exist_ok=True)
    with SqliteSaver.from_conn_string(str(HISTORY_DB_PATH)) as checkpointer:
        yield create_libby_agent(checkpointer=checkpointer)


def extract_response(messages: Sequence[object]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return str(message.content)
    raise RuntimeError("Libby returned no spoken response")


def main() -> None:
    parser = argparse.ArgumentParser(description="Talk with Libby")
    parser.add_argument(
        "--thread",
        default="default",
        help="Persistent conversation thread ID (default: default)",
    )
    parser.add_argument("message", nargs="+", help="Message for Libby")
    args = parser.parse_args()
    prompt = " ".join(args.message)

    with open_libby_agent() as agent:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": {"thread_id": args.thread}},
        )
    print(extract_response(result["messages"]))


if __name__ == "__main__":
    main()
