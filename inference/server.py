"""OpenAI-compatible FastAPI server for scratch GPT checkpoints."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from collections.abc import Iterator
from typing import Any

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from inference.generate import generate, load_model_and_tokenizer, stream_generate


class ChatMessage(BaseModel):
    """OpenAI-style chat message."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """Subset of OpenAI chat completion request fields."""

    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, alias="max_tokens")
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


def chatml_from_messages(messages: list[ChatMessage]) -> str:
    """Convert OpenAI messages into ChatML prompt text."""
    chunks = []
    if not any(message.role == "system" for message in messages):
        chunks.append("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
    for message in messages:
        role = message.role if message.role in {"system", "user", "assistant"} else "user"
        chunks.append(f"<|im_start|>{role}\n{message.content}<|im_end|>\n")
    chunks.append("<|im_start|>assistant\n")
    return "".join(chunks)


def completion_payload(request_id: str, model_name: str, content: str) -> dict[str, Any]:
    """Build a non-streaming OpenAI-compatible completion payload."""
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def stream_payloads(request_id: str, model_name: str, pieces: Iterator[str]) -> Iterator[str]:
    """Yield OpenAI-compatible SSE chunks."""
    for piece in pieces:
        payload = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload)}\n\n"
    final = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


def create_app(model_path: str, quantize: bool = False) -> FastAPI:
    """Create and configure the FastAPI app."""
    model, tokenizer = load_model_and_tokenizer(model_path, quantize=quantize)
    app = FastAPI(title="scratch-llm-openai-server")
    model_name = str(model_path)

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        """Return one available model entry."""
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "scratch-llm"}]}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> Any:
        """Generate a chat completion response."""
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        prompt = chatml_from_messages(request.messages)
        if request.stream:
            pieces = stream_generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            return StreamingResponse(stream_payloads(request_id, model_name, pieces), media_type="text/event-stream")
        content = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        return completion_payload(request_id, model_name, content)

    return app


def parse_args() -> argparse.Namespace:
    """Parse server CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="checkpoints/dpo")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--quantize", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Launch the API server."""
    import uvicorn

    args = parse_args()
    app = create_app(args.model, quantize=args.quantize)
    torch.set_grad_enabled(False)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
