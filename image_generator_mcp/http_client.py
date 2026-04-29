from __future__ import annotations

import json
from typing import Any, Callable

import httpx

from .constants import DEFAULT_TIMEOUT_SECONDS
from .models import BinaryInput
from .validation import normalize_base_url


def auth_headers(api_key: str) -> dict[str, str]:
    value = (api_key or "").strip()
    if not value:
        raise ValueError("api_key is required for every call.")
    return {"Authorization": f"Bearer {value}"}


def gemini_headers(api_key: str) -> dict[str, str]:
    value = (api_key or "").strip()
    if not value:
        raise ValueError("api_key is required for every call.")
    return {"x-goog-api-key": value}


def parse_response(response: httpx.Response) -> dict[str, Any]:
    if response.is_error:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Non-JSON response: {response.text}") from exc


async def post_json(
    api_key: str,
    base_url: str,
    path: str,
    payload: dict[str, Any],
    base_url_normalizer: Callable[[str], str] = normalize_base_url,
    headers_builder: Callable[[str], dict[str, str]] | None = None,
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    build_headers = headers_builder or auth_headers
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            base_url_normalizer(base_url) + path,
            headers=build_headers(api_key),
            json=payload,
        )
    return parse_response(response)


async def post_multipart(
    api_key: str,
    base_url: str,
    path: str,
    data: dict[str, Any],
    file_parts: list[tuple[str, BinaryInput]],
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    files = [(name, (part.filename, part.data, part.mime_type)) for name, part in file_parts]
    return await post_multipart_raw(api_key, base_url, path, data, files, timeout_seconds=timeout_seconds)


async def post_multipart_raw(
    api_key: str,
    base_url: str,
    path: str,
    data: dict[str, Any],
    files: Any,
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            normalize_base_url(base_url) + path,
            headers=auth_headers(api_key),
            data={k: str(v) for k, v in data.items()},
            files=files,
        )
    return parse_response(response)


async def post_stream_json(
    api_key: str,
    base_url: str,
    path: str,
    payload: dict[str, Any],
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        async with client.stream(
            "POST",
            normalize_base_url(base_url) + path,
            headers=auth_headers(api_key),
            json=payload,
        ) as response:
            await raise_for_stream_error(response)
            return [event async for event in iter_sse_events(response)]


async def post_stream_multipart(
    api_key: str,
    base_url: str,
    path: str,
    data: dict[str, Any],
    file_parts: list[tuple[str, BinaryInput]],
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    files = [(name, (part.filename, part.data, part.mime_type)) for name, part in file_parts]
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        async with client.stream(
            "POST",
            normalize_base_url(base_url) + path,
            headers=auth_headers(api_key),
            data={k: str(v) for k, v in data.items()},
            files=files,
        ) as response:
            await raise_for_stream_error(response)
            return [event async for event in iter_sse_events(response)]


async def raise_for_stream_error(response: httpx.Response) -> None:
    if response.is_error:
        text = await response.aread()
        raise RuntimeError(f"HTTP {response.status_code}: {text.decode(errors='replace')}")


async def iter_sse_events(response: httpx.Response):
    buffer: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if buffer:
                event = parse_sse_block(buffer)
                buffer = []
                if event is not None:
                    yield event
            continue
        buffer.append(line)
    if buffer:
        event = parse_sse_block(buffer)
        if event is not None:
            yield event


def parse_sse_block(lines: list[str]) -> dict[str, Any] | None:
    data_lines = []
    event_name = None
    for line in lines:
        if line.startswith("event:"):
            event_name = line[6:].strip()
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if not data_lines:
        return None
    data = "\n".join(data_lines)
    if data == "[DONE]":
        return {"type": "done"}
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        parsed = {"data": data}
    if event_name and isinstance(parsed, dict) and "type" not in parsed:
        parsed["type"] = event_name
    return parsed
