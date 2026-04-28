from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from mcp.server.fastmcp import FastMCP
from PIL import Image


QUALITY_CHOICES = {"auto", "low", "medium", "high"}
FORMAT_CHOICES = {"png", "jpeg", "jpg", "webp"}
BACKGROUND_CHOICES = {"auto", "opaque", "transparent"}
MODERATION_CHOICES = {"auto", "low"}
ACTION_CHOICES = {"auto", "generate", "edit"}
POPULAR_SIZE_CHOICES = {
    "auto",
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "2048x2048",
    "2048x1152",
    "3840x2160",
    "2160x3840",
}


@dataclass
class BinaryInput:
    data: bytes
    filename: str
    mime_type: str


def create_mcp(host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    mcp = FastMCP(
        "gpt-image-generator",
        instructions=(
            "Generate, edit, stream, and save GPT Image outputs. "
            "Every tool call accepts api_key explicitly; the server does not store tokens."
        ),
        host=host,
        port=port,
    )

    @mcp.tool(
        description=(
            "Generate one or more images with the Image API /images/generations. "
            "Supports size, quality, format, compression, background, moderation, and local saving."
        )
    )
    async def image_generate(
        api_key: str,
        prompt: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-image-2",
        size: str = "auto",
        quality: str = "auto",
        output_format: str = "png",
        output_compression: int | None = None,
        background: str | None = None,
        moderation: str = "auto",
        n: int = 1,
        output_path: str | None = None,
        output_dir: str = "outputs",
    ) -> dict[str, Any]:
        validate_common(prompt, size, quality, output_format, background, moderation, n)
        payload = compact(
            {
                "model": model,
                "prompt": prompt,
                "size": normalize_size(size),
                "quality": quality,
                "n": max(1, n),
                "output_format": normalize_output_format(output_format),
                "output_compression": output_compression,
                "background": background,
                "moderation": moderation,
            }
        )
        body = await post_json(api_key, base_url, "/images/generations", payload)
        return save_image_api_result(body, output_path, output_dir, payload["output_format"])

    @mcp.tool(
        description=(
            "Stream Image API generation with partial_images 0-3. "
            "Saves partial and final images when the upstream API emits them."
        )
    )
    async def image_generate_stream(
        api_key: str,
        prompt: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-image-2",
        size: str = "auto",
        quality: str = "auto",
        output_format: str = "png",
        output_compression: int | None = None,
        background: str | None = None,
        moderation: str = "auto",
        partial_images: int = 2,
        output_dir: str = "outputs",
        filename_prefix: str | None = None,
    ) -> dict[str, Any]:
        validate_common(prompt, size, quality, output_format, background, moderation, 1)
        partial_images = validate_partial_images(partial_images)
        fmt = normalize_output_format(output_format)
        payload = compact(
            {
                "model": model,
                "prompt": prompt,
                "size": normalize_size(size),
                "quality": quality,
                "stream": True,
                "partial_images": partial_images,
                "output_format": fmt,
                "output_compression": output_compression,
                "background": background,
                "moderation": moderation,
            }
        )
        events = await post_stream_json(api_key, base_url, "/images/generations", payload)
        saved = save_stream_images(events, output_dir, filename_prefix, fmt, "image_generation")
        return {"events": summarize_events(events), "saved_images": saved}

    @mcp.tool(
        description=(
            "Edit or compose images with the Image API /images/edits. "
            "Each image and mask can be provided as path, url, base64, data_url, or binary_base64."
        )
    )
    async def image_edit(
        api_key: str,
        prompt: str,
        images: list[dict[str, Any]],
        mask: dict[str, Any] | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-image-2",
        size: str = "auto",
        quality: str = "auto",
        output_format: str = "png",
        output_compression: int | None = None,
        background: str | None = None,
        moderation: str = "auto",
        n: int = 1,
        output_path: str | None = None,
        output_dir: str = "outputs",
    ) -> dict[str, Any]:
        if not images:
            raise ValueError("images must include at least one image input.")
        validate_common(prompt, size, quality, output_format, background, moderation, n)
        fmt = normalize_output_format(output_format)
        data = compact(
            {
                "model": model,
                "prompt": prompt,
                "size": normalize_size(size),
                "quality": quality,
                "n": str(max(1, n)),
                "output_format": fmt,
                "output_compression": output_compression,
                "background": background,
                "moderation": moderation,
            }
        )
        files = await image_inputs_to_files(images, "image[]")
        if mask:
            files.append(("mask", await image_input_to_file(mask, "mask.png")))
        body = await post_multipart(api_key, base_url, "/images/edits", data, files)
        return save_image_api_result(body, output_path, output_dir, fmt)

    @mcp.tool(
        description=(
            "Stream Image API edits with partial_images 0-3. "
            "Image and mask inputs support path, url, base64, data_url, and binary_base64."
        )
    )
    async def image_edit_stream(
        api_key: str,
        prompt: str,
        images: list[dict[str, Any]],
        mask: dict[str, Any] | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-image-2",
        size: str = "auto",
        quality: str = "auto",
        output_format: str = "png",
        output_compression: int | None = None,
        background: str | None = None,
        moderation: str = "auto",
        partial_images: int = 2,
        output_dir: str = "outputs",
        filename_prefix: str | None = None,
    ) -> dict[str, Any]:
        if not images:
            raise ValueError("images must include at least one image input.")
        validate_common(prompt, size, quality, output_format, background, moderation, 1)
        fmt = normalize_output_format(output_format)
        data = compact(
            {
                "model": model,
                "prompt": prompt,
                "size": normalize_size(size),
                "quality": quality,
                "stream": "true",
                "partial_images": str(validate_partial_images(partial_images)),
                "output_format": fmt,
                "output_compression": output_compression,
                "background": background,
                "moderation": moderation,
            }
        )
        files = await image_inputs_to_files(images, "image[]")
        if mask:
            files.append(("mask", await image_input_to_file(mask, "mask.png")))
        events = await post_stream_multipart(api_key, base_url, "/images/edits", data, files)
        saved = save_stream_images(events, output_dir, filename_prefix, fmt, "image_generation")
        return {"events": summarize_events(events), "saved_images": saved}

    @mcp.tool(
        description=(
            "Call the Responses API image_generation tool. Supports multi-turn via previous_response_id, "
            "action auto/generate/edit, file IDs, image URLs/data URLs, masks, and local saving."
        )
    )
    async def responses_image(
        api_key: str,
        prompt: str,
        image_inputs: list[dict[str, Any]] | None = None,
        input_items: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-5.5",
        action: Literal["auto", "generate", "edit"] = "auto",
        size: str | None = None,
        quality: str | None = None,
        output_format: str | None = None,
        output_compression: int | None = None,
        background: str | None = None,
        moderation: str | None = None,
        input_image_mask: dict[str, Any] | None = None,
        output_dir: str = "outputs",
        filename_prefix: str | None = None,
    ) -> dict[str, Any]:
        tool = build_response_image_tool(
            action,
            size,
            quality,
            output_format,
            output_compression,
            background,
            moderation,
            input_image_mask,
        )
        response_input = await build_response_input(prompt, image_inputs, input_items)
        payload = compact(
            {
                "model": model,
                "input": response_input,
                "previous_response_id": previous_response_id,
                "tools": [tool],
            }
        )
        body = await post_json(api_key, base_url, "/responses", payload)
        return save_responses_result(body, output_dir, filename_prefix, output_format or "png")

    @mcp.tool(
        description=(
            "Stream the Responses API image_generation tool. Saves partial image events and final image call results."
        )
    )
    async def responses_image_stream(
        api_key: str,
        prompt: str,
        image_inputs: list[dict[str, Any]] | None = None,
        input_items: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-5.5",
        action: Literal["auto", "generate", "edit"] = "auto",
        size: str | None = None,
        quality: str | None = None,
        output_format: str | None = None,
        output_compression: int | None = None,
        background: str | None = None,
        moderation: str | None = None,
        partial_images: int = 2,
        input_image_mask: dict[str, Any] | None = None,
        output_dir: str = "outputs",
        filename_prefix: str | None = None,
    ) -> dict[str, Any]:
        tool = build_response_image_tool(
            action,
            size,
            quality,
            output_format,
            output_compression,
            background,
            moderation,
            input_image_mask,
            partial_images=validate_partial_images(partial_images),
        )
        response_input = await build_response_input(prompt, image_inputs, input_items)
        payload = compact(
            {
                "model": model,
                "input": response_input,
                "previous_response_id": previous_response_id,
                "stream": True,
                "tools": [tool],
            }
        )
        events = await post_stream_json(api_key, base_url, "/responses", payload)
        fmt = normalize_output_format(output_format or "png")
        saved = save_stream_images(
            events,
            output_dir,
            filename_prefix,
            fmt,
            "response.image_generation_call",
        )
        return {"events": summarize_events(events), "saved_images": saved}

    @mcp.tool(
        description=(
            "Upload a file to OpenAI and return its file_id for Responses API image inputs or masks. "
            "Input supports path, url, base64, data_url, and binary_base64."
        )
    )
    async def upload_file(
        api_key: str,
        file: dict[str, Any],
        base_url: str = "https://api.openai.com/v1",
        purpose: str = "vision",
    ) -> dict[str, Any]:
        binary = await resolve_binary_input(file)
        files = {"file": (binary.filename, binary.data, binary.mime_type)}
        data = {"purpose": purpose}
        body = await post_multipart_raw(api_key, base_url, "/files", data, files)
        return body

    @mcp.tool(
        description=(
            "Convert a black/white or grayscale mask image to a PNG mask with an alpha channel. "
            "Input supports path, url, base64, data_url, and binary_base64."
        )
    )
    async def add_mask_alpha(
        mask: dict[str, Any],
        output_path: str | None = None,
        output_dir: str = "outputs",
    ) -> dict[str, Any]:
        binary = await resolve_binary_input(mask)
        image = Image.open(BytesIO(binary.data)).convert("L")
        rgba = image.convert("RGBA")
        rgba.putalpha(image)
        output = resolve_output_path(output_path, output_dir, "mask-alpha", "png")
        output.parent.mkdir(parents=True, exist_ok=True)
        rgba.save(output, format="PNG")
        return {"output_path": str(output), "mime_type": "image/png"}

    return mcp


def compact(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None}


def normalize_base_url(base_url: str) -> str:
    value = (base_url or "https://api.openai.com/v1").strip().rstrip("/")
    return value if value.endswith("/v1") else f"{value}/v1"


def normalize_size(size: str) -> str:
    value = (size or "auto").strip().lower()
    if value in POPULAR_SIZE_CHOICES or re.fullmatch(r"\d+x\d+", value):
        return value
    raise ValueError("Invalid size. Use auto or WIDTHxHEIGHT, e.g. 1024x1024.")


def normalize_output_format(output_format: str) -> str:
    value = (output_format or "png").strip().lower()
    if value == "jpg":
        value = "jpeg"
    if value not in FORMAT_CHOICES:
        raise ValueError("Invalid output_format. Use png, jpeg, jpg, or webp.")
    return value


def validate_common(
    prompt: str,
    size: str,
    quality: str,
    output_format: str,
    background: str | None,
    moderation: str | None,
    n: int,
) -> None:
    if not (prompt or "").strip():
        raise ValueError("prompt is required.")
    normalize_size(size)
    if quality not in QUALITY_CHOICES:
        raise ValueError("Invalid quality. Use auto, low, medium, or high.")
    normalize_output_format(output_format)
    if background is not None and background not in BACKGROUND_CHOICES:
        raise ValueError("Invalid background. Use auto, opaque, or transparent.")
    if moderation is not None and moderation not in MODERATION_CHOICES:
        raise ValueError("Invalid moderation. Use auto or low.")
    if n < 1:
        raise ValueError("n must be at least 1.")


def validate_partial_images(partial_images: int) -> int:
    if partial_images < 0 or partial_images > 3:
        raise ValueError("partial_images must be between 0 and 3.")
    return partial_images


async def post_json(api_key: str, base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(
            normalize_base_url(base_url) + path,
            headers=auth_headers(api_key),
            json=payload,
        )
    return parse_response(response)


async def post_multipart(
    api_key: str,
    base_url: str,
    path: str,
    data: dict[str, Any],
    file_parts: list[tuple[str, BinaryInput]],
) -> dict[str, Any]:
    files = [(name, (part.filename, part.data, part.mime_type)) for name, part in file_parts]
    return await post_multipart_raw(api_key, base_url, path, data, files)


async def post_multipart_raw(
    api_key: str,
    base_url: str,
    path: str,
    data: dict[str, Any],
    files: Any,
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=300) as client:
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
) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=None) as client:
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
) -> list[dict[str, Any]]:
    files = [(name, (part.filename, part.data, part.mime_type)) for name, part in file_parts]
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            normalize_base_url(base_url) + path,
            headers=auth_headers(api_key),
            data={k: str(v) for k, v in data.items()},
            files=files,
        ) as response:
            await raise_for_stream_error(response)
            return [event async for event in iter_sse_events(response)]


def auth_headers(api_key: str) -> dict[str, str]:
    value = (api_key or "").strip()
    if not value:
        raise ValueError("api_key is required for every call.")
    return {"Authorization": f"Bearer {value}"}


def parse_response(response: httpx.Response) -> dict[str, Any]:
    if response.is_error:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Non-JSON response: {response.text}") from exc


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


async def image_inputs_to_files(images: list[dict[str, Any]], field_name: str) -> list[tuple[str, BinaryInput]]:
    return [(field_name, await image_input_to_file(item, f"{field_name}.png")) for item in images]


async def image_input_to_file(item: dict[str, Any], default_filename: str) -> BinaryInput:
    binary = await resolve_binary_input(item)
    if not binary.filename:
        binary.filename = default_filename
    return binary


async def resolve_binary_input(item: dict[str, Any]) -> BinaryInput:
    if not isinstance(item, dict):
        raise ValueError("Image/file input must be an object.")
    mime_type = item.get("mime_type") or item.get("content_type")
    filename = item.get("filename") or item.get("name")

    if item.get("path"):
        path = Path(str(item["path"])).expanduser().resolve()
        data = path.read_bytes()
        return BinaryInput(data, filename or path.name, mime_type or guess_mime(path.name))

    if item.get("url"):
        url = str(item["url"])
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.get(url)
        if response.is_error:
            raise RuntimeError(f"Failed to fetch image URL {url}: HTTP {response.status_code}")
        parsed = urlparse(url)
        name = filename or Path(parsed.path).name or "remote-image"
        return BinaryInput(data=response.content, filename=name, mime_type=mime_type or response.headers.get("content-type", "").split(";")[0] or guess_mime(name))

    if item.get("data_url"):
        parsed_mime, data = decode_data_url(str(item["data_url"]))
        ext = extension_for_mime(parsed_mime)
        return BinaryInput(data, filename or f"image.{ext}", mime_type or parsed_mime)

    raw_b64 = item.get("base64") or item.get("b64_json") or item.get("binary_base64") or item.get("bytes_base64")
    if raw_b64:
        raw_value = str(raw_b64)
        if raw_value.startswith("data:"):
            parsed_mime, data = decode_data_url(raw_value)
            mt = mime_type or parsed_mime
            return BinaryInput(data, filename or f"image.{extension_for_mime(mt)}", mt)
        data = base64.b64decode(raw_value)
        mt = mime_type or "image/png"
        return BinaryInput(data, filename or f"image.{extension_for_mime(mt)}", mt)

    raise ValueError("Input must include one of: path, url, data_url, base64, binary_base64, bytes_base64.")


def decode_data_url(data_url: str) -> tuple[str, bytes]:
    match = re.fullmatch(r"data:([^;,]+)?(;base64)?,(.*)", data_url, flags=re.DOTALL)
    if not match:
        raise ValueError("Invalid data_url.")
    mime_type = match.group(1) or "application/octet-stream"
    is_base64 = bool(match.group(2))
    payload = match.group(3)
    if not is_base64:
        raise ValueError("Only base64 data URLs are supported.")
    return mime_type, base64.b64decode(payload)


def guess_mime(filename: str) -> str:
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


def extension_for_mime(mime_type: str) -> str:
    if mime_type == "image/jpeg":
        return "jpeg"
    if mime_type == "image/webp":
        return "webp"
    if mime_type == "image/png":
        return "png"
    return (mimetypes.guess_extension(mime_type) or ".bin").lstrip(".")


async def build_response_input(
    prompt: str,
    image_inputs: list[dict[str, Any]] | None,
    input_items: list[dict[str, Any]] | None,
) -> Any:
    if input_items:
        return input_items
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for item in image_inputs or []:
        content.append(await response_input_image_content(item))
    return [{"role": "user", "content": content}]


async def response_input_image_content(item: dict[str, Any]) -> dict[str, Any]:
    if item.get("file_id"):
        return {"type": "input_image", "file_id": item["file_id"]}
    if item.get("url"):
        return {"type": "input_image", "image_url": item["url"]}
    if item.get("data_url"):
        return {"type": "input_image", "image_url": item["data_url"]}
    if item.get("base64") or item.get("b64_json") or item.get("binary_base64") or item.get("bytes_base64") or item.get("path"):
        binary = await resolve_binary_input(item)
        data_url = f"data:{binary.mime_type};base64,{base64.b64encode(binary.data).decode('ascii')}"
        return {"type": "input_image", "image_url": data_url}
    raise ValueError("Response image input must include file_id, url, data_url, path, or base64.")


def build_response_image_tool(
    action: str,
    size: str | None,
    quality: str | None,
    output_format: str | None,
    output_compression: int | None,
    background: str | None,
    moderation: str | None,
    input_image_mask: dict[str, Any] | None,
    partial_images: int | None = None,
) -> dict[str, Any]:
    if action not in ACTION_CHOICES:
        raise ValueError("Invalid action. Use auto, generate, or edit.")
    tool = {"type": "image_generation", "action": action}
    if size is not None:
        tool["size"] = normalize_size(size)
    if quality is not None:
        if quality not in QUALITY_CHOICES:
            raise ValueError("Invalid quality. Use auto, low, medium, or high.")
        tool["quality"] = quality
    if output_format is not None:
        tool["output_format"] = normalize_output_format(output_format)
    if output_compression is not None:
        tool["output_compression"] = output_compression
    if background is not None:
        if background not in BACKGROUND_CHOICES:
            raise ValueError("Invalid background. Use auto, opaque, or transparent.")
        tool["background"] = background
    if moderation is not None:
        if moderation not in MODERATION_CHOICES:
            raise ValueError("Invalid moderation. Use auto or low.")
        tool["moderation"] = moderation
    if partial_images is not None:
        tool["partial_images"] = partial_images
    if input_image_mask:
        if input_image_mask.get("file_id"):
            tool["input_image_mask"] = {"file_id": input_image_mask["file_id"]}
        else:
            raise ValueError("Responses input_image_mask requires file_id. Use upload_file first.")
    return tool


def save_image_api_result(
    body: dict[str, Any],
    output_path: str | None,
    output_dir: str,
    output_format: str,
) -> dict[str, Any]:
    data = body.get("data") or []
    if not data:
        raise RuntimeError(f"No image data returned: {body}")
    saved = []
    multiple = len(data) > 1
    for idx, item in enumerate(data):
        b64 = item.get("b64_json")
        if not b64:
            continue
        path = resolve_output_path(output_path, output_dir, "image", output_format, idx if multiple else None)
        write_b64_image(b64, path)
        saved.append(
            {
                "path": str(path),
                "revised_prompt": item.get("revised_prompt"),
            }
        )
    return {"saved_images": saved, "raw": redact_large_images(body)}


def save_responses_result(
    body: dict[str, Any],
    output_dir: str,
    filename_prefix: str | None,
    output_format: str,
) -> dict[str, Any]:
    calls = [item for item in body.get("output", []) if item.get("type") == "image_generation_call"]
    saved = []
    fmt = normalize_output_format(output_format)
    prefix = filename_prefix or "response-image"
    for idx, call in enumerate(calls):
        b64 = call.get("result")
        if not b64:
            continue
        path = resolve_output_path(None, output_dir, prefix, fmt, idx if len(calls) > 1 else None)
        write_b64_image(b64, path)
        saved.append(
            {
                "path": str(path),
                "id": call.get("id"),
                "revised_prompt": call.get("revised_prompt"),
            }
        )
    return {
        "response_id": body.get("id"),
        "saved_images": saved,
        "image_generation_calls": redact_large_images(calls),
    }


def save_stream_images(
    events: list[dict[str, Any]],
    output_dir: str,
    filename_prefix: str | None,
    output_format: str,
    event_prefix: str,
) -> list[dict[str, Any]]:
    saved = []
    fmt = normalize_output_format(output_format)
    prefix = filename_prefix or "stream-image"
    for index, event in enumerate(events):
        event_type = str(event.get("type", ""))
        b64 = (
            event.get("b64_json")
            or event.get("partial_image_b64")
            or event.get("result")
        )
        if not b64 and isinstance(event.get("response"), dict):
            saved.extend(save_stream_images([event["response"]], output_dir, prefix, fmt, event_prefix))
            continue
        if not b64 and event.get("output"):
            for item in event["output"]:
                if isinstance(item, dict) and item.get("type") == "image_generation_call" and item.get("result"):
                    b64 = item["result"]
                    break
        if not b64:
            continue
        partial_idx = event.get("partial_image_index")
        label = f"{prefix}-partial-{partial_idx}" if "partial" in event_type else f"{prefix}-final"
        path = resolve_output_path(None, output_dir, label, fmt, index)
        write_b64_image(b64, path)
        saved.append({"path": str(path), "event_type": event_type, "partial_image_index": partial_idx})
    return saved


def summarize_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for event in events:
        result.append(redact_large_images(event))
    return result


def redact_large_images(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if key in {"b64_json", "partial_image_b64", "result"} and isinstance(item, str) and len(item) > 128:
                redacted[key] = f"<base64 {len(item)} chars>"
            else:
                redacted[key] = redact_large_images(item)
        return redacted
    if isinstance(value, list):
        return [redact_large_images(item) for item in value]
    return value


def write_b64_image(b64: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(b64))


def resolve_output_path(
    output_path: str | None,
    output_dir: str,
    stem: str,
    extension: str,
    index: int | None = None,
) -> Path:
    ext = normalize_output_format(extension)
    if output_path:
        path = Path(output_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if index is not None:
            path = path.with_name(f"{path.stem}-{index}{path.suffix or f'.{ext}'}")
        if not path.suffix:
            path = path.with_suffix(f".{ext}")
        return path.resolve()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"-{index}" if index is not None else ""
    return (Path.cwd() / output_dir / f"{stem}-{timestamp}{suffix}.{ext}").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT Image MCP server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport to run.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for SSE/streamable-http.")
    parser.add_argument("--port", type=int, default=8000, help="Port for SSE/streamable-http.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_mcp(host=args.host, port=args.port).run(transport=args.transport)


if __name__ == "__main__":
    main()
