from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .models import BinaryInput


async def image_inputs_to_files(images: list[dict[str, object]], field_name: str) -> list[tuple[str, BinaryInput]]:
    return [(field_name, await image_input_to_file(item, f"{field_name}.png")) for item in images]


async def image_input_to_file(item: dict[str, object], default_filename: str) -> BinaryInput:
    binary = await resolve_binary_input(item)
    if not binary.filename:
        binary.filename = default_filename
    return binary


async def resolve_binary_input(item: dict[str, object]) -> BinaryInput:
    if not isinstance(item, dict):
        raise ValueError("Image/file input must be an object.")
    mime_type = item.get("mime_type") or item.get("content_type")
    filename = item.get("filename") or item.get("name")

    if item.get("path"):
        path = Path(str(item["path"])).expanduser().resolve()
        data = path.read_bytes()
        return BinaryInput(data, str(filename or path.name), str(mime_type or guess_mime(path.name)))

    if item.get("url"):
        url = str(item["url"])
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.get(url)
        if response.is_error:
            raise RuntimeError(f"Failed to fetch image URL {url}: HTTP {response.status_code}")
        parsed = urlparse(url)
        name = str(filename or Path(parsed.path).name or "remote-image")
        content_type = response.headers.get("content-type", "").split(";")[0]
        return BinaryInput(data=response.content, filename=name, mime_type=str(mime_type or content_type or guess_mime(name)))

    if item.get("data_url"):
        parsed_mime, data = decode_data_url(str(item["data_url"]))
        ext = extension_for_mime(parsed_mime)
        return BinaryInput(data, str(filename or f"image.{ext}"), str(mime_type or parsed_mime))

    raw_b64 = item.get("base64") or item.get("b64_json") or item.get("binary_base64") or item.get("bytes_base64")
    if raw_b64:
        raw_value = str(raw_b64)
        if raw_value.startswith("data:"):
            parsed_mime, data = decode_data_url(raw_value)
            mt = str(mime_type or parsed_mime)
            return BinaryInput(data, str(filename or f"image.{extension_for_mime(mt)}"), mt)
        data = base64.b64decode(raw_value)
        mt = str(mime_type or "image/png")
        return BinaryInput(data, str(filename or f"image.{extension_for_mime(mt)}"), mt)

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
