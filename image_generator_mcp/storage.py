from __future__ import annotations

import base64
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from .constants import DEFAULT_TIMEOUT_SECONDS
from .inputs import decode_data_url, extension_for_mime
from .validation import normalize_output_extension, normalize_output_format

logger = logging.getLogger("image_generator_mcp")


def save_image_api_result(
    body: dict[str, Any],
    output_path: str | None,
    output_dir: str,
    output_format: str,
    filename_prefix: str | None = None,
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    data = body.get("data") or []
    if not data:
        return {
            "status": "no_image_data",
            "message": "The upstream request succeeded, but the response did not include image data.",
            "saved_images": [],
            "raw": redact_large_images(body),
        }
    saved = []
    multiple = len(data) > 1
    for idx, item in enumerate(data):
        b64 = extract_image_b64(item)
        url = item.get("url")
        path = resolve_output_path(output_path, output_dir, filename_prefix or "image", output_format, idx if multiple else None)
        if b64:
            write_b64_image(b64, path)
        elif url:
            try:
                path = download_image_to_path(url, path, timeout_seconds=timeout_seconds)
            except Exception as exc:
                saved.append(
                    {
                        "path": None,
                        "revised_prompt": item.get("revised_prompt"),
                        "url": url,
                        "download_error": str(exc),
                    }
                )
                continue
        else:
            continue
        saved.append(
            {
                "path": str(path),
                "revised_prompt": item.get("revised_prompt"),
                "url": url,
            }
        )
    return {
        "status": saved_images_status(saved),
        "message": saved_images_message(saved),
        "saved_images": saved,
        "raw": redact_large_images(body),
    }


def save_gemini_result(
    body: dict[str, Any],
    output_path: str | None,
    output_dir: str,
    filename_prefix: str | None,
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    image_parts = []
    text_parts = []
    for candidate_index, candidate in enumerate(body.get("candidates") or []):
        content = candidate.get("content") or {}
        for part_index, part in enumerate(content.get("parts") or []):
            inline_data = part.get("inlineData") or part.get("inline_data")
            file_data = part.get("fileData") or part.get("file_data")
            if inline_data:
                image_parts.append((candidate_index, part_index, inline_data))
            elif file_data and (file_data.get("fileUri") or file_data.get("file_uri")):
                image_parts.append((candidate_index, part_index, file_data))
            elif part.get("text"):
                text_parts.append(part["text"])

    if not image_parts:
        return {
            "status": "no_image_data",
            "message": "The upstream request succeeded, but the response did not include image data.",
            "saved_images": [],
            "text": "\n".join(text_parts),
            "raw": redact_large_images(body),
        }

    saved = []
    multiple = len(image_parts) > 1
    prefix = filename_prefix or "gemini-image"
    for idx, (candidate_index, part_index, image_data) in enumerate(image_parts):
        mime_type = image_data.get("mimeType") or image_data.get("mime_type") or "image/png"
        ext = extension_for_mime(mime_type)
        if ext == "jpg":
            ext = "jpeg"
        path = resolve_output_path(output_path, output_dir, prefix, ext, idx if multiple else None)
        b64 = image_data.get("data")
        uri = image_data.get("fileUri") or image_data.get("file_uri")
        if b64:
            write_b64_image(b64, path)
        elif uri:
            try:
                path = download_image_to_path(uri, path, timeout_seconds=timeout_seconds)
            except Exception as exc:
                saved.append(
                    {
                        "path": None,
                        "mime_type": mime_type,
                        "candidate_index": candidate_index,
                        "part_index": part_index,
                        "uri": uri,
                        "download_error": str(exc),
                    }
                )
                continue
        else:
            continue
        saved.append(
            {
                "path": str(path),
                "mime_type": mime_type,
                "candidate_index": candidate_index,
                "part_index": part_index,
                "uri": uri,
            }
        )

    return {
        "status": saved_images_status(saved),
        "message": saved_images_message(saved),
        "saved_images": saved,
        "text": "\n".join(text_parts),
        "raw": redact_large_images(body),
    }


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
        "status": saved_images_status(saved),
        "message": saved_images_message(saved),
        "response_id": body.get("id"),
        "saved_images": saved,
        "image_generation_calls": redact_large_images(calls),
    }


def save_generic_image_result(
    body: dict[str, Any],
    output_dir: str,
    filename_prefix: str | None,
    output_format: str = "png",
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    images = collect_embedded_images(body)
    if not images:
        return {
            "status": "no_image_data",
            "message": "The upstream request succeeded, but no embedded image data or image URL was found in the response.",
            "saved_images": [],
            "raw": redact_large_images(body),
        }

    saved = []
    prefix = filename_prefix or "image"
    for idx, image in enumerate(images):
        mime_type = image.get("mime_type") or "image/png"
        ext = extension_for_mime(mime_type) if mime_type.startswith("image/") else output_format
        path = resolve_output_path(None, output_dir, prefix, ext, idx if len(images) > 1 else None)
        if image.get("data_url"):
            parsed_mime, data = decode_data_url(image["data_url"])
            path = path.with_suffix(f".{extension_for_mime(parsed_mime)}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        elif image.get("b64"):
            write_b64_image(image["b64"], path)
        elif image.get("url"):
            try:
                path = download_image_to_path(image["url"], path, timeout_seconds=timeout_seconds)
            except Exception as exc:
                saved.append(
                    {
                        "path": None,
                        "mime_type": mime_type,
                        "source": image.get("source"),
                        "url": image["url"],
                        "download_error": str(exc),
                    }
                )
                continue
        else:
            continue
        saved.append(
            {
                "path": str(path),
                "mime_type": mime_type,
                "source": image.get("source"),
            }
        )

    return {
        "status": saved_images_status(saved),
        "message": saved_images_message(saved),
        "saved_images": saved,
        "raw": redact_large_images(body),
    }


def saved_images_status(saved: list[dict[str, Any]]) -> str:
    if any(item.get("path") for item in saved):
        for item in saved:
            if item.get("path"):
                logger.info("Saved image to %s", item["path"])
        return "partial_success" if any(item.get("download_error") for item in saved) else "saved"
    if any(item.get("download_error") for item in saved):
        logger.warning("Image generation completed, but image download failed: %s", saved)
        return "download_failed"
    logger.warning("Image generation completed, but no image file was saved.")
    return "no_saved_images"


def saved_images_message(saved: list[dict[str, Any]]) -> str:
    if any(item.get("path") for item in saved):
        return "Image generation completed and saved locally."
    if any(item.get("download_error") for item in saved):
        return "Image generation completed, but downloading the returned image URL failed."
    return "Image generation completed, but no image file was saved."


def collect_embedded_images(value: Any) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, payload: str, mime_type: str = "image/png", source: str = "") -> None:
        if not payload:
            return
        key = (kind, payload[:128])
        if key in seen:
            return
        seen.add(key)
        item = {"mime_type": mime_type, "source": source}
        item[kind] = payload
        images.append(item)

    def walk(node: Any, source: str = "$") -> None:
        if isinstance(node, dict):
            mime_type = str(node.get("mimeType") or node.get("mime_type") or node.get("content_type") or "image/png")
            data = node.get("data")
            if isinstance(data, str) and mime_type.startswith("image/"):
                add("b64", data, mime_type, source)
            for key in ("b64_json", "base64", "image_data", "partial_image_b64", "result"):
                payload = node.get(key)
                if isinstance(payload, str):
                    add("b64", payload, mime_type, f"{source}.{key}")
            for key in ("image_url", "image"):
                payload = node.get(key)
                if isinstance(payload, dict):
                    url = payload.get("url")
                    if isinstance(url, str):
                        add_url_or_data_url(url, mime_type, f"{source}.{key}.url")
                elif isinstance(payload, str):
                    add_url_or_data_url(payload, mime_type, f"{source}.{key}")
            url = node.get("url")
            if isinstance(url, str) and looks_like_image_url(url):
                add_url_or_data_url(url, mime_type, f"{source}.url")
            for key, child in node.items():
                walk(child, f"{source}.{key}")
        elif isinstance(node, list):
            for idx, child in enumerate(node):
                walk(child, f"{source}[{idx}]")
        elif isinstance(node, str):
            for match in re.finditer(r"data:(image/[^;,]+);base64,([A-Za-z0-9+/=\\s]+)", node):
                add("data_url", match.group(0).replace("\n", ""), match.group(1), source)
            for match in re.finditer(r"https?://[^\s)>\]\"']+", node):
                url = match.group(0)
                if looks_like_image_url(url):
                    add("url", url, "image/png", source)

    def add_url_or_data_url(url: str, mime_type: str, source: str) -> None:
        if url.startswith("data:image/"):
            add("data_url", url, mime_type, source)
        elif looks_like_image_url(url):
            add("url", url, mime_type, source)

    walk(value)
    return images


def looks_like_image_url(value: str) -> bool:
    if value.startswith("data:image/"):
        return True
    lowered = value.lower().split("?", 1)[0]
    return lowered.startswith(("http://", "https://")) and lowered.endswith((".png", ".jpg", ".jpeg", ".webp"))


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


def extract_image_b64(item: dict[str, Any]) -> str | None:
    for key in ("b64_json", "b64", "base64", "image", "image_data"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    return None


def redact_large_images(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if key in {
                "b64",
                "b64_json",
                "base64",
                "data",
                "image",
                "image_data",
                "partial_image_b64",
                "result",
            } and isinstance(item, str) and len(item) > 128:
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


def download_image_to_path(url: str, path: Path, timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS) -> Path:
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        response = client.get(url)
    if response.is_error:
        raise RuntimeError(f"Failed to download image URL {url}: HTTP {response.status_code}")
    mime_type = response.headers.get("content-type", "").split(";")[0]
    ext = extension_for_mime(mime_type) if mime_type else path.suffix.lstrip(".")
    if ext and ext != "bin" and path.suffix.lower() != f".{ext}".lower():
        path = path.with_suffix(f".{ext}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return path


def resolve_output_path(
    output_path: str | None,
    output_dir: str,
    stem: str,
    extension: str,
    index: int | None = None,
) -> Path:
    ext = normalize_output_extension(extension)
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
