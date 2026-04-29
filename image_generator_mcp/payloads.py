from __future__ import annotations

import base64
from typing import Any

from .constants import ACTION_CHOICES, BACKGROUND_CHOICES, MODERATION_CHOICES, QUALITY_CHOICES
from .inputs import resolve_binary_input
from .validation import (
    compact,
    normalize_gemini_aspect_ratio,
    normalize_gemini_image_size,
    normalize_output_format,
    normalize_size,
)


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


async def build_chat_messages(prompt: str, images: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if not images:
        return [{"role": "user", "content": prompt}]
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for item in images:
        content.append(await chat_image_url_content(item))
    return [{"role": "user", "content": content}]


async def chat_image_url_content(item: dict[str, Any]) -> dict[str, Any]:
    if item.get("url"):
        url = str(item["url"])
    elif item.get("data_url"):
        url = str(item["data_url"])
    else:
        binary = await resolve_binary_input(item)
        encoded = base64.b64encode(binary.data).decode("ascii")
        url = f"data:{binary.mime_type};base64,{encoded}"
    return {"type": "image_url", "image_url": {"url": url}}


async def build_gemini_generate_content_payload(
    prompt: str,
    images: list[dict[str, Any]] | None,
    aspect_ratio: str | None,
    image_size: str | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    candidate_count: int | None,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [{"text": prompt}]
    for item in images or []:
        parts.append(await gemini_inline_image_part(item))

    image_config = compact(
        {
            "aspectRatio": normalize_gemini_aspect_ratio(aspect_ratio),
            "imageSize": normalize_gemini_image_size(image_size),
        }
    )
    generation_config = compact(
        {
            "responseModalities": ["TEXT", "IMAGE"],
            "temperature": temperature,
            "topP": top_p,
            "topK": top_k,
            "candidateCount": candidate_count,
            "imageConfig": image_config or None,
        }
    )
    return {"contents": [{"role": "user", "parts": parts}], "generationConfig": generation_config}


async def gemini_inline_image_part(item: dict[str, Any]) -> dict[str, Any]:
    binary = await resolve_binary_input(item)
    return {
        "inlineData": {
            "mimeType": binary.mime_type,
            "data": base64.b64encode(binary.data).decode("ascii"),
        }
    }


async def build_xai_image_inputs(images: list[dict[str, Any]]) -> list[str]:
    result = []
    for item in images:
        if item.get("url"):
            result.append(str(item["url"]))
            continue
        if item.get("data_url"):
            result.append(str(item["data_url"]))
            continue
        binary = await resolve_binary_input(item)
        encoded = base64.b64encode(binary.data).decode("ascii")
        result.append(f"data:{binary.mime_type};base64,{encoded}")
    return result


async def build_xai_edit_image_payload(images: list[dict[str, Any]]) -> dict[str, Any]:
    image_values = await build_xai_image_inputs(images)
    if len(image_values) == 1:
        return {"image": {"url": image_values[0]}}
    image_objects = [{"type": "image_url", "url": value} for value in image_values]
    return {"images": image_objects}


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
