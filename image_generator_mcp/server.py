from __future__ import annotations

import argparse
import os
from io import BytesIO
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from PIL import Image

from .constants import DEFAULT_TIMEOUT_SECONDS
from .http_client import auth_headers, gemini_headers, post_json, post_multipart, post_multipart_raw, post_stream_json, post_stream_multipart
from .inputs import image_input_to_file, image_inputs_to_files, resolve_binary_input
from .payloads import build_chat_messages, build_gemini_generate_content_payload, build_response_image_tool, build_response_input, build_xai_edit_image_payload
from .storage import (
    resolve_output_path,
    save_gemini_result,
    save_generic_image_result,
    save_image_api_result,
    save_responses_result,
    save_stream_images,
    summarize_events,
)
from .validation import (
    compact,
    normalize_gemini_base_url,
    normalize_output_format,
    normalize_size,
    normalize_xai_aspect_ratio,
    normalize_xai_resolution,
    normalize_xai_response_format,
    validate_common,
    validate_partial_images,
)


def resolve_api_key(api_key: str | None, base_url: str, provider: str | None = None) -> str:
    explicit = (api_key or "").strip()
    if explicit:
        return explicit

    normalized_provider = (provider or "").lower()
    lowered_base_url = (base_url or "").lower()
    names = ["IMAGE_GENERATOR_API_KEY"]
    if normalized_provider == "gemini" or "googleapis.com" in lowered_base_url:
        names.extend(["GEMINI_API_KEY", "GOOGLE_API_KEY"])
    elif normalized_provider == "xai" or "api.x.ai" in lowered_base_url:
        names.append("XAI_API_KEY")
    else:
        names.append("OPENAI_API_KEY")
    names.append("API_KEY")

    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value

    raise ValueError(
        "api_key was not provided and no matching environment variable is set. "
        f"Tried: {', '.join(dict.fromkeys(names))}."
    )


def create_mcp(host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    mcp = FastMCP(
        "image-generator-mcp",
        instructions=(
            "Generate, edit, stream, and save image outputs across OpenAI, Gemini, and xAI. "
            "Tools accept api_key explicitly, or read provider keys from environment variables."
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
        timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
        output_path: str | None = None,
        output_dir: str = "outputs",
        api_key: str | None = None,
    ) -> dict[str, Any]:
        validate_common(prompt, size, quality, output_format, background, moderation, n)
        api_key = resolve_api_key(api_key, base_url)
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
        body = await post_json(api_key, base_url, "/images/generations", payload, timeout_seconds=timeout_seconds)
        return save_image_api_result(body, output_path, output_dir, payload["output_format"], model, timeout_seconds)

    @mcp.tool(
        description=(
            "Stream Image API generation with partial_images 0-3. "
            "Saves partial and final images when the upstream API emits them."
        )
    )
    async def image_generate_stream(
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
        timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
        output_dir: str = "outputs",
        filename_prefix: str | None = None,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        validate_common(prompt, size, quality, output_format, background, moderation, 1)
        api_key = resolve_api_key(api_key, base_url)
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
        events = await post_stream_json(api_key, base_url, "/images/generations", payload, timeout_seconds=timeout_seconds)
        saved = save_stream_images(events, output_dir, filename_prefix or model, fmt, "image_generation")
        return {"events": summarize_events(events), "saved_images": saved}

    @mcp.tool(
        description=(
            "Edit or compose images with the Image API /images/edits. "
            "Each image and mask can be provided as path, url, base64, data_url, or binary_base64."
        )
    )
    async def image_edit(
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
        timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
        output_path: str | None = None,
        output_dir: str = "outputs",
        api_key: str | None = None,
    ) -> dict[str, Any]:
        if not images:
            raise ValueError("images must include at least one image input.")
        api_key = resolve_api_key(api_key, base_url)
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
        body = await post_multipart(api_key, base_url, "/images/edits", data, files, timeout_seconds=timeout_seconds)
        return save_image_api_result(body, output_path, output_dir, fmt, model, timeout_seconds)

    @mcp.tool(
        description=(
            "Stream Image API edits with partial_images 0-3. "
            "Image and mask inputs support path, url, base64, data_url, and binary_base64."
        )
    )
    async def image_edit_stream(
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
        timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
        output_dir: str = "outputs",
        filename_prefix: str | None = None,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        if not images:
            raise ValueError("images must include at least one image input.")
        api_key = resolve_api_key(api_key, base_url)
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
        events = await post_stream_multipart(api_key, base_url, "/images/edits", data, files, timeout_seconds=timeout_seconds)
        saved = save_stream_images(events, output_dir, filename_prefix or model, fmt, "image_generation")
        return {"events": summarize_events(events), "saved_images": saved}

    @mcp.tool(
        description=(
            "Call the Responses API image_generation tool. Supports multi-turn via previous_response_id, "
            "action auto/generate/edit, file IDs, image URLs/data URLs, masks, and local saving."
        )
    )
    async def responses_image(
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
        timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
        output_dir: str = "outputs",
        filename_prefix: str | None = None,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        api_key = resolve_api_key(api_key, base_url)
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
        body = await post_json(api_key, base_url, "/responses", payload, timeout_seconds=timeout_seconds)
        return save_responses_result(body, output_dir, filename_prefix or model, output_format or "png")

    @mcp.tool(
        description=(
            "Stream the Responses API image_generation tool. Saves partial image events and final image call results."
        )
    )
    async def responses_image_stream(
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
        timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
        output_dir: str = "outputs",
        filename_prefix: str | None = None,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        api_key = resolve_api_key(api_key, base_url)
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
        events = await post_stream_json(api_key, base_url, "/responses", payload, timeout_seconds=timeout_seconds)
        fmt = normalize_output_format(output_format or "png")
        saved = save_stream_images(
            events,
            output_dir,
            filename_prefix or model,
            fmt,
            "response.image_generation_call",
        )
        return {"events": summarize_events(events), "saved_images": saved}

    @mcp.tool(
        description=(
            "Generate or edit images with Google's Gemini generateContent API. "
            "Supports text prompts, optional input images, aspect_ratio/image_size config, and local saving."
        )
    )
    async def gemini_image(
        prompt: str,
        images: list[dict[str, Any]] | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        model: str = "gemini-3-pro-image-preview",
        aspect_ratio: str | None = None,
        image_size: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        candidate_count: int | None = None,
        auth_mode: Literal["auto", "google", "bearer"] = "auto",
        timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
        output_path: str | None = None,
        output_dir: str = "outputs",
        filename_prefix: str | None = None,
        raw_payload: dict[str, Any] | None = None,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        if not (prompt or "").strip():
            raise ValueError("prompt is required.")
        api_key = resolve_api_key(api_key, base_url, "gemini")
        payload = raw_payload or await build_gemini_generate_content_payload(
            prompt,
            images,
            aspect_ratio,
            image_size,
            temperature,
            top_p,
            top_k,
            candidate_count,
        )
        path = f"/models/{model}:generateContent"
        body = await post_json(
            api_key,
            base_url,
            path,
            payload,
            base_url_normalizer=normalize_gemini_base_url,
            headers_builder=select_gemini_headers(base_url, auth_mode),
            timeout_seconds=timeout_seconds,
        )
        return save_gemini_result(body, output_path, output_dir, filename_prefix or model, timeout_seconds)

    @mcp.tool(
        description=(
            "Generate images with xAI's OpenAI-compatible /images/generations endpoint. "
            "Supports Grok image models such as grok-imagine-image and local saving."
        )
    )
    async def xai_image_generate(
        prompt: str,
        base_url: str = "https://api.x.ai/v1",
        model: str = "grok-imagine-image",
        n: int = 1,
        response_format: str = "b64_json",
        aspect_ratio: str | None = None,
        resolution: str | None = None,
        user: str | None = None,
        timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
        output_path: str | None = None,
        output_dir: str = "outputs",
        api_key: str | None = None,
    ) -> dict[str, Any]:
        if not (prompt or "").strip():
            raise ValueError("prompt is required.")
        api_key = resolve_api_key(api_key, base_url, "xai")
        if n < 1:
            raise ValueError("n must be at least 1.")
        fmt = normalize_xai_response_format(response_format)
        payload = compact(
            {
                "model": model,
                "prompt": prompt,
                "n": n,
                "response_format": fmt,
                "aspect_ratio": normalize_xai_aspect_ratio(aspect_ratio),
                "resolution": normalize_xai_resolution(resolution),
                "user": user,
            }
        )
        body = await post_json(api_key, base_url, "/images/generations", payload, timeout_seconds=timeout_seconds)
        return save_image_api_result(body, output_path, output_dir, "jpeg", model, timeout_seconds)

    @mcp.tool(
        description=(
            "Edit images with xAI or OpenAI-compatible /images/edits endpoints. "
            "Supports JSON image URL/data URL payloads and multipart/form-data uploads."
        )
    )
    async def xai_image_edit(
        prompt: str,
        images: list[dict[str, Any]],
        base_url: str = "https://api.x.ai/v1",
        model: str = "grok-imagine-image",
        request_format: Literal["json", "multipart"] = "json",
        n: int = 1,
        response_format: str | None = "b64_json",
        output_mime_type: str | None = None,
        size: str | None = None,
        quality: str | None = None,
        output_format: str | None = None,
        output_compression: int | None = None,
        background: str | None = None,
        moderation: str | None = None,
        aspect_ratio: str | None = None,
        resolution: str | None = None,
        timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
        output_path: str | None = None,
        output_dir: str = "outputs",
        api_key: str | None = None,
    ) -> dict[str, Any]:
        if not (prompt or "").strip():
            raise ValueError("prompt is required.")
        api_key = resolve_api_key(api_key, base_url, "xai")
        if not images:
            raise ValueError("images must include at least one image input.")
        if len(images) > 5:
            raise ValueError("xAI image editing supports up to 5 input images.")
        if n < 1:
            raise ValueError("n must be at least 1.")
        if request_format not in {"json", "multipart"}:
            raise ValueError("request_format must be json or multipart.")
        if request_format == "multipart":
            fmt = normalize_output_format(output_format or output_mime_type_to_format(output_mime_type) or "png")
            data = compact(
                {
                    "model": model,
                    "prompt": prompt,
                    "n": str(max(1, n)),
                    "response_format": normalize_xai_response_format(response_format) if response_format else None,
                    "output_mime_type": output_mime_type,
                    "size": normalize_size(size) if size else None,
                    "quality": quality,
                    "output_format": fmt,
                    "output_compression": output_compression,
                    "background": background,
                    "moderation": moderation,
                    "aspect_ratio": normalize_xai_aspect_ratio(aspect_ratio),
                    "resolution": normalize_xai_resolution(resolution),
                }
            )
            if quality is not None and quality not in {"auto", "low", "medium", "high"}:
                raise ValueError("Invalid quality. Use auto, low, medium, or high.")
            if background is not None and background not in {"auto", "opaque", "transparent"}:
                raise ValueError("Invalid background. Use auto, opaque, or transparent.")
            if moderation is not None and moderation not in {"auto", "low"}:
                raise ValueError("Invalid moderation. Use auto or low.")
            files = await image_inputs_to_files(images, "image[]")
            body = await post_multipart(api_key, base_url, "/images/edits", data, files, timeout_seconds=timeout_seconds)
            return save_image_api_result(body, output_path, output_dir, fmt, model, timeout_seconds)
        payload = compact(
            {
                "model": model,
                "prompt": prompt,
                "n": n,
                "response_format": normalize_xai_response_format(response_format or "b64_json"),
                "output_mime_type": output_mime_type,
                "aspect_ratio": normalize_xai_aspect_ratio(aspect_ratio),
                "resolution": normalize_xai_resolution(resolution),
            }
        )
        payload.update(await build_xai_edit_image_payload(images))
        body = await post_json(api_key, base_url, "/images/edits", payload, timeout_seconds=timeout_seconds)
        fmt = "jpeg" if output_mime_type == "image/jpeg" else "png"
        return save_image_api_result(body, output_path, output_dir, fmt, model, timeout_seconds)

    @mcp.tool(
        description=(
            "Call an OpenAI-compatible /chat/completions image model and save any image data or image URLs found in the response."
        )
    )
    async def chat_image(
        prompt: str,
        images: list[dict[str, Any]] | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "grok-4.2-image",
        max_tokens: int = 64,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
        output_dir: str = "outputs",
        filename_prefix: str | None = None,
        raw_payload: dict[str, Any] | None = None,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        if not (prompt or "").strip():
            raise ValueError("prompt is required.")
        api_key = resolve_api_key(api_key, base_url)
        payload = raw_payload or compact(
            {
                "model": model,
                "messages": await build_chat_messages(prompt, images),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "response_format": response_format,
            }
        )
        body = await post_json(api_key, base_url, "/chat/completions", payload, timeout_seconds=timeout_seconds)
        return save_generic_image_result(body, output_dir, filename_prefix or model, timeout_seconds=timeout_seconds)

    @mcp.tool(
        description=(
            "Call an OpenAI-compatible /responses image model directly and save any image data or image URLs found in the response."
        )
    )
    async def responses_direct_image(
        prompt: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "grok-4.2-image",
        max_output_tokens: int = 64,
        timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
        output_dir: str = "outputs",
        filename_prefix: str | None = None,
        raw_payload: dict[str, Any] | None = None,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        if not (prompt or "").strip():
            raise ValueError("prompt is required.")
        api_key = resolve_api_key(api_key, base_url)
        payload = raw_payload or {
            "model": model,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
        }
        body = await post_json(api_key, base_url, "/responses", payload, timeout_seconds=timeout_seconds)
        return save_generic_image_result(body, output_dir, filename_prefix or model, timeout_seconds=timeout_seconds)

    @mcp.tool(
        description=(
            "Upload a file to OpenAI and return its file_id for Responses API image inputs or masks. "
            "Input supports path, url, base64, data_url, and binary_base64."
        )
    )
    async def upload_file(
        file: dict[str, Any],
        base_url: str = "https://api.openai.com/v1",
        purpose: str = "vision",
        timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        api_key = resolve_api_key(api_key, base_url)
        binary = await resolve_binary_input(file)
        files = {"file": (binary.filename, binary.data, binary.mime_type)}
        data = {"purpose": purpose}
        body = await post_multipart_raw(api_key, base_url, "/files", data, files, timeout_seconds=timeout_seconds)
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


def output_mime_type_to_format(output_mime_type: str | None) -> str | None:
    if output_mime_type == "image/png":
        return "png"
    if output_mime_type == "image/jpeg":
        return "jpeg"
    if output_mime_type == "image/webp":
        return "webp"
    return None


def select_gemini_headers(base_url: str, auth_mode: str):
    if auth_mode == "google":
        return gemini_headers
    if auth_mode == "bearer":
        return auth_headers
    if auth_mode != "auto":
        raise ValueError("Invalid auth_mode. Use auto, google, or bearer.")
    return gemini_headers if "googleapis.com" in (base_url or "") else auth_headers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Image Generator MCP server.")
    parser.add_argument(
        "--transport",
        "-t",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport to run.",
    )
    parser.add_argument("--host", "-H", default="127.0.0.1", help="Host for SSE/streamable-http.")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port for SSE/streamable-http.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_mcp(host=args.host, port=args.port).run(transport=args.transport)


if __name__ == "__main__":
    main()
