from __future__ import annotations

import re
from typing import Any

from .constants import (
    BACKGROUND_CHOICES,
    FORMAT_CHOICES,
    GEMINI_ASPECT_RATIOS,
    GEMINI_IMAGE_SIZES,
    MODERATION_CHOICES,
    POPULAR_SIZE_CHOICES,
    QUALITY_CHOICES,
    XAI_ASPECT_RATIOS,
    XAI_RESOLUTIONS,
    XAI_RESPONSE_FORMAT_CHOICES,
)


def compact(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None}


def normalize_base_url(base_url: str, default: str = "https://api.openai.com/v1") -> str:
    value = (base_url or default).strip().rstrip("/")
    return value if value.endswith("/v1") else f"{value}/v1"


def normalize_gemini_base_url(base_url: str) -> str:
    value = (base_url or "https://generativelanguage.googleapis.com/v1beta").strip().rstrip("/")
    if re.search(r"/v\d+(?:beta|alpha)?$", value):
        return value
    return f"{value}/v1beta"


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


def normalize_xai_response_format(response_format: str) -> str:
    value = (response_format or "b64_json").strip()
    if value not in XAI_RESPONSE_FORMAT_CHOICES:
        raise ValueError("Invalid response_format. Use b64_json or url.")
    return value


def normalize_gemini_aspect_ratio(aspect_ratio: str | None) -> str | None:
    if aspect_ratio is None:
        return None
    value = aspect_ratio.strip()
    if value.lower() == "auto":
        return None
    if value not in GEMINI_ASPECT_RATIOS:
        raise ValueError("Invalid Gemini aspect_ratio.")
    return value


def normalize_gemini_image_size(image_size: str | None) -> str | None:
    if image_size is None:
        return None
    value = image_size.strip().upper()
    if value == "AUTO":
        return None
    if value not in GEMINI_IMAGE_SIZES:
        raise ValueError("Invalid Gemini image_size. Use 512, 1K, 2K, or 4K.")
    return value


def normalize_xai_aspect_ratio(aspect_ratio: str | None) -> str | None:
    if aspect_ratio is None:
        return None
    value = aspect_ratio.strip()
    if value not in XAI_ASPECT_RATIOS:
        raise ValueError("Invalid xAI aspect_ratio.")
    return value


def normalize_xai_resolution(resolution: str | None) -> str | None:
    if resolution is None:
        return None
    value = resolution.strip().lower()
    if value not in XAI_RESOLUTIONS:
        raise ValueError("Invalid xAI resolution. Use 1k or 2k.")
    return value


def normalize_output_extension(extension: str) -> str:
    value = (extension or "png").strip().lower().lstrip(".")
    if value == "jpg":
        value = "jpeg"
    return value or "png"
