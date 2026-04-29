# Image Generator MCP Tool Guide

This document is for MCP clients and AI agents. Use MCP tool calls, not plain HTTP POSTs to this page.

## General Rules

- Prefer the minimal tool for the provider and task.
- Do not pass `api_key` or `base_url` unless the user explicitly provides overrides. The server can read them from environment variables.
- Generated files are saved on the MCP server machine. Return `saved_images[*].path` to the user when present.
- A successful result usually contains `status: "saved"` and `saved_images`.
- If `status` is `download_failed` or `no_image_data`, explain that the upstream request completed but no local image file was saved.

## Image Inputs

For edit/reference-image tools, pass `images` as a list of objects. Use absolute paths when possible.

```json
{"path": "/absolute/path/image.png"}
```

```json
{"url": "https://example.com/image.png"}
```

```json
{"data_url": "data:image/png;base64,..."}
```

```json
{"base64": "...", "mime_type": "image/png", "filename": "image.png"}
```

## OpenAI-Compatible Image API

Use `image_generate` for text-to-image.

Minimal arguments:

```json
{
  "prompt": "A clean product render of a ceramic coffee dripper"
}
```

Common optional arguments:

```json
{
  "prompt": "A clean product render of a ceramic coffee dripper",
  "model": "gpt-image-2",
  "size": "1536x1024",
  "quality": "medium",
  "output_format": "png",
  "output_dir": "outputs"
}
```

Use `image_edit` for image editing or reference-image composition.

```json
{
  "prompt": "Keep the product shape, change the material to brushed steel",
  "images": [{"path": "/absolute/path/dripper.png"}],
  "size": "1536x1024",
  "quality": "medium"
}
```

Use `image_generate_stream` or `image_edit_stream` only when partial image events are useful.

## Gemini

Use `gemini_image` for Gemini text-to-image and image editing.

Text-to-image:

```json
{
  "prompt": "A clean product render of a ceramic coffee dripper",
  "model": "gemini-3-pro-image-preview",
  "aspect_ratio": "16:9",
  "image_size": "2K"
}
```

Image editing/reference image:

```json
{
  "prompt": "Keep the layout, make the scene look like golden hour",
  "images": [{"path": "/absolute/path/scene.png"}],
  "aspect_ratio": "16:9"
}
```

Use `auth_mode: "bearer"` only for OpenAI-compatible proxy providers that expect Bearer auth. For Google Gemini endpoints, leave `auth_mode` as `auto`.

## xAI

Use `xai_image_generate` for xAI text-to-image.

```json
{
  "prompt": "A cinematic poster of a neon ramen shop in heavy rain",
  "model": "grok-imagine-image",
  "aspect_ratio": "16:9",
  "resolution": "2k"
}
```

Use `xai_image_edit` for xAI image editing.

```json
{
  "prompt": "Replace the background with a bright studio backdrop",
  "images": [{"path": "/absolute/path/portrait.jpg"}],
  "model": "grok-imagine-image"
}
```

For OpenAI-compatible proxies that require multipart edits, set:

```json
{
  "prompt": "Add a person to the foreground and preserve the original scene",
  "images": [{"path": "/absolute/path/city.png"}],
  "model": "grok-4.2-image",
  "request_format": "multipart"
}
```

## Responses API

Use `responses_image` when the provider supports OpenAI Responses image generation.

```json
{
  "prompt": "Design a compact camera with a transparent shell",
  "action": "generate",
  "quality": "high"
}
```

For multi-turn:

```json
{
  "prompt": "Now make it look realistic",
  "previous_response_id": "resp_...",
  "action": "auto"
}
```

## Generic Chat/Responses Image Extraction

Use these only when a provider returns images through non-standard chat or responses payloads:

- `chat_image`
- `responses_direct_image`

Example:

```json
{
  "prompt": "Add a person to the foreground and preserve the original scene",
  "images": [{"path": "/absolute/path/cyberpunk-city.png"}],
  "model": "grok-4.2-image",
  "max_tokens": 512
}
```

## File Uploads And Masks

Use `upload_file` when a provider requires a `file_id`.

```json
{
  "file": {"path": "/absolute/path/mask.png"},
  "purpose": "vision"
}
```

Use `add_mask_alpha` to convert a black/white mask to a PNG mask with alpha.

```json
{
  "mask": {"path": "/absolute/path/mask.png"},
  "output_dir": "outputs"
}
```

## Provider Overrides

Only pass these when the user asks for a specific proxy or key:

```json
{
  "prompt": "A product photo on a white background",
  "api_key": "sk-...",
  "base_url": "https://api.openai.com/v1"
}
```

