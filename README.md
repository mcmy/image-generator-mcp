# GPT Image Generator MCP

MCP server for GPT Image workflows from `doc.md`. The server does not store API
tokens; pass `api_key` on every tool call.

## Install

```bash
uv sync
```

## Run

Stdio:

```bash
uv run python main.py
```

SSE:

```bash
uv run python main.py --transport sse --host 127.0.0.1 --port 8000
```

Streamable HTTP:

```bash
uv run python main.py --transport streamable-http --host 127.0.0.1 --port 8000
```

FastMCP defaults:

- SSE endpoint: `http://127.0.0.1:8000/sse`
- SSE messages: `http://127.0.0.1:8000/messages/`
- Streamable HTTP endpoint: `http://127.0.0.1:8000/mcp`

## Tools

- `image_generate`: Image API text-to-image generation.
- `image_generate_stream`: Image API streaming generation with `partial_images`.
- `image_edit`: Image API edits, masks, and reference-image composition.
- `image_edit_stream`: Streaming Image API edits.
- `responses_image`: Responses API `image_generation` tool, including multi-turn via `previous_response_id`.
- `responses_image_stream`: Streaming Responses API image generation.
- `upload_file`: Upload a local/remote/base64 file and return an OpenAI `file_id`.
- `add_mask_alpha`: Convert a black/white mask to a PNG mask with alpha.

## Image Inputs

Image and file inputs are JSON objects. Supported forms:

```json
{"path": "/absolute/or/relative/image.png"}
```

```json
{"url": "https://example.com/image.png"}
```

```json
{"base64": "iVBORw0KGgo...", "mime_type": "image/png", "filename": "image.png"}
```

```json
{"data_url": "data:image/png;base64,iVBORw0KGgo..."}
```

```json
{"binary_base64": "iVBORw0KGgo...", "mime_type": "image/png"}
```

Raw binary bytes are not portable in standard MCP JSON tool arguments, so binary
content is accepted as base64.

Responses API image inputs also support:

```json
{"file_id": "file_..."}
```

For Responses API masks, upload the mask first with `upload_file`, then pass:

```json
{"file_id": "file_..."}
```

## Example Tool Arguments

Generate an image:

```json
{
  "api_key": "sk-...",
  "prompt": "A minimalist tea package design on a white tabletop",
  "size": "1536x1024",
  "quality": "medium",
  "output_format": "png"
}
```

Edit with image references:

```json
{
  "api_key": "sk-...",
  "prompt": "Create a photorealistic gift basket containing all reference items",
  "images": [
    {"path": "body-lotion.png"},
    {"url": "https://example.com/soap.png"},
    {"base64": "iVBORw0KGgo...", "mime_type": "image/png"}
  ],
  "quality": "high"
}
```

Responses API multi-turn:

```json
{
  "api_key": "sk-...",
  "prompt": "Now make it look realistic",
  "previous_response_id": "resp_...",
  "action": "auto",
  "quality": "high"
}
```

## Options

Common generation/edit options:

- `model`: defaults to `gpt-image-2` for Image API, `gpt-5.5` for Responses API.
- `base_url`: defaults to `https://api.openai.com/v1`.
- `size`: `auto`, popular sizes like `1024x1024`, or valid `WIDTHxHEIGHT`.
- `quality`: `auto`, `low`, `medium`, `high`.
- `output_format`: `png`, `jpeg`, `jpg`, `webp`.
- `output_compression`: `0` to `100`, for JPEG/WebP.
- `background`: `auto`, `opaque`, `transparent` where supported by the model.
- `moderation`: `auto` or `low`.
- `partial_images`: `0` to `3` for streaming tools.

Generated files are saved under `outputs/` by default. Use `output_path`,
`output_dir`, or `filename_prefix` to control filenames.
