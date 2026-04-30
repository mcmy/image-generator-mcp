# Image Generator MCP

MCP server for image generation and editing workflows across OpenAI-compatible
Image APIs, Gemini `generateContent`, and xAI Grok image endpoints. The server
does not store API tokens. Configure provider keys in the MCP server environment,
or pass `api_key` on individual tool calls when needed.

## Install

```bash
uv sync
```

Install from PyPI after publishing:

```bash
uvx image-generator-mcp
```

## Run

Set the provider key and optional base URL before starting the server:

```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="AIza..."
export XAI_API_KEY="xai-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
export GEMINI_BASE_URL="https://generativelanguage.googleapis.com/v1beta"
export XAI_BASE_URL="https://api.x.ai/v1"
```

The generic `IMAGE_GENERATOR_API_KEY` and `API_KEY` environment variables are
also accepted for keys. The generic `IMAGE_GENERATOR_BASE_URL` and `BASE_URL`
environment variables are also accepted for URLs. Explicit `api_key` and
`base_url` tool arguments override environment variables.

Stdio:

```bash
uv run python main.py
```

SSE:

```bash
uv run python main.py --transport sse --host 127.0.0.1 --port 8000
uv run python main.py -t sse -H 127.0.0.1 -p 8000
```

Streamable HTTP:

```bash
uv run python main.py --transport streamable-http --host 127.0.0.1 --port 8000
uv run python main.py -t streamable-http -H 127.0.0.1 -p 8000
uv run python main.py -t mcp -H 127.0.0.1 -p 8000
```

Installed console script:

```bash
image-generator-mcp
image-generator-mcp --transport sse --host 127.0.0.1 --port 8000
image-generator-mcp --transport streamable-http --host 127.0.0.1 --port 8000
image-generator-mcp -t streamable-http -H 127.0.0.1 -p 8000
image-generator-mcp -t mcp -H 127.0.0.1 -p 8000
```

FastMCP defaults:

- SSE endpoint: `http://127.0.0.1:8000/sse`
- SSE messages: `http://127.0.0.1:8000/messages/`
- Streamable HTTP endpoint: `http://127.0.0.1:8000/mcp`
- Human-readable docs: `http://127.0.0.1:8000/`, `http://127.0.0.1:8000/docs`, `http://127.0.0.1:8000/doc`

## MCP Client Config

For most AI clients, prefer stdio and put secrets in `env` so the model does not
need to provide them as tool arguments:

```json
{
  "mcpServers": {
    "image-generator": {
      "command": "uv",
      "args": ["run", "python", "main.py"],
      "cwd": "/Users/canny/project/python/image-generator-mcp",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "GEMINI_API_KEY": "AIza...",
        "GEMINI_BASE_URL": "https://generativelanguage.googleapis.com/v1beta",
        "XAI_API_KEY": "xai-...",
        "XAI_BASE_URL": "https://api.x.ai/v1"
      }
    }
  }
}
```

If your client only supports remote MCP URLs, run Streamable HTTP and configure
the client to use `http://127.0.0.1:8000/mcp`. Use `--host 0.0.0.0` only when
the server must be reachable from another machine.

## Publish To PyPI

Build the package:

```bash
uv build
```

Publish to TestPyPI first:

```bash
UV_PUBLISH_TOKEN="<testpypi-token>" ./scripts/publish-pypi.sh test.pypi
```

Publish to PyPI:

```bash
UV_PUBLISH_TOKEN="<pypi-token>" ./scripts/publish-pypi.sh pypi
```

The script removes `dist/`, builds a fresh source distribution and wheel, then
uses `uv publish`. Keep PyPI tokens in environment variables or your CI secret
store; do not commit them.

## Tools

For AI clients, the lowest-friction tools are:

- OpenAI-compatible text-to-image: `image_generate` with only `prompt`.
- OpenAI-compatible image edit: `image_edit` with `prompt` and `images`.
- Gemini generation/editing: `gemini_image` with `prompt`, plus optional `images`.
- xAI generation: `xai_image_generate` with only `prompt`.
- xAI editing: `xai_image_edit` with `prompt` and `images`.

Use the streaming and direct/chat variants only when a provider specifically
requires those API shapes.

- `image_generate`: OpenAI-compatible Image API text-to-image generation via `/images/generations`.
- `image_generate_stream`: OpenAI-compatible streaming generation with `partial_images`.
- `image_edit`: OpenAI-compatible Image API edits, masks, and reference-image composition via `/images/edits`.
- `image_edit_stream`: OpenAI-compatible streaming Image API edits.
- `responses_image`: OpenAI Responses API `image_generation` tool, including multi-turn via `previous_response_id`.
- `responses_image_stream`: Streaming Responses API image generation.
- `gemini_image`: Gemini image generation/editing through `/v1beta/models/{model}:generateContent`.
- `xai_image_generate`: xAI image generation through `/v1/images/generations`.
- `xai_image_edit`: xAI/OpenAI-compatible image editing through `/v1/images/edits`, with JSON or multipart request formats.
- `chat_image`: OpenAI-compatible `/chat/completions` image model call with generic image extraction; supports optional image inputs for multimodal image edits.
- `responses_direct_image`: OpenAI-compatible `/responses` image model call with generic image extraction.
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

OpenAI-compatible image generation:

```json
{
  "api_key": "sk-...",
  "prompt": "A minimalist tea package design on a white tabletop",
  "size": "1536x1024",
  "quality": "medium",
  "output_format": "png"
}
```

Gemini image generation:

```json
{
  "api_key": "AIza...",
  "prompt": "A clean product render of a ceramic coffee dripper",
  "model": "gemini-3-pro-image-preview",
  "aspect_ratio": "16:9",
  "image_size": "2K"
}
```

Gemini image editing with a reference image:

```json
{
  "api_key": "AIza...",
  "prompt": "Keep the product shape, change the material to brushed steel",
  "images": [{"path": "dripper.png"}]
}
```

xAI Grok image generation:

```json
{
  "api_key": "xai-...",
  "prompt": "A cinematic poster of a neon ramen shop in heavy rain",
  "model": "grok-imagine-image",
  "aspect_ratio": "16:9",
  "resolution": "2k"
}
```

xAI image editing:

```json
{
  "api_key": "xai-...",
  "prompt": "Replace the background with a bright studio backdrop",
  "images": [{"path": "portrait.jpg"}],
  "model": "grok-imagine-image"
}
```

OpenAI-compatible multipart image editing with a Grok image model:

```json
{
  "api_key": "sk-...",
  "base_url": "https://api.vectorengine.ai/",
  "prompt": "Add a person to the foreground and preserve the original scene",
  "images": [{"path": "cyberpunk-city.png"}],
  "model": "grok-4.2-image",
  "request_format": "multipart"
}
```

Chat Completions image editing with a multimodal Grok image model:

```json
{
  "api_key": "sk-...",
  "base_url": "https://api.vectorengine.ai/",
  "prompt": "Add a person to the foreground and preserve the original scene",
  "images": [{"path": "cyberpunk-city.png"}],
  "model": "grok-4.2-image",
  "max_tokens": 512
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

OpenAI-compatible generation/edit options:

- `model`: defaults to `gpt-image-2` for Image API, `gpt-5.5` for Responses API.
- `base_url`: defaults to `OPENAI_BASE_URL`, `IMAGE_GENERATOR_BASE_URL`, `BASE_URL`, or `https://api.openai.com/v1`.
- `size`: `auto`, popular sizes like `1024x1024`, or valid `WIDTHxHEIGHT`.
- `quality`: `auto`, `low`, `medium`, `high`.
- `output_format`: `png`, `jpeg`, `jpg`, `webp`.
- `output_compression`: `0` to `100`, for JPEG/WebP.
- `background`: `auto`, `opaque`, `transparent` where supported by the model.
- `moderation`: `auto` or `low`.
- `partial_images`: `0` to `3` for streaming tools.

Gemini options:

- `base_url`: defaults to `GEMINI_BASE_URL`, `GOOGLE_BASE_URL`, `IMAGE_GENERATOR_BASE_URL`, `BASE_URL`, or `https://generativelanguage.googleapis.com/v1beta`.
- `model`: defaults to `gemini-3-pro-image-preview`.
- `aspect_ratio`: `auto`, `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, and other Gemini-supported ratios.
- `image_size`: `512`, `1K`, `2K`, or `4K`.
- `raw_payload`: optional full `generateContent` JSON override for advanced requests.

xAI options:

- `base_url`: defaults to `XAI_BASE_URL`, `IMAGE_GENERATOR_BASE_URL`, `BASE_URL`, or `https://api.x.ai/v1`.
- `model`: defaults to `grok-imagine-image`; override if your account or proxy uses another image model name.
- `request_format` on `xai_image_edit`: `json` for xAI-style image URL/data URL payloads, or `multipart` for OpenAI-compatible file upload edits.
- `response_format`: `b64_json` or `url`.
- `aspect_ratio`: common ratios including `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `2:3`, `3:2`, `2:1`, `1:2`, `19.5:9`, `9:19.5`, `20:9`, and `9:20`.
- `resolution`: `1k` or `2k`.

Chat image options:

- `base_url`: defaults to `https://api.openai.com/v1`.
- `model`: defaults to `grok-4.2-image`; override for any OpenAI-compatible image-capable chat model.
- `images`: optional image inputs for multimodal generation/editing. Supported forms are the same as other image inputs: `path`, `url`, `data_url`, `base64`, and `binary_base64`.
- `raw_payload`: optional full `/chat/completions` JSON override for provider-specific message formats.

## Model Interface Matrix

The MCP intentionally exposes multiple tools for providers and proxies that route
the same image model through different API shapes. If a provider supports more
than one route, each usable route is listed.

| Model family / model | Generation tool | Generation endpoint | Editing tool | Editing endpoint | Request format / notes |
| --- | --- | --- | --- | --- | --- |
| `gpt-image-2` and OpenAI-compatible Image API models | `image_generate` | `POST {base_url}/images/generations` | `image_edit` | `POST {base_url}/images/edits` | JSON for generation, `multipart/form-data` for edits. Supports masks through `image_edit`. |
| `gpt-image-2` streaming | `image_generate_stream` | `POST {base_url}/images/generations` | `image_edit_stream` | `POST {base_url}/images/edits` | Same Image API routes with `stream=true` and optional `partial_images`. |
| Responses API image generation models | `responses_image` | `POST {base_url}/responses` | `responses_image` | `POST {base_url}/responses` | Uses the Responses `image_generation` tool. Set `action` to `generate`, `edit`, or `auto`. |
| Responses API image generation streaming | `responses_image_stream` | `POST {base_url}/responses` | `responses_image_stream` | `POST {base_url}/responses` | Same Responses image tool with `stream=true`. |
| `gemini-3.1-flash-image-preview`, `gemini-3-pro-image-preview`, and Gemini image models | `gemini_image` | `POST {base_url}/models/{model}:generateContent` | `gemini_image` | `POST {base_url}/models/{model}:generateContent` | `generateContent` with text parts for generation; add `images` for editing/reference-image composition. Use `auth_mode="bearer"` for OpenAI-compatible proxies such as VectorEngine, or Google API key headers for Google endpoints. |
| `grok-imagine-image` and xAI-compatible image models | `xai_image_generate` | `POST {base_url}/images/generations` | `xai_image_edit` | `POST {base_url}/images/edits` | Default edit `request_format="json"` sends image URLs/data URLs in a JSON body, matching xAI-style payloads. |
| `grok-4.2-image` on VectorEngine or OpenAI-compatible Image API proxies | `image_generate` | `POST {base_url}/images/generations` | `image_edit` or `xai_image_edit` with `request_format="multipart"` | `POST {base_url}/images/edits` | Multipart edit path is required on VectorEngine; the JSON xAI-style edit path can fail with `Content-Type isn't multipart/form-data`. |
| `grok-4.2-image` through Chat Completions | `chat_image` | `POST {base_url}/chat/completions` | `chat_image` with `images` | `POST {base_url}/chat/completions` | Returns image URLs or embedded image data inside the chat response; MCP extracts and saves any images it finds. |
| Generic OpenAI-compatible `/responses` image models | `responses_direct_image` | `POST {base_url}/responses` | `responses_direct_image` with provider-specific `raw_payload` | `POST {base_url}/responses` | Generic extraction path for providers that return image URLs/base64 in non-OpenAI-standard response shapes. |

Generated files are saved under `outputs/` relative to the MCP process current
working directory by default; they are not written inside the Python package.
Use `output_path`, `output_dir`, or `filename_prefix` to control filenames.
