# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

llm-vertex is an [LLM](https://llm.datasette.io/) plugin that provides access to Google's Gemini models via Vertex AI. It registers models with the `vertex/` prefix (e.g., `vertex/gemini-2.5-flash`) and supports both sync (`Vertex`) and async (`AsyncVertex`) execution.

## Commands

```bash
# Install with test dependencies
llm install -e '.[test]'

# Run all tests
pytest

# Run a single test
pytest tests/test_vertex.py::test_prompt

# Run tests matching a keyword
pytest tests/test_vertex.py -k "schema"
```

Tests use `pytest-recording` (VCR cassettes in `tests/cassettes/`) to replay HTTP responses. The `@pytest.mark.vcr` decorator enables cassette playback. Async tests use `pytest-asyncio` with `asyncio_mode = "strict"`.

## Architecture

The entire plugin is a single file: `llm_vertex.py`. There is no package directory.

**Key classes:**
- `_SharedGemini` — mixin with all shared logic: options, request building, streaming response parsing, credential management, attachment handling, thinking trace/tool call preservation
- `Vertex(_SharedGemini, llm.Model)` — sync model using `httpx.stream`
- `AsyncVertex(_SharedGemini, llm.AsyncModel)` — async model using `httpx.AsyncClient`
- `VertexEmbeddingModel(llm.EmbeddingModel)` — embedding models using Vertex AI predict endpoint

**Model registration:** `register_models()` iterates a hardcoded list of model IDs and registers both `Vertex` and `AsyncVertex` instances with capability flags (`can_google_search`, `can_thinking_budget`, `thinking_levels`, etc.). Options classes are built dynamically via Pydantic `create_model` based on these flags.

**Authentication flow** (in priority order): OAuth2/ADC credentials > API key. Credentials are cached on the class. Project/region come from env vars > llm config > credential defaults.

**Streaming:** Uses `ijson` for incremental JSON parsing of the Vertex AI streaming response. `_merge_streaming_parts()` handles merging text deltas and preserving `thoughtSignature` fields across streaming events.

**Schema handling:** `cleanup_schema()` strips unsupported JSON Schema keys for Gemini and resolves `$ref`/`$defs` references (with recursive schema detection). Applied to both structured output schemas and tool input schemas.

**Region overrides:** Models in `MODEL_REGION_REQUIREMENTS` (Gemini 3+) automatically use the global endpoint regardless of user config.

**CLI commands:** Registered via `register_commands()` hook — `llm vertex set-project`, `set-region`, `set-credentials`, `set-thinking-level`, `list-regions`, `config`.

## Adding New Models

Adding a model requires updating multiple data structures in `llm_vertex.py`. For each new model ID:

1. Add to the model list in `register_models()` (registers both sync and async)
2. Add to `GOOGLE_SEARCH_MODELS` if it supports Google Search grounding (most Gemini models do; Gemma models don't)
3. Add to `THINKING_BUDGET_MODELS` if it supports `thinking_budget` (integer token budget — Gemini 2.5 series)
4. Add to `MODEL_THINKING_LEVELS` if it supports `thinking_level` (named levels like "low"/"high" — Gemini 3+ series)
5. Add to `MODEL_REGION_REQUIREMENTS` if it requires a specific endpoint (e.g., `"global"` for Gemini 3+ models)
6. Add to `NO_VISION_MODELS` if it lacks vision/attachment support
7. Add to `NO_MEDIA_RESOLUTION_MODELS` if it doesn't support the `mediaResolution` parameter (Gemma 3 models)

The `can_schema` flag is computed inline: disabled for models with `"flash-thinking"` or `"gemma-3"` in the ID.

This plugin mirrors [llm-gemini](https://github.com/simonw/llm-gemini) (Google AI API). When adding models, check llm-gemini for capability flags as a reference. The Vertex AI model list and capabilities can be verified at https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models.
