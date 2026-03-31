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
