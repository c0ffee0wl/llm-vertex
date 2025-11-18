# llm-vertex

[![PyPI](https://img.shields.io/pypi/v/llm-vertex.svg)](https://pypi.org/project/llm-vertex/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-gemini-vertex?include_prereleases&label=changelog)](https://github.com/simonw/llm-gemini-vertex/releases)
[![Tests](https://github.com/simonw/llm-gemini-vertex/workflows/Test/badge.svg)](https://github.com/simonw/llm-gemini-vertex/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-gemini-vertex/blob/main/LICENSE)

Access Google's Gemini models via Vertex AI for enterprise use

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-vertex
```

## Authentication Setup

This plugin uses Google Cloud Vertex AI, which supports three authentication methods:

### Option 1: API Key (Recommended for Testing Only)

**Fastest setup**, but recommended for testing only:

```bash
# Set via llm keys command
llm keys set vertex
# Or via environment variable
export GOOGLE_CLOUD_API_KEY="YOUR_API_KEY"
```

Get your API key from the [Google Cloud Console](https://console.cloud.google.com/apis/credentials).

**Note:** Vertex AI API keys are different from Google AI Studio keys. Make sure to create a Vertex AI-compatible API key in your GCP project. API keys are convenient for development and testing but not recommended for production. For production, use Application Default Credentials (Option 2).

### Option 2: Application Default Credentials (Recommended for Production)

If you're already using Google Cloud, authenticate with:

```bash
gcloud auth application-default login
```

This sets up Application Default Credentials (ADC) that the plugin will automatically use.

### Option 3: Service Account JSON File

1. Create a service account in your GCP project with Vertex AI User permissions
2. Download the JSON key file
3. Set the environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

Or configure it via the plugin:

```bash
llm vertex set-credentials /path/to/service-account.json
```

## Configuration

### Set Your GCP Project ID

```bash
# Via environment variable
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Or via plugin config
llm vertex set-project your-project-id
```

### Set Your Region (Optional)

The plugin defaults to the `global` endpoint. However, the global endpoint has important limitations:

- ⚠️ **Does not support** tuning, batch prediction, or RAG corpus creation
- ⚠️ **Does not guarantee** region-specific ML processing
- ⚠️ **Does not provide** data residency compliance

For production use or if you need specific features, use a regional endpoint:

```bash
# Via environment variable
export GOOGLE_CLOUD_REGION="us-central1"

# Or via plugin config
llm vertex set-region us-central1
```

To see all available regions:

```bash
llm vertex list-regions
```

**Available regions include:**
- **United States**: `us-central1`, `us-east1`, `us-east4`, `us-east5`, `us-south1`, `us-west1`, `us-west4`
- **Canada**: `northamerica-northeast1`
- **South America**: `southamerica-east1`
- **Europe**: `europe-west1`, `europe-west2`, `europe-west3`, `europe-west4`, `europe-west6`, `europe-west8`, `europe-west9`, `europe-north1`, `europe-southwest1`, `europe-central2`
- **Asia Pacific**: `asia-east1`, `asia-northeast1`, `asia-northeast3`, `asia-southeast1`, `asia-south1`, `australia-southeast1`, `australia-southeast2`
- **Middle East**: `me-central1`, `me-central2`, `me-west1`

For the latest region availability and model-specific regional support, see the [official Vertex AI locations documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations).

### View Current Configuration

```bash
llm vertex config
```

## Usage

Now run the model using `-m vertex/gemini-2.5-flash`, for example:

```bash
llm -m vertex/gemini-2.5-flash "A short joke about a pelican and a walrus"
```

You can set the [default model](https://llm.datasette.io/en/stable/setup.html#setting-a-custom-default-model) to avoid the extra `-m` option:

```bash
llm models default vertex/gemini-2.5-flash
llm "A joke about a pelican and a walrus"
```

## Available models

All Gemini models are available through Vertex AI:

### Gemini 3 (Latest - Preview)

- `vertex/gemini-3-pro-preview`: Gemini 3 Pro preview (global region only)
- `vertex/gemini-3-pro-preview-11-2025`: Gemini 3 Pro November 2025 version (global region only)
- `vertex/gemini-3-pro-preview-11-2025-thinking`: Gemini 3 Pro with thinking mode (global region only)

**Note:** Gemini 3 models automatically use the global endpoint regardless of your configured region.

### Gemini 2.5 and earlier

- `vertex/gemini-2.5-flash-lite-preview-09-2025`
- `vertex/gemini-2.5-flash-preview-09-2025`
- `vertex/gemini-flash-lite-latest`: Latest Gemini Flash Lite
- `vertex/gemini-flash-latest`: Latest Gemini Flash
- `vertex/gemini-2.5-flash-lite`: Gemini 2.5 Flash Lite
- `vertex/gemini-2.5-pro`: Gemini 2.5 Pro
- `vertex/gemini-2.5-flash`: Gemini 2.5 Flash
- `vertex/gemini-2.0-flash`: Gemini 2.0 Flash
- `vertex/gemini-2.0-flash-thinking-exp-01-21`: Experimental "thinking" model
- `vertex/gemini-1.5-flash-8b-latest`: The least expensive model
- `vertex/gemini-1.5-pro-latest`
- `vertex/gemini-1.5-flash-latest`

And many more. Use the `vertex/` prefix to reference models:

```bash
llm -m vertex/gemini-1.5-flash-8b-latest --schema 'name,age int,bio' 'invent a dog'
```

**Note:** This plugin provides Gemini models via **Vertex AI** (enterprise API). For the public Google AI Studio API, use the separate [llm-gemini](https://pypi.org/project/llm-gemini/) plugin.

## Model Regional Availability

Different Gemini models are available in different regions. Here's the detailed availability for the main models:

### Gemini 2.5 Flash

The `gemini-2.5-flash` (GA) model is available in the following regions:

| Region Code | Geographic Location | Notes |
|-------------|---------------------|-------|
| **Global** | Global endpoint | Limited features (no tuning, batch prediction, or RAG) |
| **United States** | | |
| us-central1 | Iowa, USA | |
| us-east1 | South Carolina, USA | |
| us-east4 | Northern Virginia, USA | |
| us-east5 | Columbus, Ohio, USA | |
| us-south1 | Dallas, Texas, USA | |
| us-west1 | Oregon, USA | |
| us-west4 | Las Vegas, Nevada, USA | |
| **Europe** | | |
| europe-central2 | Warsaw, Poland | |
| europe-north1 | Hamina, Finland | |
| europe-southwest1 | Madrid, Spain | |
| europe-west1 | St. Ghislain, Belgium | |
| europe-west4 | Eemshaven, Netherlands | |
| europe-west8 | Milan, Italy | |
| **Canada** | | |
| northamerica-northeast1 | Montréal, Canada | |
| **Asia Pacific** | | |
| asia-northeast1 | Tokyo, Japan | 128K context window only* |
| asia-northeast3 | Seoul, South Korea | 128K context window only* |
| asia-south1 | Mumbai, India | 128K context window only* |
| asia-southeast1 | Jurong West, Singapore | 128K context window only* |
| australia-southeast1 | Sydney, Australia | 128K context window only* |

*Regions marked with asterisk have limitations: 128K context window only, supervised fine-tuning not supported.

The preview model (`gemini-2.5-flash-preview-09-2025`) is available only via the **Global** endpoint.

### Gemini 2.5 Pro

The `gemini-2.5-pro` model is available in the following regions:

| Region Code | Geographic Location | Notes |
|-------------|---------------------|-------|
| **Global** | Global endpoint | Limited features (no tuning, batch prediction, or RAG) |
| **United States** | | |
| us-central1 | Iowa, USA | |
| us-east1 | South Carolina, USA | |
| us-east4 | Northern Virginia, USA | |
| us-east5 | Columbus, Ohio, USA | |
| us-south1 | Dallas, Texas, USA | |
| us-west1 | Oregon, USA | |
| us-west4 | Las Vegas, Nevada, USA | |
| **Europe** | | |
| europe-central2 | Warsaw, Poland | |
| europe-north1 | Hamina, Finland | |
| europe-southwest1 | Madrid, Spain | |
| europe-west1 | St. Ghislain, Belgium | |
| europe-west4 | Eemshaven, Netherlands | |
| europe-west8 | Milan, Italy | |
| europe-west9 | Paris, France | |
| **Asia Pacific** | | |
| asia-northeast1 | Tokyo, Japan | 128K context window only; supervised fine-tuning not supported |

**Important Notes:**
- For production use cases requiring specific features (tuning, batch prediction, RAG corpus), use a **regional endpoint** instead of the global endpoint
- Region availability may change; check the official documentation for the latest information:
  - [Gemini 2.5 Flash Documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash)
  - [Gemini 2.5 Pro Documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-pro)
  - [All Vertex AI Locations](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations)

### Gemini 3 Pro (Preview)

The `gemini-3-pro-preview` and related Gemini 3 models (launched November 18, 2025) are currently **only available via the Global endpoint**.

| Model ID | Availability | Auto-Region Override |
|----------|--------------|---------------------|
| gemini-3-pro-preview | Global only | Yes - automatically uses global endpoint |
| gemini-3-pro-preview-11-2025 | Global only | Yes - automatically uses global endpoint |
| gemini-3-pro-preview-11-2025-thinking | Global only | Yes - automatically uses global endpoint |

**Key Features:**
- 1 million token context window
- 64K token output limit
- Multimodal support (text, images, audio, video)
- Google Search grounding
- Thinking budget parameter support
- Knowledge cutoff: January 2025

**Important:** These models automatically use the global endpoint regardless of your configured region setting. You don't need to change your `GOOGLE_CLOUD_REGION` configuration - the plugin handles this automatically.

For more information, see the [Gemini 3 Pro Documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-pro).

### Images, audio and video

Gemini models are multi-modal. You can provide images, audio or video files as input like this:

```bash
llm -m vertex/gemini-2.5-flash 'extract text' -a image.jpg
```
Or with a URL:
```bash
llm -m vertex/gemini-2.5-flash-lite 'describe image' \
  -a https://static.simonwillison.net/static/2024/pelicans.jpg
```
Audio works too:

```bash
llm -m vertex/gemini-2.5-flash 'transcribe audio' -a audio.mp3
```

And video:

```bash
llm -m vertex/gemini-2.5-flash 'describe what happens' -a video.mp4
```

### JSON output

Use `-o json_object 1` to force the output to be JSON:

```bash
llm -m vertex/gemini-2.5-flash -o json_object 1 \
  '3 largest cities in California, list of {"name": "..."}'
```
Outputs:
```json
{"cities": [{"name": "Los Angeles"}, {"name": "San Diego"}, {"name": "San Jose"}]}
```

### Code execution

Gemini models can write and execute code - they can decide to write Python code, execute it in a secure sandbox and use the result as part of their response.

To enable this feature, use `-o code_execution 1`:

```bash
llm -m vertex/gemini-2.5-flash -o code_execution 1 \
'use python to calculate (factorial of 13) * 3'
```

### Google search

Some Gemini models support Grounding with Google Search, where the model can run a Google search and use the results as part of answering a prompt.

To run a prompt with Google search enabled, use `-o google_search 1`:

```bash
llm -m vertex/gemini-2.5-flash -o google_search 1 \
  'What happened in Ireland today?'
```

### URL context

Gemini models support a URL context tool which, when enabled, allows the models to fetch additional content from URLs as part of their execution.

You can enable that with the `-o url_context 1` option:

```bash
llm -m vertex/gemini-2.5-flash -o url_context 1 'Latest headline on simonwillison.net'
```

### Chat

To chat interactively with the model, run `llm chat`:

```bash
llm chat -m vertex/gemini-2.5-flash
```

### Timeouts

By default there is no timeout against the Vertex AI API. You can use the `timeout` option to protect against API requests that hang indefinitely:

```bash
llm -m vertex/gemini-2.5-flash 'epic saga about mice' -o timeout 1.5
```

## Embeddings

The plugin also adds support for the `gemini-embedding-exp-03-07` and `text-embedding-004` embedding models.

Run that against a single string like this:
```bash
llm embed -m text-embedding-004 -c 'hello world'
```

See the [LLM embeddings documentation](https://llm.datasette.io/en/stable/embeddings/cli.html) for further details.

## Prerequisites

### GCP Project Setup

1. Create a GCP project at https://console.cloud.google.com
2. Enable the Vertex AI API for your project
3. Set up billing for your project

### Service Account Setup (if not using ADC)

1. Go to IAM & Admin > Service Accounts in GCP Console
2. Create a new service account
3. Grant it the "Vertex AI User" role
4. Create and download a JSON key
5. Configure the plugin with the path to this key (see Configuration above)

## Costs

Vertex AI charges for model usage. See [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) for details.

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-gemini-vertex
python3 -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
llm install -e '.[test]'
```
To run the tests:
```bash
pytest
```

## Troubleshooting

### "No GCP project ID found" error

Make sure you've set the `GOOGLE_CLOUD_PROJECT` environment variable or run:
```bash
llm vertex set-project your-project-id
```

### Authentication errors

Verify your authentication setup:
```bash
llm vertex config
```

For API key:
```bash
llm keys set vertex
```

For ADC:
```bash
gcloud auth application-default login
```

For service account, ensure `GOOGLE_APPLICATION_CREDENTIALS` points to a valid JSON file.

### "API not enabled" errors

Enable the Vertex AI API:
```bash
gcloud services enable aiplatform.googleapis.com --project=your-project-id
```
