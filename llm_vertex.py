import click
import copy
import httpx
import ijson
import json
import llm
import os
from google.auth import default
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from pydantic import Field
from typing import Optional

SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
]

# https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/ground-gemini#supported_models_2
GOOGLE_SEARCH_MODELS = {
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-001",
    "gemini-1.5-flash-001",
    "gemini-1.5-pro-002",
    "gemini-1.5-flash-002",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-pro-exp-03-25",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-preview-09-2025",
    "gemini-2.5-flash-lite-preview-09-2025",
}

# Older Google models used google_search_retrieval instead of google_search
GOOGLE_SEARCH_MODELS_USING_SEARCH_RETRIEVAL = {
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-001",
    "gemini-1.5-flash-001",
    "gemini-1.5-pro-002",
    "gemini-1.5-flash-002",
    "gemini-2.0-flash-exp",
}

THINKING_BUDGET_MODELS = {
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.0-flash-thinking-exp-1219",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro-exp-03-25",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-preview-09-2025",
    "gemini-2.5-flash-lite-preview-09-2025",
}

NO_VISION_MODELS = {"gemma-3-1b-it", "gemma-3n-e4b-it"}

# Valid Vertex AI regions as of 2025
# See https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations
VALID_REGIONS = {
    "global",  # Global endpoint (limited features)
    # United States
    "us-central1",
    "us-east1",
    "us-east4",
    "us-east5",
    "us-south1",
    "us-west1",
    "us-west4",
    # Canada
    "northamerica-northeast1",
    # South America
    "southamerica-east1",
    # Europe
    "europe-west1",
    "europe-west2",
    "europe-west3",
    "europe-west4",
    "europe-west6",
    "europe-west8",
    "europe-west9",
    "europe-north1",
    "europe-southwest1",
    "europe-central2",
    # Asia Pacific
    "asia-east1",
    "asia-northeast1",
    "asia-northeast3",
    "asia-southeast1",
    "asia-south1",
    "australia-southeast1",
    "australia-southeast2",
    # Middle East
    "me-central1",
    "me-central2",
    "me-west1",
}

ATTACHMENT_TYPES = {
    # Text
    "text/plain",
    "text/csv",
    # PDF
    "application/pdf",
    # Images
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
    # Audio
    "audio/wav",
    "audio/mp3",
    "audio/aiff",
    "audio/aac",
    "audio/ogg",
    "application/ogg",
    "audio/flac",
    "audio/mpeg",  # Treated as audio/mp3
    # Video
    "video/mp4",
    "video/mpeg",
    "video/mov",
    "video/avi",
    "video/x-flv",
    "video/mpg",
    "video/webm",
    "video/wmv",
    "video/3gpp",
    "video/quicktime",
}


def _save_vertex_config(key, value):
    """
    Save a configuration value to the llm keys.json file.
    This is used to store vertex-specific configuration like project ID and region.
    """
    import json
    from pathlib import Path

    keys_path = llm.user_dir() / "keys.json"

    # Load existing keys or create default structure
    if keys_path.exists():
        with open(keys_path, "r") as f:
            keys = json.load(f)
    else:
        keys = {"// Note": "This file stores secret API credentials. Do not share!"}
        keys_path.parent.mkdir(parents=True, exist_ok=True)

    # Update the key
    keys[key] = value

    # Write back to file
    with open(keys_path, "w") as f:
        json.dump(keys, f, indent=2)
        f.write("\n")

    # Set restrictive permissions if this is a new file
    if not keys_path.exists():
        keys_path.chmod(0o600)


def validate_region(region):
    """
    Validate that a region is a known Vertex AI region.

    Returns True if valid, False otherwise.
    """
    return region in VALID_REGIONS


def get_region_suggestions(invalid_region):
    """
    Get suggestions for similar valid regions when an invalid region is provided.

    This helps users who might have typos or be using the wrong format.
    """
    import difflib

    # Get close matches using difflib
    suggestions = difflib.get_close_matches(
        invalid_region.lower(),
        [r.lower() for r in VALID_REGIONS],
        n=3,
        cutoff=0.6
    )

    return suggestions


def get_api_key():
    """
    Get Vertex AI API key from environment variable or llm config.
    API keys are recommended for testing only, not production.

    Returns API key string or None if not configured.
    """
    # Check environment variable first
    api_key = os.environ.get("GOOGLE_CLOUD_API_KEY")
    if api_key:
        return api_key

    # Try to get from llm config
    try:
        api_key = llm.get_key("vertex")
        if api_key:
            return api_key
    except:
        pass

    return None


def get_vertex_credentials():
    """
    Get Google Cloud credentials for Vertex AI.
    Supports:
    1. Explicit service account file via GOOGLE_APPLICATION_CREDENTIALS env var or config
    2. Application Default Credentials (ADC)

    Returns a tuple of (credentials, project_id)
    If credentials cannot be obtained, returns (None, None)
    """
    # Try to get credentials path from environment or config
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    if creds_path and os.path.exists(creds_path):
        # Use explicit service account file
        credentials = service_account.Credentials.from_service_account_file(
            creds_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        # Extract project_id from service account file
        with open(creds_path) as f:
            service_account_info = json.load(f)
            project_id = service_account_info.get("project_id")
        return credentials, project_id
    else:
        # Try to use Application Default Credentials
        try:
            credentials, project_id = default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            return credentials, project_id
        except Exception:
            # If ADC is not available, return None - API key may be used instead
            return None, None


def get_access_token(credentials):
    """
    Get a fresh OAuth2 access token from credentials.
    Refreshes the token if it's expired.
    """
    if not credentials.valid:
        credentials.refresh(Request())
    return credentials.token


def get_project_and_region():
    """
    Get GCP project ID and region from environment variables or config.
    Precedence: env vars > config > defaults

    Returns a tuple of (project_id, region)
    """
    # Get project ID
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        # Try to get from llm config
        try:
            project_id = llm.get_key("", "vertex-project", "GOOGLE_CLOUD_PROJECT")
        except:
            pass

    # Get region (default to 'global')
    region = os.environ.get("GOOGLE_CLOUD_REGION", "global")
    if region == "global":
        # Try to get from config
        try:
            config_region = llm.get_key("", "vertex-region", "GOOGLE_CLOUD_REGION")
            if config_region:
                region = config_region
        except:
            pass

    return project_id, region


def build_vertex_endpoint(region, project_id, model_id, method="streamGenerateContent"):
    """
    Build the Vertex AI endpoint URL.

    For 'global' region: https://aiplatform.googleapis.com/v1/...
    For specific regions: https://{region}-aiplatform.googleapis.com/v1/...
    """
    if region == "global":
        base_url = "https://aiplatform.googleapis.com"
    else:
        base_url = f"https://{region}-aiplatform.googleapis.com"

    return f"{base_url}/v1/projects/{project_id}/locations/{region}/publishers/google/models/{model_id}:{method}"


@llm.hookimpl
def register_models(register):
    # Register both sync and async versions of each model
    for model_id in (
        "gemini-pro",
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-001",
        "gemini-1.5-flash-001",
        "gemini-1.5-pro-002",
        "gemini-1.5-flash-002",
        "gemini-1.5-flash-8b-latest",
        "gemini-1.5-flash-8b-001",
        "gemini-exp-1114",
        "gemini-exp-1121",
        "gemini-exp-1206",
        "gemini-2.0-flash-exp",
        "learnlm-1.5-pro-experimental",
        # Gemma 3 models:
        "gemma-3-1b-it",
        "gemma-3-4b-it",
        "gemma-3-12b-it",  # 12th March 2025
        "gemma-3-27b-it",
        "gemma-3n-e4b-it",  # 20th May 2025
        "gemini-2.0-flash-thinking-exp-1219",
        "gemini-2.0-flash-thinking-exp-01-21",
        # Released 5th Feb 2025:
        "gemini-2.0-flash",
        "gemini-2.0-pro-exp-02-05",
        # Released 25th Feb 2025:
        "gemini-2.0-flash-lite",
        # 25th March 2025:
        "gemini-2.5-pro-exp-03-25",
        # 4th April 2025 (paid):
        "gemini-2.5-pro-preview-03-25",
        # 17th April 2025:
        "gemini-2.5-flash-preview-04-17",
        # 6th May 2025:
        "gemini-2.5-pro-preview-05-06",
        # 20th May 2025:
        "gemini-2.5-flash-preview-05-20",
        # 5th June 2025:
        "gemini-2.5-pro-preview-06-05",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        # 22nd July 2025:
        "gemini-2.5-flash-lite",
        # 25th Spetember 2025:
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-2.5-flash-preview-09-2025",
        "gemini-2.5-flash-lite-preview-09-2025",
    ):
        can_google_search = model_id in GOOGLE_SEARCH_MODELS
        can_thinking_budget = model_id in THINKING_BUDGET_MODELS
        can_vision = model_id not in NO_VISION_MODELS
        can_schema = "flash-thinking" not in model_id and "gemma-3" not in model_id
        register(
            Vertex(
                model_id,
                can_vision=can_vision,
                can_google_search=can_google_search,
                can_thinking_budget=can_thinking_budget,
                can_schema=can_schema,
            ),
            AsyncVertex(
                model_id,
                can_vision=can_vision,
                can_google_search=can_google_search,
                can_thinking_budget=can_thinking_budget,
                can_schema=can_schema,
            ),
        )


def resolve_type(attachment):
    mime_type = attachment.resolve_type()
    # https://github.com/simonw/llm/issues/587#issuecomment-2439785140
    if mime_type == "audio/mpeg":
        mime_type = "audio/mp3"
    if mime_type == "application/ogg":
        mime_type = "audio/ogg"
    return mime_type


def cleanup_schema(schema, in_properties=False):
    "Gemini supports only a subset of JSON schema"
    keys_to_remove = ("$schema", "additionalProperties", "title")

    # First pass: resolve $ref references using $defs
    if isinstance(schema, dict) and "$defs" in schema:
        defs = schema.pop("$defs")
        _resolve_refs(schema, defs)

    if isinstance(schema, dict):
        # Only remove keys if we're not inside a 'properties' block.
        if not in_properties:
            for key in keys_to_remove:
                schema.pop(key, None)
        for key, value in list(schema.items()):
            # If the key is 'properties', set the flag for its value.
            if key == "properties" and isinstance(value, dict):
                cleanup_schema(value, in_properties=True)
            else:
                cleanup_schema(value, in_properties=in_properties)
    elif isinstance(schema, list):
        for item in schema:
            cleanup_schema(item, in_properties=in_properties)
    return schema


def _resolve_refs(schema, defs):
    """Recursively resolve $ref references in schema using definitions."""
    if isinstance(schema, dict):
        if "$ref" in schema:
            # Extract the reference path (e.g., "#/$defs/Dog" -> "Dog")
            ref_path = schema.pop("$ref")
            if ref_path.startswith("#/$defs/"):
                def_name = ref_path.split("/")[-1]
                if def_name in defs:
                    # Replace the $ref with the actual definition
                    schema.update(copy.deepcopy(defs[def_name]))

        # Recursively resolve refs in nested structures
        for value in schema.values():
            _resolve_refs(value, defs)
    elif isinstance(schema, list):
        for item in schema:
            _resolve_refs(item, defs)


class _SharedGemini:
    can_stream = True
    supports_schema = True
    supports_tools = True

    attachment_types = set()

    # Vertex AI credentials and configuration
    _credentials = None
    _project_id = None
    _region = None

    class Options(llm.Options):
        code_execution: Optional[bool] = Field(
            description="Enables the model to generate and run Python code",
            default=None,
        )
        temperature: Optional[float] = Field(
            description=(
                "Controls the randomness of the output. Use higher values for "
                "more creative responses, and lower values for more "
                "deterministic responses."
            ),
            default=None,
            ge=0.0,
            le=2.0,
        )
        max_output_tokens: Optional[int] = Field(
            description="Sets the maximum number of tokens to include in a candidate.",
            default=None,
        )
        top_p: Optional[float] = Field(
            description=(
                "Changes how the model selects tokens for output. Tokens are "
                "selected from the most to least probable until the sum of "
                "their probabilities equals the topP value."
            ),
            default=None,
            ge=0.0,
            le=1.0,
        )
        top_k: Optional[int] = Field(
            description=(
                "Changes how the model selects tokens for output. A topK of 1 "
                "means the selected token is the most probable among all the "
                "tokens in the model's vocabulary, while a topK of 3 means "
                "that the next token is selected from among the 3 most "
                "probable using the temperature."
            ),
            default=None,
            ge=1,
        )
        json_object: Optional[bool] = Field(
            description="Output a valid JSON object {...}",
            default=None,
        )
        timeout: Optional[float] = Field(
            description=(
                "The maximum time in seconds to wait for a response. "
                "If the model does not respond within this time, "
                "the request will be aborted."
            ),
            default=None,
        )
        url_context: Optional[bool] = Field(
            description=(
                "Enable the URL context tool so the model can fetch content "
                "from URLs mentioned in the prompt"
            ),
            default=None,
        )

    class OptionsWithGoogleSearch(Options):
        google_search: Optional[bool] = Field(
            description="Enables the model to use Google Search to improve the accuracy and recency of responses from the model",
            default=None,
        )

    class OptionsWithThinkingBudget(OptionsWithGoogleSearch):
        thinking_budget: Optional[int] = Field(
            description="Indicates the thinking budget in tokens. Set to 0 to disable.",
            default=None,
        )

    def __init__(
        self,
        gemini_model_id,
        can_vision=True,
        can_google_search=False,
        can_thinking_budget=False,
        can_schema=False,
    ):
        self.model_id = "vertex/{}".format(gemini_model_id)
        self.gemini_model_id = gemini_model_id
        self.can_google_search = can_google_search
        self.supports_schema = can_schema
        if can_google_search:
            self.Options = self.OptionsWithGoogleSearch
        self.can_thinking_budget = can_thinking_budget
        if can_thinking_budget:
            self.Options = self.OptionsWithThinkingBudget
        if can_vision:
            self.attachment_types = ATTACHMENT_TYPES

    def get_credentials_and_config(self):
        """
        Get Vertex AI credentials, project, and region.
        Caches credentials to avoid re-authentication on every request.

        Returns a tuple of (credentials, project_id, region)

        Note: credentials may be None if using API key authentication
        """
        # Check if we're using API key authentication
        api_key = get_api_key()

        # Only try to get credentials if not using API key or if not cached
        if self._credentials is None and not api_key:
            self._credentials, creds_project = get_vertex_credentials()
            # Store the project from credentials for later use
            if not hasattr(self, '_creds_project'):
                self._creds_project = creds_project
        elif self._credentials is None and api_key:
            # Using API key, skip credential fetching
            _, creds_project = None, None
            if not hasattr(self, '_creds_project'):
                self._creds_project = None

        # Get project and region (with env var/config override)
        project_id, region = get_project_and_region()

        # If project_id not found in env/config, use the one from credentials
        if not project_id:
            project_id = getattr(self, '_creds_project', None)

        if not project_id:
            raise llm.ModelError(
                "No GCP project ID found. Set GOOGLE_CLOUD_PROJECT environment variable "
                "or run: llm vertex set-project <project-id>"
            )

        return self._credentials, project_id, region

    def get_auth_header(self):
        """
        Get the authentication header (API key or OAuth2 token).
        API keys are recommended for testing only, not production.
        """
        # Check for API key first (simpler authentication)
        api_key = get_api_key()
        if api_key:
            return {"x-goog-api-key": api_key}

        # Fall back to OAuth2 authentication
        credentials, _, _ = self.get_credentials_and_config()
        if credentials is None:
            raise llm.ModelError(
                "No authentication available. Either set an API key with 'llm keys set vertex' "
                "or configure Application Default Credentials with 'gcloud auth application-default login'"
            )
        token = get_access_token(credentials)
        return {"Authorization": f"Bearer {token}"}

    def build_messages(self, prompt, conversation):
        messages = []
        if conversation:
            for response in conversation.responses:
                parts = []
                for attachment in response.attachments:
                    mime_type = resolve_type(attachment)
                    parts.append(
                        {
                            "inlineData": {
                                "data": attachment.base64_content(),
                                "mimeType": mime_type,
                            }
                        }
                    )
                if response.prompt.prompt:
                    parts.append({"text": response.prompt.prompt})
                if response.prompt.tool_results:
                    parts.extend(
                        [
                            {
                                "function_response": {
                                    "name": tool_result.name,
                                    "response": {
                                        "output": tool_result.output,
                                    },
                                }
                            }
                            for tool_result in response.prompt.tool_results
                        ]
                    )
                messages.append({"role": "user", "parts": parts})
                model_parts = []
                response_text = response.text_or_raise()
                model_parts.append({"text": response_text})
                tool_calls = response.tool_calls_or_raise()
                if tool_calls:
                    model_parts.extend(
                        [
                            {
                                "function_call": {
                                    "name": tool_call.name,
                                    "args": tool_call.arguments,
                                }
                            }
                            for tool_call in tool_calls
                        ]
                    )
                messages.append({"role": "model", "parts": model_parts})

        parts = []
        if prompt.prompt:
            parts.append({"text": prompt.prompt})
        if prompt.tool_results:
            parts.extend(
                [
                    {
                        "function_response": {
                            "name": tool_result.name,
                            "response": {
                                "output": tool_result.output,
                            },
                        }
                    }
                    for tool_result in prompt.tool_results
                ]
            )
        for attachment in prompt.attachments:
            mime_type = resolve_type(attachment)
            parts.append(
                {
                    "inlineData": {
                        "data": attachment.base64_content(),
                        "mimeType": mime_type,
                    }
                }
            )

        messages.append({"role": "user", "parts": parts})
        return messages

    def build_request_body(self, prompt, conversation):
        body = {
            "contents": self.build_messages(prompt, conversation),
            "safetySettings": SAFETY_SETTINGS,
        }
        if prompt.system:
            body["systemInstruction"] = {"parts": [{"text": prompt.system}]}

        tools = []
        if prompt.options and prompt.options.code_execution:
            tools.append({"codeExecution": {}})
        if prompt.options and self.can_google_search and prompt.options.google_search:
            tool_name = (
                "google_search_retrieval"
                if self.model_id in GOOGLE_SEARCH_MODELS_USING_SEARCH_RETRIEVAL
                else "google_search"
            )
            tools.append({tool_name: {}})
        if prompt.options and prompt.options.url_context:
            tools.append({"url_context": {}})
        if prompt.tools:
            tools.append(
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                        }
                        for tool in prompt.tools
                    ]
                }
            )
        if tools:
            body["tools"] = tools

        generation_config = {}

        if prompt.schema:
            generation_config.update(
                {
                    "response_mime_type": "application/json",
                    "response_schema": cleanup_schema(copy.deepcopy(prompt.schema)),
                }
            )

        if self.can_thinking_budget and prompt.options.thinking_budget is not None:
            generation_config["thinking_config"] = {
                "thinking_budget": prompt.options.thinking_budget
            }

        config_map = {
            "temperature": "temperature",
            "max_output_tokens": "maxOutputTokens",
            "top_p": "topP",
            "top_k": "topK",
        }
        if prompt.options and prompt.options.json_object:
            generation_config["response_mime_type"] = "application/json"

        if any(
            getattr(prompt.options, key, None) is not None for key in config_map.keys()
        ):
            for key, other_key in config_map.items():
                config_value = getattr(prompt.options, key, None)
                if config_value is not None:
                    generation_config[other_key] = config_value

        if generation_config:
            body["generationConfig"] = generation_config

        return body

    def process_part(self, part, response):
        if "functionCall" in part:
            response.add_tool_call(
                llm.ToolCall(
                    name=part["functionCall"]["name"],
                    arguments=part["functionCall"]["args"],
                )
            )
        if "text" in part:
            return part["text"]
        elif "executableCode" in part:
            return f'```{part["executableCode"]["language"].lower()}\n{part["executableCode"]["code"].strip()}\n```\n'
        elif "codeExecutionResult" in part:
            return f'```\n{part["codeExecutionResult"]["output"].strip()}\n```\n'
        return ""

    def process_candidates(self, candidates, response):
        # We only use the first candidate
        for part in candidates[0]["content"]["parts"]:
            yield self.process_part(part, response)

    def set_usage(self, response):
        try:
            # Don't record the "content" key from that last candidate
            for candidate in response.response_json["candidates"]:
                candidate.pop("content", None)
            usage = response.response_json.pop("usageMetadata")
            input_tokens = usage.pop("promptTokenCount", None)
            # See https://github.com/simonw/llm-gemini/issues/75#issuecomment-2861827509
            candidates_token_count = usage.get("candidatesTokenCount") or 0
            thoughts_token_count = usage.get("thoughtsTokenCount") or 0
            output_tokens = candidates_token_count + thoughts_token_count
            tool_token_count = usage.get("toolUsePromptTokenCount") or 0
            if tool_token_count:
                if input_tokens is None:
                    input_tokens = tool_token_count
                else:
                    input_tokens += tool_token_count
            usage.pop("totalTokenCount", None)
            if input_tokens is not None:
                response.set_usage(
                    input=input_tokens, output=output_tokens, details=usage or None
                )
        except (IndexError, KeyError):
            pass


class Vertex(_SharedGemini, llm.Model):
    def execute(self, prompt, stream, response, conversation):
        # Get Vertex AI credentials and configuration
        _, project_id, region = self.get_credentials_and_config()
        url = build_vertex_endpoint(region, project_id, self.gemini_model_id, "streamGenerateContent")
        gathered = []
        body = self.build_request_body(prompt, conversation)

        with httpx.stream(
            "POST",
            url,
            timeout=prompt.options.timeout,
            headers=self.get_auth_header(),
            json=body,
        ) as http_response:
            events = ijson.sendable_list()
            coro = ijson.items_coro(events, "item")
            for chunk in http_response.iter_bytes():
                coro.send(chunk)
                if events:
                    for event in events:
                        if isinstance(event, dict) and "error" in event:
                            raise llm.ModelError(event["error"]["message"])
                        try:
                            yield from self.process_candidates(
                                event["candidates"], response
                            )
                        except KeyError:
                            yield ""
                        gathered.append(event)
                    events.clear()
        response.response_json = gathered[-1]
        resolved_model = gathered[-1]["modelVersion"]
        response.set_resolved_model(resolved_model)
        self.set_usage(response)


class AsyncVertex(_SharedGemini, llm.AsyncModel):
    async def execute(self, prompt, stream, response, conversation):
        # Get Vertex AI credentials and configuration
        _, project_id, region = self.get_credentials_and_config()
        url = build_vertex_endpoint(region, project_id, self.gemini_model_id, "streamGenerateContent")
        gathered = []
        body = self.build_request_body(prompt, conversation)

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                url,
                timeout=prompt.options.timeout,
                headers=self.get_auth_header(),
                json=body,
            ) as http_response:
                events = ijson.sendable_list()
                coro = ijson.items_coro(events, "item")
                async for chunk in http_response.aiter_bytes():
                    coro.send(chunk)
                    if events:
                        for event in events:
                            if isinstance(event, dict) and "error" in event:
                                raise llm.ModelError(event["error"]["message"])
                            try:
                                for chunk in self.process_candidates(
                                    event["candidates"], response
                                ):
                                    yield chunk
                            except KeyError:
                                yield ""
                            gathered.append(event)
                        events.clear()
        response.response_json = gathered[-1]
        resolved_model = gathered[-1]["modelVersion"]
        response.set_resolved_model(resolved_model)
        self.set_usage(response)


@llm.hookimpl
def register_embedding_models(register):
    register(GeminiEmbeddingModel("text-embedding-004", "text-embedding-004"))
    # gemini-embedding-exp-03-07 in different truncation sizes
    register(
        GeminiEmbeddingModel(
            "gemini-embedding-exp-03-07", "gemini-embedding-exp-03-07"
        ),
    )
    for i in (128, 256, 512, 1024, 2048):
        register(
            GeminiEmbeddingModel(
                f"gemini-embedding-exp-03-07-{i}", f"gemini-embedding-exp-03-07", i
            ),
        )


class GeminiEmbeddingModel(llm.EmbeddingModel):
    batch_size = 20

    # Vertex AI credentials (shared across instances)
    _credentials = None

    def __init__(self, model_id, gemini_model_id, truncate=None):
        self.model_id = model_id
        self.gemini_model_id = gemini_model_id
        self.truncate = truncate

    def get_credentials_and_config(self):
        """Get Vertex AI credentials, project, and region."""
        # Check if we're using API key authentication
        api_key = get_api_key()

        # Only try to get credentials if not using API key or if not cached
        if self._credentials is None and not api_key:
            self._credentials, creds_project = get_vertex_credentials()
            # Store the project from credentials for later use
            if not hasattr(self, '_creds_project'):
                self._creds_project = creds_project
        elif self._credentials is None and api_key:
            # Using API key, skip credential fetching
            if not hasattr(self, '_creds_project'):
                self._creds_project = None

        # Get project and region
        project_id, region = get_project_and_region()

        # If project_id not found in env/config, use the one from credentials
        if not project_id:
            project_id = getattr(self, '_creds_project', None)

        if not project_id:
            raise llm.ModelError(
                "No GCP project ID found. Set GOOGLE_CLOUD_PROJECT environment variable "
                "or run: llm vertex set-project <project-id>"
            )

        return self._credentials, project_id, region

    def embed_batch(self, items):
        # Get Vertex AI credentials and configuration
        credentials, project_id, region = self.get_credentials_and_config()

        headers = {
            "Content-Type": "application/json",
        }

        # Check for API key first
        api_key = get_api_key()
        if api_key:
            headers["x-goog-api-key"] = api_key
        else:
            # Use OAuth2 token
            if credentials is None:
                raise llm.ModelError(
                    "No authentication available. Either set an API key with 'llm keys set vertex' "
                    "or configure Application Default Credentials with 'gcloud auth application-default login'"
                )
            token = get_access_token(credentials)
            headers["Authorization"] = f"Bearer {token}"

        data = {
            "requests": [
                {
                    "model": f"projects/{project_id}/locations/{region}/publishers/google/models/{self.gemini_model_id}",
                    "content": {"parts": [{"text": item}]},
                }
                for item in items
            ]
        }

        # Build Vertex AI endpoint
        url = build_vertex_endpoint(region, project_id, self.gemini_model_id, "batchEmbedContents")

        with httpx.Client() as client:
            response = client.post(
                url,
                headers=headers,
                json=data,
                timeout=None,
            )

        response.raise_for_status()
        values = [item["values"] for item in response.json()["embeddings"]]
        if self.truncate:
            values = [value[: self.truncate] for value in values]
        return values


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def vertex():
        "Commands relating to the llm-vertex plugin (Vertex AI)"

    @vertex.command(name="set-project")
    @click.argument("project_id")
    def set_project(project_id):
        """
        Set the GCP project ID for Vertex AI

        Example: llm vertex set-project my-gcp-project
        """
        _save_vertex_config("vertex-project", project_id)
        click.echo(f"GCP project ID set to: {project_id}")

    @vertex.command(name="set-region")
    @click.argument("region")
    def set_region(region):
        """
        Set the GCP region for Vertex AI (default: global)

        Example: llm vertex set-region us-central1

        Use 'llm vertex list-regions' to see all available regions.
        """
        # Validate region
        if not validate_region(region):
            suggestions = get_region_suggestions(region)
            error_msg = f"Invalid region: {region}\n\n"
            if suggestions:
                error_msg += f"Did you mean one of these?\n"
                for suggestion in suggestions:
                    error_msg += f"  - {suggestion}\n"
            error_msg += f"\nUse 'llm vertex list-regions' to see all valid regions."
            error_msg += f"\nSee https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations for details."
            raise click.ClickException(error_msg)

        # Warn about global endpoint limitations
        if region == "global":
            click.echo("⚠️  Warning: The 'global' endpoint has limitations:")
            click.echo("   - Does not support tuning, batch prediction, or RAG corpus creation")
            click.echo("   - Does not guarantee region-specific ML processing")
            click.echo("   - Does not provide data residency compliance")
            click.echo("   Consider using a specific region if you need these features.\n")

        _save_vertex_config("vertex-region", region)
        click.echo(f"GCP region set to: {region}")

    @vertex.command(name="list-regions")
    def list_regions():
        """
        List all available Vertex AI regions

        Shows all regions where Vertex AI Generative AI models are available.
        Note: Some models may not be available in all regions.
        """
        click.echo("Available Vertex AI Regions:\n")

        # Group regions by area
        regions_by_area = {
            "Global": ["global"],
            "United States": [
                "us-central1", "us-east1", "us-east4", "us-east5",
                "us-south1", "us-west1", "us-west4"
            ],
            "Canada": ["northamerica-northeast1"],
            "South America": ["southamerica-east1"],
            "Europe": [
                "europe-west1", "europe-west2", "europe-west3", "europe-west4",
                "europe-west6", "europe-west8", "europe-west9", "europe-north1",
                "europe-southwest1", "europe-central2"
            ],
            "Asia Pacific": [
                "asia-east1", "asia-northeast1", "asia-northeast3",
                "asia-southeast1", "asia-south1", "australia-southeast1",
                "australia-southeast2"
            ],
            "Middle East": ["me-central1", "me-central2", "me-west1"],
        }

        for area, regions in regions_by_area.items():
            click.echo(f"{area}:")
            for region in sorted(regions):
                if region == "global":
                    click.echo(f"  {region} (limited features - see docs)")
                else:
                    click.echo(f"  {region}")
            click.echo()

        click.echo("⚠️  Note: 'global' endpoint limitations:")
        click.echo("   - Does not support tuning, batch prediction, or RAG corpus")
        click.echo("   - Does not guarantee region-specific data processing")
        click.echo("   - Does not provide data residency compliance\n")
        click.echo("For more details, see:")
        click.echo("https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations")

    @vertex.command(name="set-credentials")
    @click.argument("credentials_path")
    def set_credentials(credentials_path):
        """
        Set the path to the service account JSON file

        Example: llm vertex set-credentials /path/to/service-account.json
        """
        if not os.path.exists(credentials_path):
            raise click.ClickException(f"File not found: {credentials_path}")
        _save_vertex_config("vertex-credentials-path", credentials_path)
        click.echo(f"Credentials path set to: {credentials_path}")
        click.echo("Note: You can also set GOOGLE_APPLICATION_CREDENTIALS environment variable")

    @vertex.command(name="config")
    def show_config():
        """
        Show current Vertex AI configuration
        """
        project_id, region = get_project_and_region()
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "Not set")
        api_key = get_api_key()

        click.echo("Current Vertex AI Configuration:")
        click.echo(f"  Project ID: {project_id or 'Not set (will use ADC default)'}")
        click.echo(f"  Region: {region}")
        click.echo(f"  API Key: {'Set' if api_key else 'Not set'}")
        click.echo(f"  Credentials Path: {creds_path}")
        click.echo("\nEnvironment Variables:")
        click.echo(f"  GOOGLE_CLOUD_PROJECT: {os.environ.get('GOOGLE_CLOUD_PROJECT', 'Not set')}")
        click.echo(f"  GOOGLE_CLOUD_REGION: {os.environ.get('GOOGLE_CLOUD_REGION', 'Not set')}")
        click.echo(f"  GOOGLE_CLOUD_API_KEY: {'Set' if os.environ.get('GOOGLE_CLOUD_API_KEY') else 'Not set'}")
        click.echo(f"  GOOGLE_APPLICATION_CREDENTIALS: {creds_path}")
        click.echo("\nAuthentication Method:")
        if api_key:
            click.echo("  Using API key (recommended for testing only)")
        else:
            click.echo("  Using OAuth2/ADC (recommended for production)")
