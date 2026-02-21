from click.testing import CliRunner
import llm
from llm.cli import cli
import nest_asyncio
import json
import os
import pytest
import pydantic
import sys
from pydantic import BaseModel
from typing import List, Optional
from llm_vertex import cleanup_schema

nest_asyncio.apply()

# For Vertex AI, we need GCP project and credentials setup
# Set these environment variables for testing:
# - GOOGLE_CLOUD_PROJECT (required)
# - GOOGLE_APPLICATION_CREDENTIALS (path to service account JSON)
# OR use `gcloud auth application-default login`

# Ensure environment is configured for tests
if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
    os.environ["GOOGLE_CLOUD_PROJECT"] = os.environ.get("PYTEST_GCP_PROJECT", "test-project")
if not os.environ.get("GOOGLE_CLOUD_REGION"):
    os.environ["GOOGLE_CLOUD_REGION"] = "global"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_prompt():
    model = llm.get_model("gemini-1.5-flash-latest")
    response = model.prompt("Name for a pet pelican, just the name")
    assert str(response) == "Percy\n"
    assert response.response_json == {
        "candidates": [
            {
                "finishReason": "STOP",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "NEGLIGIBLE",
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability": "NEGLIGIBLE",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE",
                    },
                ],
            }
        ],
        "modelVersion": "gemini-1.5-flash-latest",
    }
    assert response.token_details == {
        "candidatesTokenCount": 2,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 9}],
        "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 2}],
    }
    assert response.input_tokens == 9
    assert response.output_tokens == 2

    # Skip async test on Python 3.14 due to httpcore cleanup incompatibility
    # https://github.com/simonw/llm-gemini/issues/114
    if sys.version_info < (3, 14):
        # And try it async too
        async_model = llm.get_async_model("gemini-1.5-flash-latest")
        response = await async_model.prompt(
            "Name for a pet pelican, just the name"
        )
        text = await response.text()
        assert text == "Percy\n"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_prompt_with_pydantic_schema():
    class Dog(pydantic.BaseModel):
        name: str
        age: int
        bio: str

    class Dogs(BaseModel):
        dogs: List[Dog]

    model = llm.get_model("gemini-1.5-flash-latest")
    response = model.prompt(
        "Invent a cool dog", schema=Dog, stream=False
    )
    assert json.loads(response.text()) == {
        "age": 3,
        "bio": "A fluffy Samoyed with exceptional intelligence and a love for belly rubs. He's mastered several tricks, including fetching the newspaper and opening doors.",
        "name": "Cloud",
    }
    assert response.response_json == {
        "candidates": [
            {
                "finishReason": "STOP",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "NEGLIGIBLE",
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability": "NEGLIGIBLE",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE",
                    },
                ],
            }
        ],
        "modelVersion": "gemini-1.5-flash-latest",
    }
    assert response.input_tokens == 10


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_prompt_with_multiple_dogs():
    class Dog(pydantic.BaseModel):
        name: str
        age: int
        bio: str

    class Dogs(BaseModel):
        dogs: List[Dog]

    model = llm.get_model("gemini-2.0-flash")
    response = model.prompt(
        "Invent 3 cool dogs", schema=Dogs, stream=False
    )
    result = json.loads(response.text())

    # Verify we got 3 dogs
    assert "dogs" in result
    assert len(result["dogs"]) == 3

    # Verify each dog has the required fields
    for dog in result["dogs"]:
        assert "name" in dog
        assert "age" in dog
        assert "bio" in dog
        assert isinstance(dog["name"], str)
        assert isinstance(dog["age"], int)
        assert isinstance(dog["bio"], str)


@pytest.mark.vcr
@pytest.mark.parametrize(
    "model_id",
    (
        "gemini-embedding-exp-03-07",
        "gemini-embedding-exp-03-07-128",
        "gemini-embedding-exp-03-07-512",
    ),
)
def test_embedding(model_id, monkeypatch):
    # Ensure GCP environment is set for embedding tests
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    model = llm.get_embedding_model(model_id)
    response = model.embed("Some text goes here")
    expected_length = 3072
    if model_id.endswith("-128"):
        expected_length = 128
    elif model_id.endswith("-512"):
        expected_length = 512
    assert len(response) == expected_length


@pytest.mark.parametrize(
    "schema,expected",
    [
        # Test 1: Top-level keys removal
        (
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "title": "Example Schema",
                "additionalProperties": False,
                "type": "object",
            },
            {"type": "object"},
        ),
        # Test 2: Preserve keys within a "properties" block
        (
            {
                "type": "object",
                "properties": {
                    "authors": {"type": "string"},
                    "title": {"type": "string"},
                    "reference": {"type": "string"},
                    "year": {"type": "string"},
                },
                "title": "This should be removed from the top-level",
            },
            {
                "type": "object",
                "properties": {
                    "authors": {"type": "string"},
                    "title": {"type": "string"},
                    "reference": {"type": "string"},
                    "year": {"type": "string"},
                },
            },
        ),
        # Test 3: Nested keys outside and inside properties block
        (
            {
                "definitions": {
                    "info": {
                        "title": "Info title",  # should be removed because it's not inside a "properties" block
                        "description": "A description",
                        "properties": {
                            "name": {
                                "title": "Name Title",
                                "type": "string",
                            },  # title here should be preserved
                            "$schema": {
                                "type": "string"
                            },  # should be preserved as it's within properties
                        },
                    }
                },
                "$schema": "http://example.com/schema",
            },
            {
                "definitions": {
                    "info": {
                        "description": "A description",
                        "properties": {
                            "name": {"title": "Name Title", "type": "string"},
                            "$schema": {"type": "string"},
                        },
                    }
                }
            },
        ),
        # Test 4: List of schemas
        (
            [
                {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                },
                {"title": "Should be removed", "type": "array"},
            ],
            [{"type": "object"}, {"type": "array"}],
        ),
    ],
)
def test_cleanup_schema(schema, expected):
    # Use a deep copy so the original test data remains unchanged.
    result = cleanup_schema(schema)
    assert result == expected


# Tests for $ref resolution - patterns that now work with nested models
@pytest.mark.parametrize(
    "schema,expected",
    [
        # Test 1: Direct model reference (Person with Address)
        (
            {
                "properties": {
                    "name": {"type": "string"},
                    "address": {"$ref": "#/$defs/Address"},
                },
                "required": ["name", "address"],
                "type": "object",
                "$defs": {
                    "Address": {
                        "properties": {
                            "street": {"type": "string"},
                            "city": {"type": "string"},
                        },
                        "required": ["street", "city"],
                        "type": "object",
                    }
                },
            },
            {
                "properties": {
                    "name": {"type": "string"},
                    "address": {
                        "properties": {
                            "street": {"type": "string"},
                            "city": {"type": "string"},
                        },
                        "required": ["street", "city"],
                        "type": "object",
                    },
                },
                "required": ["name", "address"],
                "type": "object",
            },
        ),
        # Test 2: List of models (Dogs with List[Dog])
        (
            {
                "properties": {
                    "dogs": {"items": {"$ref": "#/$defs/Dog"}, "type": "array"}
                },
                "required": ["dogs"],
                "type": "object",
                "$defs": {
                    "Dog": {
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                        },
                        "required": ["name", "age"],
                        "type": "object",
                    }
                },
            },
            {
                "properties": {
                    "dogs": {
                        "items": {
                            "properties": {
                                "name": {"type": "string"},
                                "age": {"type": "integer"},
                            },
                            "required": ["name", "age"],
                            "type": "object",
                        },
                        "type": "array",
                    }
                },
                "required": ["dogs"],
                "type": "object",
            },
        ),
        # Test 3: Optional model field
        (
            {
                "properties": {
                    "name": {"type": "string"},
                    "employer": {
                        "anyOf": [{"$ref": "#/$defs/Company"}, {"type": "null"}]
                    },
                },
                "required": ["name"],
                "type": "object",
                "$defs": {
                    "Company": {
                        "properties": {"company_name": {"type": "string"}},
                        "required": ["company_name"],
                        "type": "object",
                    }
                },
            },
            {
                "properties": {
                    "name": {"type": "string"},
                    "employer": {
                        "anyOf": [
                            {
                                "properties": {"company_name": {"type": "string"}},
                                "required": ["company_name"],
                                "type": "object",
                            },
                            {"type": "null"},
                        ]
                    },
                },
                "required": ["name"],
                "type": "object",
            },
        ),
        # Test 4: Nested composition (Customer -> List[Order] -> List[Item])
        (
            {
                "properties": {
                    "name": {"type": "string"},
                    "orders": {"items": {"$ref": "#/$defs/Order"}, "type": "array"},
                },
                "required": ["name", "orders"],
                "type": "object",
                "$defs": {
                    "Order": {
                        "properties": {
                            "items": {
                                "items": {"$ref": "#/$defs/Item"},
                                "type": "array",
                            }
                        },
                        "required": ["items"],
                        "type": "object",
                    },
                    "Item": {
                        "properties": {
                            "product_name": {"type": "string"},
                            "quantity": {"type": "integer"},
                        },
                        "required": ["product_name", "quantity"],
                        "type": "object",
                    },
                },
            },
            {
                "properties": {
                    "name": {"type": "string"},
                    "orders": {
                        "items": {
                            "properties": {
                                "items": {
                                    "items": {
                                        "properties": {
                                            "product_name": {"type": "string"},
                                            "quantity": {"type": "integer"},
                                        },
                                        "required": ["product_name", "quantity"],
                                        "type": "object",
                                    },
                                    "type": "array",
                                }
                            },
                            "required": ["items"],
                            "type": "object",
                        },
                        "type": "array",
                    },
                },
                "required": ["name", "orders"],
                "type": "object",
            },
        ),
    ],
)
def test_cleanup_schema_with_refs(schema, expected):
    """Test that $ref resolution works for various nested model patterns."""
    import copy

    result = cleanup_schema(copy.deepcopy(schema))
    assert result == expected


@pytest.mark.vcr
def test_nested_model_direct_reference():
    """Test Pattern 1: Direct model reference (Person with Address)"""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    model = llm.get_model("gemini-2.0-flash")
    response = model.prompt(
        "Create a person named Alice living in San Francisco",
        schema=Person,
        stream=False,
    )
    result = json.loads(response.text())
    assert "name" in result
    assert "address" in result
    assert "street" in result["address"]
    assert "city" in result["address"]


@pytest.mark.vcr
def test_nested_model_list():
    """Test Pattern 2: List of models (already covered by test_prompt_with_multiple_dogs)"""
    pass  # Covered by test_prompt_with_multiple_dogs


@pytest.mark.vcr
def test_nested_model_optional():
    """Test Pattern 3: Optional model field"""

    class Company(BaseModel):
        company_name: str

    class Person(BaseModel):
        name: str
        employer: Optional[Company]

    model = llm.get_model("gemini-2.0-flash")
    response = model.prompt(
        "Create a person named Bob who works at TechCorp",
        schema=Person,
        stream=False,
    )
    result = json.loads(response.text())
    assert "name" in result
    assert "employer" in result
    if result["employer"] is not None:
        assert "company_name" in result["employer"]


@pytest.mark.vcr
def test_nested_model_deep_composition():
    """Test Pattern 4: Nested composition (Customer -> Orders -> Items)"""

    class Item(BaseModel):
        product_name: str
        quantity: int

    class Order(BaseModel):
        items: List[Item]

    class Customer(BaseModel):
        name: str
        orders: List[Order]

    model = llm.get_model("gemini-2.0-flash")
    response = model.prompt(
        "Create a customer named Carol with 2 orders, each containing 2 items",
        schema=Customer,
        stream=False,
    )
    result = json.loads(response.text())
    assert "name" in result
    assert "orders" in result
    assert len(result["orders"]) > 0
    for order in result["orders"]:
        assert "items" in order
        assert len(order["items"]) > 0
        for item in order["items"]:
            assert "product_name" in item
            assert "quantity" in item


@pytest.mark.vcr
def test_cli_vertex_config(tmpdir, monkeypatch):
    """Test the new Vertex AI CLI commands"""
    user_dir = tmpdir / "llm.datasette.io"
    user_dir.mkdir()
    monkeypatch.setenv("LLM_USER_PATH", str(user_dir))
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")

    runner = CliRunner()

    # Test config command
    result = runner.invoke(cli, ["vertex", "config"])
    assert result.exit_code == 0
    assert "test-project" in result.output
    assert "us-central1" in result.output

    # Test set-project command
    result2 = runner.invoke(cli, ["vertex", "set-project", "my-gcp-project"])
    assert result2.exit_code == 0
    assert "my-gcp-project" in result2.output

    # Test set-region command
    result3 = runner.invoke(cli, ["vertex", "set-region", "europe-west1"])
    assert result3.exit_code == 0
    assert "europe-west1" in result3.output


@pytest.mark.vcr
def test_resolved_model():
    model = llm.get_model("gemini-flash-latest")
    response = model.prompt("hi")
    response.text()
    assert response.resolved_model == "gemini-2.5-flash-preview-09-2025"


@pytest.mark.vcr
def test_tools():
    model = llm.get_model("gemini-2.0-flash")
    names = ["Charles", "Sammy"]
    chain_response = model.chain(
        "Two names for a pet pelican",
        tools=[
            llm.Tool.function(lambda: names.pop(0), name="pelican_name_generator"),
        ],
    )
    text = chain_response.text()
    assert text == "Okay, here are two names for a pet pelican: Charles and Sammy.\n"
    # This one did three
    assert len(chain_response._responses) == 3
    first, second, third = chain_response._responses
    assert len(first.tool_calls()) == 1
    assert first.tool_calls()[0].name == "pelican_name_generator"
    assert len(second.tool_calls()) == 1
    assert second.tool_calls()[0].name == "pelican_name_generator"
    assert second.prompt.tool_results[0].output == "Charles"
    assert third.prompt.tool_results[0].output == "Sammy"


@pytest.mark.vcr
def test_tools_with_nested_pydantic_models():
    """Test that tools with nested Pydantic models work correctly.

    This verifies that the cleanup_schema function is applied to tool input schemas,
    which is critical for nested models that use $ref and $defs.
    """
    class Address(BaseModel):
        street: str
        city: str

    class PersonInput(BaseModel):
        name: str
        age: int
        address: Address

    def create_person(name: str, age: int, address: dict) -> str:
        return f"Created person: {name}, age {age}, living at {address['street']}, {address['city']}"

    model = llm.get_model("gemini-2.0-flash")

    # Create a tool with nested Pydantic model input schema
    tool = llm.Tool(
        name="create_person",
        description="Create a person with name, age, and address",
        input_schema=PersonInput,
        function=create_person,
    )

    # This should not raise an error about "$defs" or "$ref"
    # The cleanup_schema should remove those before sending to the API
    chain_response = model.chain(
        "Create a person named Alice, age 30, living at 123 Main St in San Francisco",
        tools=[tool],
    )

    # Verify the tool was called
    responses = chain_response._responses
    assert len(responses) >= 1

    # Check that the first response contains a tool call
    first_response = responses[0]
    tool_calls = first_response.tool_calls()
    assert len(tool_calls) >= 1
    assert tool_calls[0].name == "create_person"


@pytest.mark.vcr
def test_tools_with_gemini_3_thought_signatures(monkeypatch):
    """Test that tool calls with Gemini 3 thought signatures are preserved.

    Gemini 3 models include thoughtSignature fields on functionCall parts during
    streaming. This test verifies that:
    1. The tool call is executed correctly
    2. function_call_parts with thoughtSignature are stored in response_json
    3. original_model_parts are stored for exact API response preservation
    """
    monkeypatch.setenv("GOOGLE_CLOUD_API_KEY", "test-api-key")

    def multiply(a: int, b: int):
        """Multiply two integers"""
        return str(a * b)

    model = llm.get_model("vertex/gemini-3-flash-preview")
    chain_response = model.chain(
        "What is 7 * 3?",
        tools=[llm.Tool.function(multiply, name="multiply")],
    )
    text = chain_response.text()
    assert text == "7 * 3 is 21.\n"

    # Should have 2 responses: tool call + final answer
    assert len(chain_response._responses) == 2
    first, second = chain_response._responses

    # First response should have a tool call
    assert len(first.tool_calls()) == 1
    assert first.tool_calls()[0].name == "multiply"
    assert first.tool_calls()[0].arguments == {"a": 7, "b": 3}

    # Second response should have the tool result
    assert second.prompt.tool_results[0].output == "21"

    # Verify function_call_parts with thoughtSignature are preserved
    fc_parts = first.response_json.get("function_call_parts")
    assert fc_parts is not None
    assert len(fc_parts) == 1
    assert fc_parts[0]["functionCall"]["name"] == "multiply"
    assert "thoughtSignature" in fc_parts[0]

    # Verify original_model_parts are stored
    original_parts = first.response_json.get("original_model_parts")
    assert original_parts is not None
    assert len(original_parts) >= 2  # thinking trace + functionCall
    # Check thinking trace
    thinking_parts = [p for p in original_parts if p.get("thought")]
    assert len(thinking_parts) >= 1
    # Check functionCall part
    fc_original = [p for p in original_parts if "functionCall" in p]
    assert len(fc_original) == 1
    assert "thoughtSignature" in fc_original[0]


def test_recursive_schema_detection_direct():
    """Test that direct recursion is detected and raises an error."""
    # Direct recursion: Node has a next field that references Node
    schema = {
        "properties": {
            "value": {"type": "integer"},
            "next": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}]},
        },
        "required": ["value"],
        "type": "object",
        "$defs": {
            "Node": {
                "properties": {
                    "value": {"type": "integer"},
                    "next": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}]},
                },
                "required": ["value"],
                "type": "object",
            }
        },
    }

    with pytest.raises(ValueError) as exc_info:
        cleanup_schema(copy.deepcopy(schema))

    assert "Recursive schema detected" in str(exc_info.value)
    assert "Node" in str(exc_info.value)


def test_recursive_schema_detection_indirect():
    """Test that indirect recursion is detected and raises an error."""
    # Indirect recursion: A -> B -> A
    schema = {
        "properties": {
            "name": {"type": "string"},
            "b_ref": {"$ref": "#/$defs/B"},
        },
        "required": ["name"],
        "type": "object",
        "$defs": {
            "A": {
                "properties": {
                    "name": {"type": "string"},
                    "b_ref": {"$ref": "#/$defs/B"},
                },
                "required": ["name"],
                "type": "object",
            },
            "B": {
                "properties": {
                    "value": {"type": "integer"},
                    "a_ref": {"$ref": "#/$defs/A"},
                },
                "required": ["value"],
                "type": "object",
            },
        },
    }

    with pytest.raises(ValueError) as exc_info:
        cleanup_schema(copy.deepcopy(schema))

    assert "Recursive schema detected" in str(exc_info.value)


def test_recursive_schema_detection_pydantic():
    """Test that recursive Pydantic models are detected and raise an error."""
    from typing import Optional
    from pydantic import BaseModel

    class Node(BaseModel):
        value: int
        next: Optional["Node"] = None

    # Get the schema from the Pydantic model
    schema = Node.model_json_schema()

    with pytest.raises(ValueError) as exc_info:
        cleanup_schema(copy.deepcopy(schema))

    assert "Recursive schema detected" in str(exc_info.value)


def test_youtube_url_detection():
    """Test that YouTube URLs are correctly detected, including Shorts."""
    from llm_vertex import is_youtube_url

    # Regular YouTube URLs
    assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert is_youtube_url("http://youtube.com/watch?v=dQw4w9WgXcQ")
    assert is_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ")

    # YouTube short URLs
    assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ")
    assert is_youtube_url("http://youtu.be/dQw4w9WgXcQ")

    # YouTube embed URLs
    assert is_youtube_url("https://www.youtube.com/embed/dQw4w9WgXcQ")
    assert is_youtube_url("http://youtube.com/embed/dQw4w9WgXcQ")

    # YouTube Shorts URLs
    assert is_youtube_url("https://www.youtube.com/shorts/dQw4w9WgXcQ")
    assert is_youtube_url("http://youtube.com/shorts/dQw4w9WgXcQ")
    assert is_youtube_url("https://youtube.com/shorts/dQw4w9WgXcQ")

    # Non-YouTube URLs
    assert not is_youtube_url("https://example.com/watch?v=dQw4w9WgXcQ")
    assert not is_youtube_url("https://vimeo.com/123456")
    assert not is_youtube_url("https://www.google.com")
    assert not is_youtube_url("")
    assert not is_youtube_url(None)


class TestMergeStreamingParts:
    """Tests for _SharedGemini._merge_streaming_parts()."""

    def test_thinking_text_merged_with_thought_signature_and_function_call(self):
        """Thinking text deltas merged with thoughtSignature preserved, functionCall kept intact."""
        from llm_vertex import _SharedGemini

        gathered = [
            {
                "candidates": [{
                    "content": {
                        "parts": [
                            {"thought": True, "text": "Let me think"},
                            {"thought": True, "text": " about this", "thoughtSignature": "sig-abc"},
                        ]
                    }
                }]
            },
            {
                "candidates": [{
                    "content": {
                        "parts": [
                            {
                                "functionCall": {"name": "load_github", "args": {"url": "http://example.com"}},
                                "thoughtSignature": "sig-fc-123",
                            }
                        ]
                    }
                }]
            },
        ]
        result = _SharedGemini._merge_streaming_parts(gathered)
        assert len(result) == 2
        # Thinking parts merged
        assert result[0]["thought"] is True
        assert result[0]["text"] == "Let me think about this"
        assert result[0]["thoughtSignature"] == "sig-abc"
        # functionCall kept intact with its thoughtSignature
        assert result[1]["functionCall"]["name"] == "load_github"
        assert result[1]["thoughtSignature"] == "sig-fc-123"

    def test_regular_text_merged_and_function_call_signature_preserved(self):
        """Regular text deltas merged, functionCall with thoughtSignature kept."""
        from llm_vertex import _SharedGemini

        gathered = [
            {
                "candidates": [{
                    "content": {
                        "parts": [
                            {"text": "Hello"},
                        ]
                    }
                }]
            },
            {
                "candidates": [{
                    "content": {
                        "parts": [
                            {"text": " world"},
                        ]
                    }
                }]
            },
            {
                "candidates": [{
                    "content": {
                        "parts": [
                            {
                                "functionCall": {"name": "do_stuff", "args": {}},
                                "thoughtSignature": "sig-xyz",
                            }
                        ]
                    }
                }]
            },
        ]
        result = _SharedGemini._merge_streaming_parts(gathered)
        assert len(result) == 2
        assert result[0]["text"] == "Hello world"
        assert "thought" not in result[0]
        assert result[1]["functionCall"]["name"] == "do_stuff"
        assert result[1]["thoughtSignature"] == "sig-xyz"

    def test_last_event_no_candidates_parts_still_captured(self):
        """Last event has only usageMetadata (no candidates) -- parts from earlier events still captured."""
        from llm_vertex import _SharedGemini

        gathered = [
            {
                "candidates": [{
                    "content": {
                        "parts": [
                            {"thought": True, "text": "thinking..."},
                        ]
                    }
                }]
            },
            {
                "candidates": [{
                    "content": {
                        "parts": [
                            {"thought": True, "text": " done", "thoughtSignature": "sig-final"},
                        ]
                    }
                }]
            },
            {
                "candidates": [{
                    "content": {
                        "parts": [
                            {
                                "functionCall": {"name": "my_tool", "args": {"x": 1}},
                                "thoughtSignature": "sig-tool",
                            }
                        ]
                    }
                }]
            },
            # Final event: only usageMetadata, no candidates
            {
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20},
                "modelVersion": "gemini-3-flash-preview",
            },
        ]
        result = _SharedGemini._merge_streaming_parts(gathered)
        assert len(result) == 2
        assert result[0]["thought"] is True
        assert result[0]["text"] == "thinking... done"
        assert result[0]["thoughtSignature"] == "sig-final"
        assert result[1]["functionCall"]["name"] == "my_tool"
        assert result[1]["thoughtSignature"] == "sig-tool"

    def test_no_merging_across_thought_boundary(self):
        """Text parts with different thought status are not merged."""
        from llm_vertex import _SharedGemini

        gathered = [
            {
                "candidates": [{
                    "content": {
                        "parts": [
                            {"thought": True, "text": "thinking part"},
                        ]
                    }
                }]
            },
            {
                "candidates": [{
                    "content": {
                        "parts": [
                            {"text": "regular text"},
                        ]
                    }
                }]
            },
        ]
        result = _SharedGemini._merge_streaming_parts(gathered)
        assert len(result) == 2
        assert result[0]["thought"] is True
        assert result[0]["text"] == "thinking part"
        assert result[1]["text"] == "regular text"
        assert "thought" not in result[1]

    def test_empty_gathered(self):
        """Empty gathered list returns empty parts."""
        from llm_vertex import _SharedGemini

        result = _SharedGemini._merge_streaming_parts([])
        assert result == []

    def test_deep_copy_preserves_isolation(self):
        """Modifying the result does not affect the original gathered data."""
        from llm_vertex import _SharedGemini

        gathered = [
            {
                "candidates": [{
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "test", "args": {"key": "value"}}, "thoughtSignature": "sig-1"}
                        ]
                    }
                }]
            }
        ]
        result = _SharedGemini._merge_streaming_parts(gathered)
        result[0]["functionCall"]["name"] = "modified"
        # Original should be untouched
        assert gathered[0]["candidates"][0]["content"]["parts"][0]["functionCall"]["name"] == "test"
