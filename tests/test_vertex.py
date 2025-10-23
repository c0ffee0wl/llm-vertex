from click.testing import CliRunner
import llm
from llm.cli import cli
import nest_asyncio
import json
import os
import pytest
import pydantic
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
