import os
import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


def test_openrouter_initialization():
    """Test that ChatOpenAI can be initialized with OpenRouter configuration."""
    # Test that we can create a ChatOpenAI instance with OpenRouter settings
    llm = ChatOpenAI(
        model="x-ai/grok-4.1-fast:free",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=1000,
    )

    # Verify the instance was created successfully
    assert llm is not None
    assert llm.model_name == "x-ai/grok-4.1-fast:free"
    assert llm.openai_api_base == "https://openrouter.ai/api/v1"


@pytest.mark.asyncio
async def test_openrouter_basic_call():
    """Test a basic API call to OpenRouter (requires valid API key)."""
    # Check for OpenRouter API key (prefer OPENROUTER_API_KEY, fallback to OPENAI_API_KEY if it looks like OpenRouter key)
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.startswith("sk-or-v1-"):
        pytest.skip("Valid OpenRouter API key not found in OPENROUTER_API_KEY or OPENAI_API_KEY environment variables")

    llm = ChatOpenAI(
        model="x-ai/grok-4.1-fast:free",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        max_tokens=100,
    )

    # Test a simple message
    messages = [{"role": "user", "content": "Hello, can you respond with just 'OpenRouter test successful'?"}]

    try:
        response = await llm.ainvoke(messages)
        assert response is not None
        assert hasattr(response, 'content')
        assert len(response.content) > 0
        # Check that the response contains expected text (case insensitive)
        assert "openrouter" in response.content.lower() or "successful" in response.content.lower()
    except Exception as e:
        # If the API call fails due to invalid key or other issues, that's expected
        # The test is mainly to verify the integration setup works
        pytest.fail(f"OpenRouter API call failed: {e}")


def test_openrouter_with_tools():
    """Test that ChatOpenAI can be configured with tools for OpenRouter."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")

    llm = ChatOpenAI(
        model="x-ai/grok-4.1-fast:free",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        max_tokens=1000,
    )

    # Define a simple tool
    tool = {
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "A test tool for OpenRouter integration",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Test message"}
                },
                "required": ["message"]
            }
        }
    }

    # Bind tools
    llm_with_tools = llm.bind_tools([tool])

    # Verify the instance was created
    assert llm_with_tools is not None