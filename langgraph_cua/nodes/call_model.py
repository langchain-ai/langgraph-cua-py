from typing import Any, Dict, Optional

from langchain_core.messages import AIMessageChunk
from langchain_openai import ChatOpenAI

from ..types import CUAState, get_configuration_with_defaults


def get_openai_env_from_state_env(env: str) -> str:
    """
    Converts one of "web", "ubuntu", or "windows" to OpenAI environment string.

    Args:
        env: The environment to convert.

    Returns:
        The corresponding OpenAI environment string.

    Raises:
        ValueError: If the environment is invalid.
    """
    if env == "web":
        return "browser"
    elif env == "ubuntu":
        return "ubuntu"
    elif env == "windows":
        return "windows"


# Scrapybara does not allow for configuring this. Must use a hardcoded value.
DEFAULT_DISPLAY_WIDTH = 1024
DEFAULT_DISPLAY_HEIGHT = 768


async def call_model(state: CUAState) -> Dict[str, Any]:
    """
    Invokes the computer preview model with the given messages.

    Args:
        state: The current state of the thread.

    Returns:
        The updated state with the model's response.
    """
    config = get_configuration_with_defaults(state)
    messages = state.get("messages", [])
    previous_response_id: Optional[str] = None
    last_message = messages[-1] if messages else None

    # Check if the last message is a tool message
    if (
        last_message
        and getattr(last_message, "type", None) == "tool"
        and config["zdr_enabled"] is False
    ):
        # If it's a tool message, check if the second-to-last message is an AI message
        if (
            len(messages) >= 2
            and getattr(messages[-2], "type", None) == "ai"
            and hasattr(messages[-2], "response_metadata")
        ):
            previous_response_id = messages[-2].response_metadata["id"]

    llm = ChatOpenAI(
        model="computer-use-preview",
        model_kwargs={"truncation": "auto", "previous_response_id": previous_response_id},
    )

    tool = {
        "type": "computer_use_preview",
        "display_width": DEFAULT_DISPLAY_WIDTH,
        "display_height": DEFAULT_DISPLAY_HEIGHT,
        "environment": get_openai_env_from_state_env(state.get("environment", "web")),
    }
    llm_with_tools = llm.bind_tools([tool])

    response: AIMessageChunk

    # Check if the last message is a tool message
    if (
        last_message
        and getattr(last_message, "type", None) == "tool"
        and config["zdr_enabled"] is False
    ):
        if previous_response_id is None:
            raise ValueError("Cannot process tool message without a previous_response_id")

        # Only pass the tool message to the model
        response = await llm_with_tools.ainvoke([last_message])
    else:
        # Pass all messages to the model
        response = await llm_with_tools.ainvoke(messages)

    return {
        "messages": response,
    }
