from typing import Any, Dict, Optional, Union, Literal

from langchain_core.messages import AIMessageChunk, SystemMessage
from langchain_core.runnables.config import RunnableConfig
# from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field


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


def _prompt_to_sys_message(prompt: Union[str, SystemMessage, None]):
    if prompt is None:
        return None
    if isinstance(prompt, str):
        return {"role": "system", "content": prompt}
    return prompt


async def call_model(state: CUAState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Invokes the computer preview model with the given messages.

    Args:
        state: The current state of the thread.

    Returns:
        The updated state with the model's response.
    """
    configuration = get_configuration_with_defaults(config)
    environment = configuration.get("environment")
    zdr_enabled = configuration.get("zdr_enabled")
    prompt = _prompt_to_sys_message(configuration.get("prompt"))
    messages = state.get("messages", [])
    previous_response_id: Optional[str] = None
    last_message = messages[-1] if messages else None

    # Check if the last message is a tool message
    if last_message and getattr(last_message, "type", None) == "tool" and zdr_enabled is False:
        # If it's a tool message, check if the second-to-last message is an AI message
        if (
            len(messages) >= 2
            and getattr(messages[-2], "type", None) == "ai"
            and hasattr(messages[-2], "response_metadata")
        ):
            previous_response_id = messages[-2].response_metadata["id"]

    llm = ChatAnthropic(model="claude-sonnet-4-5",
                            temperature=0,
                            betas=["computer-use-2025-01-24"])
    tool = {
        "type": "computer_20250124",
        "name": "computer",
        "display_width_px": DEFAULT_DISPLAY_WIDTH,
        "display_height_px": DEFAULT_DISPLAY_HEIGHT,
        "display_number": 1,
    }
    llm_with_tools = llm.bind_tools([tool])

    response: AIMessageChunk

    # Check if the last message is a tool message
    if last_message and getattr(last_message, "type", None) == "tool" and zdr_enabled is False:
        if previous_response_id is None:
            raise ValueError("Cannot process tool message without a previous_response_id")

        # Only pass the tool message to the model
        response = await llm_with_tools.ainvoke([last_message])
    else:
        # Pass all messages to the model
        if prompt is None:
            response = await llm_with_tools.ainvoke(messages)
        else:
            response = await llm_with_tools.ainvoke([prompt, *messages])

    return {
        "messages": response,
    }
