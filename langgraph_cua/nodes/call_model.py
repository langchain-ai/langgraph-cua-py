import json
import os
from typing import Any, Dict, Optional, Union

from langchain_core.messages import AIMessageChunk, SystemMessage
from langchain_core.runnables.config import RunnableConfig
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

    llm = ChatOpenAI(
        model="x-ai/grok-4.1-fast:free",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=4000,
    )

    tool = {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": "Perform actions on the computer such as clicking, typing, scrolling, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["click", "double_click", "drag", "keypress", "move", "screenshot", "wait", "scroll", "type"],
                        "description": "The type of action to perform"
                    },
                    "x": {"type": "number", "description": "X coordinate for mouse actions"},
                    "y": {"type": "number", "description": "Y coordinate for mouse actions"},
                    "text": {"type": "string", "description": "Text to type"},
                    "button": {"type": "string", "description": "Mouse button (left, right, middle)"},
                    "keys": {"type": "array", "items": {"type": "string"}, "description": "Keys to press"},
                    "path": {"type": "array", "items": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}}, "description": "Path for drag action"},
                    "scroll_x": {"type": "number", "description": "Horizontal scroll amount"},
                    "scroll_y": {"type": "number", "description": "Vertical scroll amount"},
                    "environment": {"type": "string", "description": "Environment type"},
                    "display_width": {"type": "number", "description": "Display width"},
                    "display_height": {"type": "number", "description": "Display height"},
                },
                "required": ["action"]
            }
        }
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
        "tool_outputs": response.additional_kwargs.get("tool_calls", []),
    }
