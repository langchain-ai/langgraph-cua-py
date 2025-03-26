from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import AIMessageChunk, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI

from ..types import CUAState, Provider, get_configuration_with_defaults


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


def get_available_tools(configuration: Dict[str, Any]) -> List[Dict[str, Any]]:
    provider = configuration.get("provider")
    if provider == Provider.Scrapybara:
        return [
            {
                "type": "computer_use_preview",
                "display_width": DEFAULT_DISPLAY_WIDTH,
                "display_height": DEFAULT_DISPLAY_HEIGHT,
                "environment": get_openai_env_from_state_env(configuration.get("environment")),
            }
        ]
    elif provider == Provider.Hyperbrowser:
        session_params = configuration.get("session_params", {})
        screen_config = (
            session_params.get(
                "screen_config", {"width": DEFAULT_DISPLAY_WIDTH, "height": DEFAULT_DISPLAY_HEIGHT}
            )
            if session_params
            else {"width": DEFAULT_DISPLAY_WIDTH, "height": DEFAULT_DISPLAY_HEIGHT}
        )

        return [
            {
                "type": "computer_use_preview",
                "display_width": screen_config.get("width", DEFAULT_DISPLAY_WIDTH),
                "display_height": screen_config.get("height", DEFAULT_DISPLAY_HEIGHT),
                "environment": "browser",
            },
            {
                "type": "function",
                "function": {
                    "name": "go_to_url",
                    "description": "Navigate to a URL. Can be used when on a blank page to go to a specific URL or search engine.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The fully qualified URL to navigate to",
                            },
                        },
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_url",
                    "description": "Get the current URL",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
        ]
    else:
        raise ValueError(f"Unknown provider: {provider}")


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
        model="computer-use-preview",
        model_kwargs={"truncation": "auto", "previous_response_id": previous_response_id},
    )

    tools = get_available_tools(configuration)

    llm_with_tools = llm.bind_tools(tools)

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
