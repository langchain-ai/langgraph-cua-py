import os
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from hyperbrowser.models import CreateSessionParams, ScreenConfig
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import add_messages
from playwright.async_api import Browser, Playwright, Page

DEFAULT_DISPLAY_WIDTH = 1024
DEFAULT_DISPLAY_HEIGHT = 800


class Output(TypedDict):
    """
    A computer screenshot image used with the computer use tool.
    """

    type: Literal["computer_screenshot"]  # Always "computer_screenshot"
    file_id: Optional[str]  # The identifier of an uploaded file that contains the screenshot
    image_url: Optional[str]  # The URL of the screenshot image


class AcknowledgedSafetyCheck(TypedDict):
    """
    A pending safety check for the computer call.
    """

    id: str  # The ID of the pending safety check
    code: str  # The type of the pending safety check
    message: str  # Details about the pending safety check


class ComputerCallOutput(TypedDict):
    """
    The output of a computer tool call.
    """

    call_id: str  # The ID of the computer tool call that produced the output
    output: Output  # A computer screenshot image used with the computer use tool
    type: Literal["computer_call_output"]  # Always "computer_call_output"
    id: Optional[str]  # The ID of the computer tool call output
    acknowledged_safety_checks: Optional[
        List[AcknowledgedSafetyCheck]
    ]  # Safety checks acknowledged by the developer
    status: Optional[
        Literal["in_progress", "completed", "incomplete"]
    ]  # Status of the message input


class CUAState(TypedDict):
    """State schema for the computer use agent.
    Attributes:
        messages: The messages between the user and assistant.
        session_id: The ID of the session to use for this thread.
        stream_url: The URL to the live-stream of the virtual machine.
    """

    messages: Annotated[list[AnyMessage], add_messages] = []
    session_id: Annotated[Optional[str], None] = None
    stream_url: Annotated[Optional[str], None] = None
    playwright: Annotated[Optional[Playwright], None] = None
    browser: Annotated[Optional[Browser], None] = None
    current_page: Annotated[Optional[Page], None] = None


class CUAConfiguration(TypedDict):
    """Configuration for the Computer Use Agent.
    Attributes:
        hyperbrowser_api_key: The API key to use for Hyperbrowser.
            This can be provided in the configuration, or set as an environment variable (HYPERBROWSER_API_KEY).
    """

    hyperbrowser_api_key: str  # API key for Hyperbrowser


def get_configuration_with_defaults(config: RunnableConfig) -> Dict[str, Any]:
    """
    Gets the configuration with defaults for the graph.
    Args:
        config: The configuration for the runnable.
    Returns:
        Dict with configuration values including defaults.
    """

    configurable_fields = config.get("configurable", {})
    hyperbrowser_api_key = (
        configurable_fields.get("hyperbrowser_api_key")
        or config.get("hyperbrowser_api_key")
        or os.environ.get("HYPERBROWSER_API_KEY")
    )
    session_params = configurable_fields.get("session_params")
    if not session_params:
        session_params = {}
    if not session_params.get("screen"):
        session_params["screen"] = {
            "width": DEFAULT_DISPLAY_WIDTH,
            "height": DEFAULT_DISPLAY_HEIGHT,
        }

    return {"hyperbrowser_api_key": hyperbrowser_api_key, "session_params": session_params}
