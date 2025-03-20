import os
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import add_messages


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
        instance_id: The ID of the instance to use for this thread.
        environment: The environment to use. Default is "web".
        stream_url: The URL to the live-stream of the virtual machine.
    """

    messages: Annotated[list[AnyMessage], add_messages] = []
    instance_id: Annotated[Optional[str], None] = None
    environment: Annotated[Literal["web", "ubuntu", "windows"], "web"] = "web"
    stream_url: Annotated[Optional[str], None] = None


class CUAConfiguration(TypedDict):
    """Configuration for the Computer Use Agent.

    Attributes:
        scrapybara_api_key: The API key to use for Scrapybara.
            This can be provided in the configuration, or set as an environment variable (SCRAPYBARA_API_KEY).
        timeout_hours: The number of hours to keep the virtual machine running before it times out.
            Must be between 0.01 and 24. Default is 1.
        zdr_enabled: Whether or not Zero Data Retention is enabled in the user's OpenAI account. Default False.
    """

    scrapybara_api_key: str  # API key for Scrapybara
    timeout_hours: float  # Timeout in hours (0.01-24, default: 1)
    zdr_enabled: bool  # True/False for whether or not ZDR is enabled.


def get_configuration_with_defaults(config: RunnableConfig) -> Dict[str, Any]:
    """
    Gets the configuration with defaults for the graph.

    Args:
        config: The configuration for the runnable.

    Returns:
        Dict with configuration values including defaults.
    """

    configurable_fields = config.get("configurable", {})
    scrapybara_api_key = (
        configurable_fields.get("scrapybara_api_key")
        or config.get("scrapybara_api_key")
        or os.environ.get("SCRAPYBARA_API_KEY")
    )
    timeout_hours = configurable_fields.get("timeout_hours", 1)
    zdr_enabled = configurable_fields.get("zdr_enabled", False)

    return {
        "scrapybara_api_key": scrapybara_api_key,
        "timeout_hours": timeout_hours,
        "zdr_enabled": zdr_enabled,
    }
