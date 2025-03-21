from typing import Any, Dict, Union

from langchain_core.runnables import RunnableConfig
from scrapybara import Scrapybara
from scrapybara.client import BrowserInstance, UbuntuInstance, WindowsInstance

from .types import get_configuration_with_defaults

# Copied from the OpenAI example repository
# https://github.com/openai/openai-cua-sample-app/blob/eb2d58ba77ffd3206d3346d6357093647d29d99c/utils.py#L13
BLOCKED_DOMAINS = [
    "maliciousbook.com",
    "evilvideos.com",
    "darkwebforum.com",
    "shadytok.com",
    "suspiciouspins.com",
    "ilanbigio.com",
]


def get_scrapybara_client(api_key: str) -> Scrapybara:
    """
    Gets the Scrapybara client, using the API key provided.

    Args:
        api_key: The API key for Scrapybara.

    Returns:
        The Scrapybara client.
    """
    if not api_key:
        raise ValueError(
            "Scrapybara API key not provided. Please provide one in the configurable fields, "
            "or set it as an environment variable (SCRAPYBARA_API_KEY)"
        )
    client = Scrapybara(api_key=api_key)
    return client


def init_or_load(
    inputs: Dict[str, Any], config: RunnableConfig
) -> Union[UbuntuInstance, BrowserInstance, WindowsInstance]:
    """
    Initializes or loads an instance based on the inputs provided.

    Args:
        inputs: Dictionary containing instanceId and environment to use in the virtual machine.
        config: The configuration for the runnable.

    Returns:
        The initialized or loaded instance.
    """

    instance_id = inputs.get("instance_id")
    environment = inputs.get("environment", "web")

    configuration = get_configuration_with_defaults(config)
    scrapybara_api_key = configuration.get("scrapybara_api_key")
    timeout_hours = configuration.get("timeout_hours")

    if not scrapybara_api_key:
        raise ValueError(
            "Scrapybara API key not provided. Please provide one in the configurable fields, "
            "or set it as an environment variable (SCRAPYBARA_API_KEY)"
        )

    client = get_scrapybara_client(scrapybara_api_key)

    if instance_id:
        return client.get(instance_id)

    if environment == "ubuntu":
        return client.start_ubuntu(timeout_hours=timeout_hours)
    elif environment == "windows":
        return client.start_windows(timeout_hours=timeout_hours)
    elif environment == "web":
        blocked_domains = [
            domain.replace("https://", "").replace("www.", "") for domain in BLOCKED_DOMAINS
        ]
        return client.start_browser(timeout_hours=timeout_hours, blocked_domains=blocked_domains)

    raise ValueError(
        f"Invalid environment. Must be one of 'web', 'ubuntu', or 'windows'. Received: {environment}"
    )


def is_computer_tool_call(tool_outputs: Any) -> bool:
    """
    Checks if the given tool outputs are a computer call.

    Args:
        tool_outputs: The tool outputs to check.

    Returns:
        True if the tool outputs are a computer call, false otherwise.
    """
    if not tool_outputs or not isinstance(tool_outputs, list):
        return False

    return all(output.get("type") == "computer_call" for output in tool_outputs)
