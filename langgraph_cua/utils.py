from typing import Any, Union

from langchain_core.runnables import RunnableConfig
from scrapybara import Scrapybara
from scrapybara.client import BrowserInstance, UbuntuInstance, WindowsInstance
from hyperbrowser import Hyperbrowser
from hyperbrowser.models import SessionDetail
from .types import Provider, get_configuration_with_defaults


def get_hyperbrowser_client(api_key: str) -> Hyperbrowser:
    """
    Gets the Hyperbrowser client, using the API key provided.

    Args:
        api_key: The API key for Hyperbrowser.
    """
    if not api_key:
        raise ValueError(
            "Hyperbrowser API key not provided. Please provide one in the configurable fields, "
            "or set it as an environment variable (HYPERBROWSER_API_KEY)"
        )
    client = Hyperbrowser(api_key=api_key)
    return client


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


def get_instance(
    id: str, config: RunnableConfig
) -> Union[UbuntuInstance, BrowserInstance, WindowsInstance, SessionDetail]:
    """
    Gets an instance by its ID from Scrapybara.

    Args:
        id: The ID of the instance to get.
        config: The configuration for the runnable.

    Returns:
        The instance.
    """
    configuration = get_configuration_with_defaults(config)
    provider = configuration.get("provider")
    if provider == Provider.Scrapybara:
        scrapybara_api_key = configuration.get("scrapybara_api_key")
        client = get_scrapybara_client(scrapybara_api_key)
        return client.get(id)
    elif provider == Provider.Hyperbrowser:
        hyperbrowser_api_key = configuration.get("hyperbrowser_api_key")
        client = get_hyperbrowser_client(hyperbrowser_api_key)
        return client.sessions.get(id)
    else:
        raise ValueError(
            f"Invalid provider. Must be one of {Provider.Scrapybara} or {Provider.Hyperbrowser}. Received: {provider}"
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

    return any(output.get("type") == "computer_call" for output in tool_outputs)
