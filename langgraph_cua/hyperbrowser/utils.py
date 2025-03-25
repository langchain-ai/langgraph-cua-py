import os
from typing import Any, Dict, Optional, Union

from playwright.async_api import async_playwright, Browser, Playwright
from hyperbrowser import AsyncHyperbrowser
from langchain_core.runnables import RunnableConfig
from hyperbrowser.models import SessionDetail
from .types import get_configuration_with_defaults, CUAState


def get_hyperbrowser_client(api_key: str) -> AsyncHyperbrowser:
    """
    Gets the Hyperbrowser client, using the API key provided.
    Args:
        api_key: The API key for Hyperbrowser.
    Returns:
        The Hyperbrowser client.
    """
    if not api_key:
        raise ValueError(
            "Hyperbrowser API key not provided. Please provide one in the configurable fields, "
            "or set it as an environment variable (HYPERBROWSER_API_KEY)"
        )
    client = AsyncHyperbrowser(api_key=api_key)
    return client


async def get_browser_session(id: str, config: RunnableConfig) -> SessionDetail:
    """
    Gets a browser session by its ID from Hyperbrowser.

    Args:
        id: The ID of the browser session to get.
        config: The configuration for the runnable.

    Returns:
        The browser session.
    """
    configuration = get_configuration_with_defaults(config)
    hyperbrowser_api_key = configuration.get("hyperbrowser_api_key")
    client = get_hyperbrowser_client(hyperbrowser_api_key)
    return await client.sessions.get(id)


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


async def start_playwright(state: CUAState, session: Optional[SessionDetail] = None):
    session_id = state.get("session_id")
    playwright: Optional[Playwright] = state.get("playwright")
    browser: Optional[Browser] = state.get("browser")

    if playwright and browser:
        return playwright, browser, session

    if not session:
        session = await get_browser_session(session_id)

    if not playwright:
        playwright = await async_playwright().start()
    if not browser:
        browser = await playwright.chromium.connect_over_cdp(
            f"{session.ws_endpoint}&keepAlive=true"
        )

    return playwright, browser, session
