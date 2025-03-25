import asyncio
import base64
from typing import Any, Dict, Optional
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from playwright.async_api import async_playwright, Browser, Playwright, Page
from openai.types.responses.response_computer_tool_call import ResponseComputerToolCall

from .tools import handle_computer_call, handle_function_tool_call

from ..types import CUAState
from ..utils import get_browser_session, is_computer_tool_call


async def take_browser_action(state: CUAState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Executes browser actions based on the tool call in the last message.
    Args:
        state: The current state of the CUA agent.
        config: The runnable configuration.
    Returns:
        A dictionary with updated state information.
    """
    message: AnyMessage = state.get("messages", [])[-1]
    assert message.type == "ai", "Last message must be an AI message"
    tool_outputs = message.additional_kwargs.get("tool_outputs", [])
    tool_calls = message.tool_calls

    if not is_computer_tool_call(tool_outputs) and len(tool_calls) == 0:
        # This should never happen, but include the check for proper type safety.
        raise ValueError(
            "Cannot take computer action without a computer call or function call in the last message."
        )

    tool_outputs: list[ResponseComputerToolCall] = tool_outputs

    # Reuse existing Playwright and browser instances if available
    playwright: Optional[Playwright] = state.get("playwright")
    browser: Optional[Browser] = state.get("browser")
    session_id: Optional[str] = state.get("session_id")
    stream_url: Optional[str] = state.get("stream_url")

    if not session_id:
        raise ValueError("Session ID not found in state.")

    # Initialize Playwright and browser if not already available
    if not playwright or not browser:
        session = await get_browser_session(session_id, config)
        playwright = await async_playwright().start()
        browser = await playwright.chromium.connect_over_cdp(
            f"{session.ws_endpoint}&keepAlive=true"
        )
        print("Playwright connected successfully")

    current_context = browser.contexts[0]
    page = state.get("current_page", current_context.pages[0])

    def handle_page_event(newPage: Page):
        nonlocal page
        page = newPage

    current_context.on("page", handle_page_event)

    tool_message: Optional[ToolMessage] = None

    for tool_output in tool_outputs:
        if tool_output.get("type") == "computer_call":
            await handle_computer_call(page, tool_output)
            await asyncio.sleep(1)
            screenshot = await page.screenshot()
            b64_screenshot = base64.b64encode(screenshot).decode("utf-8")
            screenshot_url = f"data:image/png;base64,{b64_screenshot}"

            output_content = {
                "type": "input_image",
                "image_url": screenshot_url,
            }
            tool_message = ToolMessage(
                content=[output_content],
                tool_call_id=tool_output.get("call_id"),
                additional_kwargs={"type": "computer_call_output"},
            )
        else:
            print("unknown tool output type", tool_output)

    for tool_call in tool_calls:
        tool_message = await handle_function_tool_call(page, tool_call)
        await asyncio.sleep(1)

    return {
        "messages": tool_message if tool_message else None,
        "session_id": session_id,
        "stream_url": stream_url,
        "playwright": playwright,
        "browser": browser,
        "current_page": page,
    }
