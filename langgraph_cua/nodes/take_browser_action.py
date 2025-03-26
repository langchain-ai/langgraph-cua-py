import base64
from math import floor
from random import random
from hyperbrowser.models import SessionDetail
import time
from typing import Any, Dict, Optional
from langchain_core.messages import AnyMessage, ToolMessage, ToolCall
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from playwright.sync_api import Page
from openai.types.responses.response_computer_tool_call import ResponseComputerToolCall

from ..utils import get_instance, is_computer_tool_call
from ..types import CUAState

CUA_KEY_TO_PLAYWRIGHT_KEY = {
    "/": "Divide",
    "\\": "Backslash",
    "alt": "Alt",
    "arrowdown": "ArrowDown",
    "arrowleft": "ArrowLeft",
    "arrowright": "ArrowRight",
    "arrowup": "ArrowUp",
    "backspace": "Backspace",
    "capslock": "CapsLock",
    "cmd": "Meta",
    "ctrl": "Control",
    "delete": "Delete",
    "end": "End",
    "enter": "Enter",
    "esc": "Escape",
    "home": "Home",
    "insert": "Insert",
    "option": "Alt",
    "pagedown": "PageDown",
    "pageup": "PageUp",
    "shift": "Shift",
    "space": " ",
    "super": "Meta",
    "tab": "Tab",
    "win": "Meta",
}

DUMMY_SCREENSHOT = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKp//2Q=="


def _translate_key(key: str) -> str:
    return CUA_KEY_TO_PLAYWRIGHT_KEY.get(key.lower(), key)


def handle_function_tool_call(page: Page, function_tool_call: ToolCall) -> ToolMessage:
    name = function_tool_call.get("name")
    arguments = function_tool_call.get("args")
    call_id = function_tool_call.get("id")

    try:
        if name == "go_to_url":
            page.goto(arguments.get("url"), timeout=15000, wait_until="domcontentloaded")
            time.sleep(1)
            return ToolMessage(
                tool_call_id=call_id,
                content={"message": f"Navigated to {arguments.get('url')}"},
                additional_kwargs={"type": "function_call_output"},
            )
        elif name == "get_current_url":
            return ToolMessage(
                tool_call_id=call_id,
                content={"message": f"The current URL is {page.url}"},
                additional_kwargs={"type": "function_call_output"},
            )
        else:
            raise ValueError(f"Unknown function call name: {name}")
    except Exception as e:
        print(f"\n\nFailed to execute function call: {e}\n\n")
        print(f"Function call details: {function_tool_call}\n\n")
        return ToolMessage(
            status="error",
            tool_call_id=call_id,
            content={"message": f"Error occured while calling function {name}: {e}"},
            additional_kwargs={"type": "function_call_output"},
        )


def handle_computer_call(page: Page, computer_call: dict):
    action = computer_call.get("action")
    call_id = computer_call.get("call_id")
    try:
        action_type = action.get("type")

        if action_type == "click":
            button = action.get("button")
            x = action.get("x")
            y = action.get("y")
            if button == "back":
                page.go_back(timeout=30000)
            elif button == "forward":
                page.go_forward(timeout=30000)
            elif button == "wheel":
                page.mouse.wheel(x, y)
            else:
                button_mapping = {"left": "left", "right": "right", "middle": "left"}
                page.mouse.click(x, y, button=button_mapping.get(button))
        elif action_type == "scroll":
            x = action.get("x")
            y = action.get("y")
            delta_x = action.get("scroll_x")
            delta_y = action.get("scroll_y")
            page.mouse.move(x, y)
            page.evaluate(f"window.scrollBy({delta_x}, {delta_y})")
        elif action_type == "keypress":
            keys = action.get("keys")
            mapped_keys = [_translate_key(key) for key in keys]
            for key in mapped_keys:
                page.keyboard.down(key)
            for key in reversed(mapped_keys):
                page.keyboard.up(key)
        elif action_type == "type":
            text = action.get("text")
            page.keyboard.type(text)
        elif action_type == "wait":
            time.sleep(2)
        elif action_type == "screenshot":
            pass
        elif action_type == "double_click":
            x = action.get("x")
            y = action.get("y")
            page.mouse.click(x, y, button="left", click_count=2)
        elif action_type == "drag":
            path = action.get("path")
            page.mouse.move(path[0].get("x"), path[0].get("y"))
            page.mouse.down()
            for point in path[1:]:
                page.mouse.move(point.get("x"), point.get("y"))
                time.sleep(40 + floor(random() * 40))
            page.mouse.up()
        elif action_type == "move":
            x = action.get("x")
            y = action.get("y")
            page.mouse.move(x, y)
        else:
            raise ValueError(f"Unknown action type received: {action_type}")

        time.sleep(1)
        screenshot = page.screenshot()
        b64_screenshot = base64.b64encode(screenshot).decode("utf-8")
        screenshot_url = f"data:image/png;base64,{b64_screenshot}"
        output_content = {
            "type": "input_image",
            "image_url": screenshot_url,
        }
        return ToolMessage(
            tool_call_id=call_id,
            content=[output_content],
            additional_kwargs={"type": "computer_call_output"},
        )

    except Exception as e:
        print(f"\n\nFailed to execute computer call: {e}\n\n")
        print(f"Computer call details: {computer_call}\n\n")
        return ToolMessage(
            tool_call_id=call_id,
            status="error",
            content=[{"type": "input_image", "image_url": DUMMY_SCREENSHOT}],
            additional_kwargs={"type": "computer_call_output", "status": "incomplete"},
        )


def take_browser_action(state: CUAState, config: RunnableConfig) -> Dict[str, Any]:
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

    instance_id = state.get("instance_id")
    if not instance_id:
        raise ValueError("Instance ID not found in state.")
    instance: SessionDetail = get_instance(instance_id, config)

    if instance.status != "active":
        raise ValueError("Instance is not active.")

    browser_state = state.get("browser_state")
    if not browser_state:
        raise ValueError("Browser state not found in state.")
    browser = browser_state.get("browser")
    if not browser:
        raise ValueError("Browser not found in browser state.")
    current_context = browser.contexts[0]
    page = browser_state.get("current_page", current_context.pages[0])

    def handle_page_event(newPage: Page):
        nonlocal page
        page = newPage
        browser_state["current_page"] = newPage

    stream_url: Optional[str] = state.get("stream_url")
    if not stream_url:
        # If the stream_url is not yet defined in state, fetch it, then write to the custom stream
        # so that it's made accessible to the client (or whatever is reading the stream) before any actions are taken.
        stream_url = instance.live_url

        writer = get_stream_writer()
        writer({"stream_url": stream_url})

    current_context.on("page", handle_page_event)

    output = tool_outputs[-1] if len(tool_outputs) > 0 else None
    tool_message: Optional[ToolMessage] = None

    for tool_call in tool_calls:
        tool_message = handle_function_tool_call(page, tool_call)
        time.sleep(1)

    if output:
        if output.get("type") == "computer_call":
            tool_message = handle_computer_call(page, output)
        else:
            print("unknown tool output type", output)

    return {
        "messages": tool_message if tool_message else None,
        "instance_id": instance.id,
        "stream_url": stream_url,
        "browser_state": browser_state,
    }
