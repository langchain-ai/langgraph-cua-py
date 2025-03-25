import asyncio
import base64
from math import floor
from random import random
from playwright.async_api import Page
from langchain_core.messages import ToolMessage, ToolCall

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


def get_available_tools(display_width: int = 1024, display_height: int = 800):
    return [
        {
            "type": "computer_use_preview",
            "display_width": display_width,
            "display_height": display_height,
            "environment": "browser",
        },
        {
            "type": "function",
            "function": {
                "name": "go_to_url",
                "description": "Navigate to a URL. The URL must be a valid URL that starts with http or https.",
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


def _translate_key(key: str) -> str:
    return CUA_KEY_TO_PLAYWRIGHT_KEY.get(key.lower(), key)


async def handle_function_tool_call(page: Page, function_tool_call: ToolCall) -> ToolMessage:
    name = function_tool_call.get("name")
    arguments = function_tool_call.get("args")
    call_id = function_tool_call.get("id")

    try:
        if name == "go_to_url":
            await page.goto(arguments.get("url"), timeout=30000, wait_until="domcontentloaded")
            await asyncio.sleep(1)
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


async def handle_computer_call(page: Page, computer_call: dict):
    action = computer_call.get("action")

    try:
        action_type = action.get("type")

        if action_type == "click":
            button = action.get("button")
            x = action.get("x")
            y = action.get("y")
            if button == "back":
                await page.go_back(timeout=30000)
            elif button == "forward":
                await page.go_forward(timeout=30000)
            elif button == "wheel":
                await page.mouse.wheel(x, y)
            else:
                button_mapping = {"left": "left", "right": "right", "middle": "left"}
                await page.mouse.click(x, y, button=button_mapping.get(button))
        elif action_type == "scroll":
            x = action.get("x")
            y = action.get("y")
            delta_x = action.get("scroll_x")
            delta_y = action.get("scroll_y")
            await page.mouse.move(x, y)
            await page.evaluate(f"window.scrollBy({delta_x}, {delta_y})")
        elif action_type == "keypress":
            keys = action.get("keys")
            mapped_keys = [_translate_key(key) for key in keys]
            for key in mapped_keys:
                await page.keyboard.down(key)
            for key in reversed(mapped_keys):
                await page.keyboard.up(key)
        elif action_type == "type":
            text = action.get("text")
            await page.keyboard.type(text)
        elif action_type == "wait":
            await page.wait_for_timeout(2000)
        elif action_type == "screenshot":
            pass
        elif action_type == "double_click":
            x = action.get("x")
            y = action.get("y")
            await page.mouse.click(x, y, button="left", click_count=2)
        elif action_type == "drag":
            path = action.get("path")
            await page.mouse.move(path[0].get("x"), path[0].get("y"))
            await page.mouse.down()
            for point in path[1:]:
                await page.mouse.move(point.get("x"), point.get("y"))
                await page.wait_for_timeout(40 + floor(random() * 40))
            await page.mouse.up()
        elif action_type == "move":
            x = action.get("x")
            y = action.get("y")
            await page.mouse.move(x, y)
        else:
            raise ValueError(f"Unknown action type received: {action_type}")

    except Exception as e:
        print(f"\n\nFailed to execute computer call: {e}\n\n")
        print(f"Computer call details: {computer_call}\n\n")
