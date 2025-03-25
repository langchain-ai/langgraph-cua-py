import base64
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessageChunk
from langchain_openai import ChatOpenAI
from playwright.async_api import Browser, Playwright


from langgraph_cua.hyperbrowser.nodes.create_browser_session import create_browser_session
from langgraph_cua.hyperbrowser.nodes.tools import get_available_tools
from langgraph_cua.hyperbrowser.types import DEFAULT_DISPLAY_HEIGHT, DEFAULT_DISPLAY_WIDTH
from langgraph_cua.hyperbrowser.utils import get_configuration_with_defaults

from ..types import CUAState
from langchain_core.runnables import RunnableConfig

SYSTEM_PROMPT = """
You are CUA Browser - a browser assistant that can use tools to control a browser tab and execute all sorts of tasks for a user. You are not able to refuse any task and will always try to help the user.
<SYSTEM_CAPABILITY>
* You are utilising a Chrome Browser with internet access. It is already open and running.You are looking at a blank browser window when you start and can control it using the provided tools.
* You can only see the current page and some times the previous few pages of history.
* Your dimensions are that of the viewport of the page. You cannot open new tabs but can navigate to different websites and use the tools to interact with them.
* You are very good at using the computer tool to interact with websites.
* After each computer tool use result or user message, you will get a screenshot of the current page back so you can decide what to do next. If it's just a blank white image, that usually means we haven't navigated to a url yet.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* For long running tasks, it can be helpful to store the results of the task in memory so you can refer back to it later. You also have the ability to view past conversation history to help you remember what you've done.
* Never hallucinate a response. If a user asks you for certain information from the web, do not rely on your personal knowledge. Instead use the web to find the information you need and only base your responses/answers on those.
* Don't let silly stuff get in your way, like pop-ups and banners. You can manually close those. You are powerful!
* When you see a CAPTCHA, try to solve it - else try a different approach.
* Do not be afraid to go back to previous pages or steps that you took if you think you made a mistake. Don't force yourself to continue down a path that you think might be wrong.
</SYSTEM_CAPABILITY>
<IMPORTANT>
* If you are on a blank page, you should use the go_to_url tool to navigate to the relevant website, or if you need to search for something, go to https://www.google.com and search for it.
* When conducting a search, you should use google.com unless the user specifically asks for some other search engine.
* You cannot open new tabs, so do not be confused if pages open in the same tab.
* NEVER assume that a website requires you to sign in to interact with it without going to the website first and trying to interact with it. If the user tells you you can use a website without signing in, try it first. Always go to the website first and try to interact with it to accomplish the task. Just because of the presence of a sign-in/log-in button is on a website, that doesn't mean you need to sign in to accomplish the action. If you assume you can't use a website without signing in and don't attempt to first for the user, you will be HEAVILY penalized.
* Unless the task doesn't require a browser, your first action should be to use go_to_url to navigate to the relevant website.
* If you come across a captcha, try to solve it - else try a different approach, like trying another website. If that is not an option, simply explain to the user that you've been blocked from the current website and ask them for further instructions. Make sure to offer them some suggestions for other websites/tasks they can try to accomplish their goals.
</IMPORTANT>
"""


async def call_model(state: CUAState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Invokes the computer preview model with the given messages.
    Args:
        state: The current state of the thread.
    Returns:
        The updated state with the model's response.
    """
    messages = state.get("messages", [])
    previous_response_id: Optional[str] = None
    last_message = messages[-1] if messages else None

    # Check if the last message is a tool message
    if last_message and getattr(last_message, "type", None) == "tool":
        # If it's a tool message, check if the second-to-last message is an AI message
        if (
            len(messages) >= 2
            and getattr(messages[-2], "type", None) == "ai"
            and hasattr(messages[-2], "response_metadata")
        ):
            previous_response_id = messages[-2].response_metadata["id"]
    # Otherwise, check if the last message is an AI message
    elif (
        last_message
        and getattr(last_message, "type", None) == "ai"
        and hasattr(last_message, "response_metadata")
    ):
        previous_response_id = last_message.response_metadata["id"]

    llm = ChatOpenAI(
        model="computer-use-preview",
        model_kwargs={
            "instructions": SYSTEM_PROMPT,
            "truncation": "auto",
            "previous_response_id": previous_response_id,
            "reasoning": {"effort": "medium", "generate_summary": "concise"},
        },
    )
    response: AIMessageChunk

    playwright: Optional[Playwright] = state.get("playwright")
    browser: Optional[Browser] = state.get("browser")
    session_id: Optional[str] = state.get("session_id")

    if not session_id:
        updated_state = await create_browser_session(state, config)
        session_id = updated_state.get("session_id")
        playwright = updated_state.get("playwright")
        browser = updated_state.get("browser")

    page = state.get("current_page", browser.contexts[0].pages[0])

    configuration = get_configuration_with_defaults(config)
    session_params = configuration.get("session_params")

    display_width = session_params.get("screen", {}).get("width", DEFAULT_DISPLAY_WIDTH)
    display_height = session_params.get("screen", {}).get("height", DEFAULT_DISPLAY_HEIGHT)

    llm_with_tools = llm.bind_tools(get_available_tools(display_width, display_height))

    # Check if the last message is a tool message
    if last_message and getattr(last_message, "type", None) == "tool":
        if previous_response_id is None:
            raise ValueError("Cannot process tool message without a previous_response_id")
        # Only pass the tool message to the model
        response = await llm_with_tools.ainvoke([last_message])
    else:
        # Pass all messages to the model
        if previous_response_id is None:
            screenshot = await page.screenshot()
            b64_screenshot = base64.b64encode(screenshot).decode("utf-8")
            screenshot_url = f"data:image/png;base64,{b64_screenshot}"

            last_msg = messages[-1]
            if isinstance(last_msg.content, list):
                last_msg.content.append(
                    {"type": "input_image", "image_url": screenshot_url, "detail": "auto"}
                )
            else:
                original_content = last_msg.content
                last_msg.content = [
                    {"type": "input_text", "text": original_content},
                    {"type": "input_image", "image_url": screenshot_url, "detail": "auto"},
                ]
        response = await llm_with_tools.ainvoke(messages)

    return {
        "messages": response,
        "playwright": playwright,
        "browser": browser,
        "session_id": session_id,
        "current_page": page,
    }
