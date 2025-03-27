import ast
import json

import pytest
from dotenv import load_dotenv

from langgraph_cua import create_cua

# Load environment variables from .env file
load_dotenv()


async def process_stream(stream):
    async for update in stream:
        print("\n---UPDATE---\n")

        if "create_vm_instance" in update:
            print("VM instance created")
            stream_url = update.get("create_vm_instance", {}).get("stream_url")
            # Open this URL in your browser to view the CUA stream
            print(f"Stream URL: {stream_url}")
        elif "take_computer_action" in update:
            print("Computer Action:")
            # Check for tool message in the messages field
            tool_message = update.get("take_computer_action", {}).get("messages")
            if tool_message:
                # Extract content from the tool message
                content = tool_message.content

                # Handle the case where content is an array
                if isinstance(content, list) and len(content) > 0:
                    # Use the first item in the array
                    content_item = content[0]
                else:
                    # Use content directly if it's not an array
                    content_item = content

                # Try to parse content if it's a string
                parsed_content = None
                if isinstance(content_item, str):
                    try:
                        # Try parsing as JSON first
                        parsed_content = json.loads(content_item)
                    except json.JSONDecodeError:
                        try:
                            # Try parsing as Python literal (for string representations of dicts)
                            parsed_content = ast.literal_eval(content_item)
                        except (SyntaxError, ValueError):
                            # If both fail, keep content as is
                            parsed_content = None
                else:
                    # If content is already a dict, use it directly
                    parsed_content = content_item if isinstance(content_item, dict) else None

                # Handle image_url specially - truncate to 100 chars
                if (
                    parsed_content
                    and isinstance(parsed_content, dict)
                    and parsed_content.get("image_url")
                ):
                    image_url = parsed_content["image_url"]
                    # Create a copy to avoid modifying the original
                    content_copy = parsed_content.copy()
                    content_copy["image_url"] = (
                        image_url[:100] + "..." if len(image_url) > 100 else image_url
                    )
                    print(f"Tool Message ID: {tool_message.tool_call_id}")
                    # Print the truncated content explicitly
                    print(f"Content type: {content_copy.get('type')}")
                    print(f"Image URL (truncated): {content_copy['image_url']}")
                else:
                    # Just print the first 200 characters of the content if we couldn't parse it
                    if isinstance(content_item, str) and len(content_item) > 200:
                        print(f"Tool Message (truncated content): {content_item[:200]}...")
                    else:
                        print(f"Tool Message: {tool_message}")
        elif "call_model" in update:
            print("Model Call:")
            if update.get("call_model", {}).get("messages"):
                messages = update["call_model"]["messages"]
                if "tool_outputs" in messages.additional_kwargs:
                    print(messages.additional_kwargs["tool_outputs"])
                else:
                    print(messages.content)
        else:
            print(update)


@pytest.mark.asyncio
async def test_browser_interaction():
    """
    Test that the agent can interact with the browser.
    This is a port of the TypeScript test to Python.
    """
    graph = create_cua()

    # Create input messages similar to the TypeScript test
    messages = [
        {
            "role": "system",
            "content": (
                "You're an advanced AI computer use assistant. The browser you are using "
                "is already initialized, and visiting google.com."
            ),
        },
        # {
        #     "role": "user",
        #     "content": (
        #         "I'm looking for a new camera. Help me find the best one. It should be 4k resolution, "
        #         "by Cannon, and under $1000. I want a digital camera, and I'll be using it mainly for photography."
        #     )
        # },
        {
            "role": "user",
            "content": (
                "I want to contribute to the LangGraph.js project. Please find the GitHub repository, and inspect the read me, "
                "along with some of the issues and open pull requests. Then, report back with a plan of action to contribute."
            ),
        },
    ]

    # Enable/disable different handling of messages based on whether or not ZDR is enabled
    zdr_enabled = True

    # Stream the graph execution
    stream = graph.astream(
        {"messages": messages},
        stream_mode="updates",
        config={
            "configurable": {"zdr_enabled": zdr_enabled},
            "recursion_limit": 100,
        },
    )

    await process_stream(stream)


@pytest.mark.asyncio
async def test_browser_interaction_hyperbrowser():
    """
    Test that the agent can interact with the browser using Hyperbrowser.
    """
    graph = create_cua(provider="hyperbrowser")

    # Create input messages similar to the TypeScript test
    messages = [
        {
            "role": "system",
            "content": (
                "You're an advanced AI computer use assistant. You are utilising a Chrome Browser with internet access."
                "It is already open and running. You are looking at a browser window when you start and can control it using the provided tools."
                "If you are on a blank page, you should use the go_to_url tool to navigate to the relevant website, "
                "or if you need to search for something, go to https://www.google.com and search for it."
            ),
        },
        # {
        #     "role": "user",
        #     "content": (
        #         "I'm looking for a new camera. Help me find the best one. It should be 4k resolution, "
        #         "by Cannon, and under $1000. I want a digital camera, and I'll be using it mainly for photography."
        #     )
        # },
        {
            "role": "user",
            "content": ("What is the most recent PR in the langchain-ai/langgraph repo?"),
        },
    ]

    # Enable/disable different handling of messages based on whether or not ZDR is enabled
    zdr_enabled = True

    # Stream the graph execution
    stream = graph.astream(
        {"messages": messages},
        stream_mode="updates",
        config={
            "configurable": {"zdr_enabled": zdr_enabled, "provider": "hyperbrowser"},
            "recursion_limit": 100,
        },
    )

    await process_stream(stream)
