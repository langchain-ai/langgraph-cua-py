import ast
import json

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph_cua import create_cua

# Load environment variables from .env file
load_dotenv()


@pytest.mark.asyncio
async def test_browser_interaction():
    """
    Test that the agent can interact with the browser.
    This is a port of the TypeScript test to Python.
    """
    graph = create_cua()

    # Create input messages similar to the TypeScript test
    messages = [
        SystemMessage(
            content=(
                "You're an advanced AI computer use assistant. The browser you are using "
                "is already initialized, and visiting google.com."
            )
        ),
        HumanMessage(
            content=(
                "I'm looking for a new camera. Help me find the best one. It should be 4k resolution, "
                "by Cannon, and under $1000. I want a digital camera, and I'll be using it mainly for photography."
            )
        ),
    ]

    # Enable/disable different handling of messages based on whether or not ZDR is enabled
    zdr_enabled = True

    # Stream the graph execution
    stream = graph.astream(
        {"messages": messages},
        {"streamMode": ["custom", "updates"], "configurable": {"zdr_enabled": zdr_enabled}},
    )

    # Process the stream updates
    async for update in stream:
        print("\n---UPDATE---\n")

        if "take_computer_action" in update:
            print("Computer Action:")
            # Check for tool message in the messages field
            tool_message = update.get("take_computer_action", {}).get("messages")
            if tool_message:
                # Extract content from the tool message
                content = tool_message.content

                # Try to parse content if it's a string
                parsed_content = None
                if isinstance(content, str):
                    try:
                        # Try parsing as JSON first
                        parsed_content = json.loads(content)
                    except json.JSONDecodeError:
                        try:
                            # Try parsing as Python literal (for string representations of dicts)
                            parsed_content = ast.literal_eval(content)
                        except (SyntaxError, ValueError):
                            # If both fail, keep content as is
                            parsed_content = None
                else:
                    # If content is already a dict, use it directly
                    parsed_content = content if isinstance(content, dict) else None

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
                    if isinstance(content, str) and len(content) > 200:
                        print(f"Tool Message (truncated content): {content[:200]}...")
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
