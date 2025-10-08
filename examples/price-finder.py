"""
Price Finder: A LangGraph workflow that routes user requests to either a general response
or a Computer Use Agent (CUA) for web-based price lookups.
"""

from typing import List, Literal

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage
# from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from langgraph_cua import create_cua
from langgraph_cua.types import CUAState

# Load environment variables from .env file
load_dotenv()

# Create the Computer Use Agent (CUA) graph
cua_graph = create_cua()


class PriceFinderState(CUAState):
    """State class for the Price Finder workflow, extending the CUA state."""

    route: Literal["respond", "computer_use_agent"]


def process_input(state: PriceFinderState):
    """
    Analyzes the user's latest message and determines whether to route to the
    computer use agent or to generate a direct response.

    Args:
        state: Current workflow state containing message history

    Returns:
        Dict with routing decision
    """
    system_message = {
        "role": "system",
        "content": (
            "You're an advanced AI assistant tasked with routing the user's query to the appropriate node."
            + "Your options are: computer use or respond. You should pick computer use if the user's request requires "
            + "using a computer (e.g. looking up a price on a website), and pick respond for ANY other inputs."
        ),
    }

    class RoutingToolSchema(BaseModel):
        """Route the user's request to the appropriate node."""

        route: Literal["respond", "computer_use_agent"] = Field(
            ...,
            description="The node to route to, either 'computer_use_agent' for any input which might require using a computer to assist the user, or 'respond' for any other input",
        )

    model = ChatAnthropic(model="claude-sonnet-4-5",
                            temperature=0)
    model_with_tools = model.with_structured_output(RoutingToolSchema)

    messages = [system_message, {"role": "user", "content": state.get("messages")[-1].content}]

    response = model_with_tools.invoke(messages)
    return {"route": response.route}


def respond(state: PriceFinderState):
    """
    Generates a general response to the user based on the entire conversation history.

    Args:
        state: Current workflow state containing full message history

    Returns:
        Dict containing the generated response
    """

    def format_messages(messages: List[AnyMessage]) -> str:
        """Formats a list of messages into a single string with type and content."""
        return "\n".join([f"{message.type}: {message.content}" for message in messages])

    system_message = {
        "role": "system",
        "content": (
            "You're an advanced AI assistant tasked with responding to the user's input."
            + "You're provided with the full conversation between the user, and the AI assistant. "
            + "This conversation may include messages from a computer use agent, along with "
            + "general user inputs and AI responses. \n\n"
            + "Given all of this, please RESPOND to the user. If there is nothing to respond to, you may return something like 'Let me know if you have any other questions.'"
        ),
    }
    human_message = {
        "role": "user",
        "content": "Here are all of the messages in the conversation:\n\n"
        + format_messages(state.get("messages")),
    }

    model = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
    response = model.invoke([system_message, human_message])
    return {"response": response}


def route_after_processing_input(state: PriceFinderState):
    """
    Conditional router that returns the route determined by process_input.

    Args:
        state: Current workflow state with route decision

    Returns:
        String route name for the next node
    """
    return state.get("route")


workflow = StateGraph(PriceFinderState)
workflow.add_node("process_input", process_input)
workflow.add_node("respond", respond)
workflow.add_node("computer_use_agent", cua_graph)

workflow.add_edge(START, "process_input")
workflow.add_conditional_edges("process_input", route_after_processing_input)
workflow.add_edge("respond", END)
workflow.add_edge("computer_use_agent", END)

graph = workflow.compile()
graph.name = "Price Finder"


async def main():
    """Run the Price Finder workflow with a sample tire price query."""
    # Define the initial conversation messages
    messages = [
        {
            "role": "system",
            "content": (
                "You're an advanced AI computer use assistant. The browser you are using "
                "is already initialized, and visiting google.com."
            ),
        },
        {
            "role": "user",
            "content": (
                "Can you go to athenaintel.com and tell me what jobs openings are available? Do not ask me any questions or preferences, just use your best judgment. And if you encounter any captcha, just go straight to the site and don't use google.com. Basically do not bother me with any questions, just use your best judgment."
            ),
        },
    ]

    # Stream the graph execution with updates visible
    stream = graph.astream({"messages": messages}, subgraphs=True, stream_mode="updates", 
        config={
                "configurable": {"zdr_enabled": True},
                "recursion_limit": 1000,
            })
    print("Stream started")

    # Process and display the stream updates
    async for update in stream:
        print(f"\n----\nUPDATE: {update}\n----\n")

    print("Done")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
