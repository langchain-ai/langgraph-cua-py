from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from hyperbrowser.models import CreateSessionParams

from langgraph_cua.hyperbrowser.nodes import call_model, create_browser_session, take_browser_action
from langgraph_cua.hyperbrowser.types import CUAConfiguration, CUAState
from langgraph_cua.hyperbrowser.utils import is_computer_tool_call


def take_action_or_end(state: CUAState):
    """
    Routes to the take_browser_action node if a computer call or function call is present
    in the last message, otherwise routes to END.
    Args:
        state: The current state of the thread.
    Returns:
        "take_browser_action" or END depending on if a computer call or function call is present.
    """
    if not state.get("messages", []):
        return END

    last_message = state.get("messages", [])[-1]
    additional_kwargs = getattr(last_message, "additional_kwargs", None)

    if not additional_kwargs:
        return END

    tool_outputs = additional_kwargs.get("tool_outputs")
    tool_calls = getattr(last_message, "tool_calls", [])

    if not is_computer_tool_call(tool_outputs) and len(tool_calls) == 0:
        return END

    if not state.get("session_id"):
        # If the instance_id is not defined, create a new instance.
        return "create_browser_session"

    return "take_browser_action"


def reinvoke_model_or_end(state: CUAState):
    """
    Routes to the call_model node if the last message is a tool message,
    otherwise routes to END.
    Args:
        state: The current state of the thread.
    Returns:
        "call_model" or END depending on if the last message is a tool message.
    """
    messages = state.get("messages", [])
    if messages and getattr(messages[-1], "type", None) == "tool":
        return "call_model"

    return END


workflow = StateGraph(CUAState, CUAConfiguration)

workflow.add_node("call_model", call_model)
workflow.add_node("create_browser_session", create_browser_session)
workflow.add_node("take_browser_action", take_browser_action)

workflow.add_edge(START, "call_model")
workflow.add_conditional_edges("call_model", take_action_or_end)
workflow.add_edge("create_browser_session", "take_browser_action")
workflow.add_conditional_edges("take_browser_action", reinvoke_model_or_end)

graph = workflow.compile()
graph.name = "Computer Use Agent"


def create_cua(
    *,
    hyperbrowser_api_key: str = None,
    recursion_limit: int = 100,
    session_params: CreateSessionParams = None,
):
    """Configuration for the Computer Use Agent.

    Attributes:
        hyperbrowser_api_key: The API key to use for Hyperbrowser.
            This can be provided in the configuration, or set as an environment variable (HYPERBROWSER_API_KEY).
        recursion_limit: The maximum number of recursive calls the agent can make. Default is 100.
    """

    # Configure the graph with the provided parameters
    configured_graph = graph.with_config(
        config={
            "configurable": {
                "hyperbrowser_api_key": hyperbrowser_api_key,
                "session_params": session_params,
            },
            "recursion_limit": recursion_limit,
        }
    )

    return configured_graph


__all__ = ["create_cua", "graph"]
