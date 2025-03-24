from langgraph.graph import END, START, StateGraph

from langgraph_cua.nodes import call_model, take_computer_action
from langgraph_cua.types import CUAState
from langgraph_cua.utils import is_computer_tool_call


def take_action_or_end(state: CUAState):
    """
    Routes to the take_computer_action node if a computer call is present
    in the last message, otherwise routes to END.

    Args:
        state: The current state of the thread.

    Returns:
        "take_computer_action" or END depending on if a computer call is present.
    """
    if not state.get("messages", []):
        return END

    last_message = state.get("messages", [])[-1]
    additional_kwargs = getattr(last_message, "additional_kwargs", None)

    if not additional_kwargs:
        return END

    tool_outputs = additional_kwargs.get("tool_outputs")

    if not is_computer_tool_call(tool_outputs):
        return END

    return "take_computer_action"


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


workflow = StateGraph(CUAState)

workflow.add_node("call_model", call_model)
workflow.add_node("take_computer_action", take_computer_action)

workflow.add_edge(START, "call_model")
workflow.add_conditional_edges("call_model", take_action_or_end)
workflow.add_conditional_edges("take_computer_action", reinvoke_model_or_end)

graph = workflow.compile()
graph.name = "Computer Use Agent"


def create_cua(
    *,
    scrapybara_api_key: str = None,
    timeout_hours: float = 1.0,
    zdr_enabled: bool = False,
    recursion_limit: int = 100,
):
    """Configuration for the Computer Use Agent.

    Attributes:
        scrapybara_api_key: The API key to use for Scrapybara.
            This can be provided in the configuration, or set as an environment variable (SCRAPYBARA_API_KEY).
        timeout_hours: The number of hours to keep the virtual machine running before it times out.
            Must be between 0.01 and 24. Default is 1.
        zdr_enabled: Whether or not Zero Data Retention is enabled in the user's OpenAI account. If True,
            the agent will not pass the 'previous_response_id' to the model, and will always pass it the full
            message history for each request. If False, the agent will pass the 'previous_response_id' to the
            model, and only the latest message in the history will be passed. Default False.
        recursion_limit: The maximum number of recursive calls the agent can make. Default is 100.
    """
    # Validate timeout_hours is within acceptable range
    if timeout_hours < 0.01 or timeout_hours > 24:
        raise ValueError("timeout_hours must be between 0.01 and 24")

    # Configure the graph with the provided parameters
    configured_graph = graph.with_config(
        config={
            "configurable": {
                "scrapybara_api_key": scrapybara_api_key,
                "timeout_hours": timeout_hours,
                "zdr_enabled": zdr_enabled,
            },
            "recursion_limit": recursion_limit,
        }
    )

    return configured_graph


__all__ = ["create_cua", graph]
