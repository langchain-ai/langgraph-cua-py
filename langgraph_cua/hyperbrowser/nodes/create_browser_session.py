from langchain_core.runnables.config import RunnableConfig
from hyperbrowser.models import SessionDetail, CreateSessionParams
from langgraph.config import get_stream_writer


from ..types import CUAState
from ..utils import get_configuration_with_defaults, get_hyperbrowser_client, start_playwright


async def create_browser_session(state: CUAState, config: RunnableConfig):
    session_id = state.get("session_id")
    configuration = get_configuration_with_defaults(config)
    hyperbrowser_api_key = configuration.get("hyperbrowser_api_key")
    session_params = configuration.get("session_params")
    stream_url = state.get("stream_url")

    if session_id is not None:
        # If the session_id already exists in state, do nothing.
        return {}

    if not hyperbrowser_api_key:
        raise ValueError(
            "Hyperbrowser API key not provided. Please provide one in the configurable fields, "
            "or set it as an environment variable (HYPERBROWSER_API_KEY)"
        )

    client = get_hyperbrowser_client(hyperbrowser_api_key)

    session: SessionDetail = await client.sessions.create(
        params=CreateSessionParams(**session_params)
    )

    playwright, browser, _ = await start_playwright(state, session)

    if not stream_url:
        stream_url = session.live_url
        writer = get_stream_writer()
        writer({"stream_url": stream_url})

    return {
        "session_id": session.id,
        "stream_url": stream_url,
        "playwright": playwright,
        "browser": browser,
    }
