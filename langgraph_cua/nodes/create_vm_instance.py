from typing import Any, Dict
from langchain_core.runnables.config import RunnableConfig
from scrapybara.client import BrowserInstance, UbuntuInstance, WindowsInstance
from hyperbrowser.models import SessionDetail, CreateSessionParams
from playwright.sync_api import sync_playwright
from ..types import CUAState, Provider
from ..utils import get_configuration_with_defaults, get_hyperbrowser_client, get_scrapybara_client

# Copied from the OpenAI example repository
# https://github.com/openai/openai-cua-sample-app/blob/eb2d58ba77ffd3206d3346d6357093647d29d99c/utils.py#L13
BLOCKED_DOMAINS = [
    "maliciousbook.com",
    "evilvideos.com",
    "darkwebforum.com",
    "shadytok.com",
    "suspiciouspins.com",
    "ilanbigio.com",
]


def create_scrapybara_instance(configuration: Dict[str, Any]):
    scrapybara_api_key = configuration.get("scrapybara_api_key")
    timeout_hours = configuration.get("timeout_hours")
    environment = configuration.get("environment")

    if not scrapybara_api_key:
        raise ValueError(
            "Scrapybara API key not provided. Please provide one in the configurable fields, "
            "or set it as an environment variable (SCRAPYBARA_API_KEY)"
        )

    client = get_scrapybara_client(scrapybara_api_key)

    instance: UbuntuInstance | BrowserInstance | WindowsInstance

    if environment == "ubuntu":
        instance = client.start_ubuntu(timeout_hours=timeout_hours)
    elif environment == "windows":
        instance = client.start_windows(timeout_hours=timeout_hours)
    elif environment == "web":
        blocked_domains = [
            domain.replace("https://", "").replace("www.", "") for domain in BLOCKED_DOMAINS
        ]
        instance = client.start_browser(
            timeout_hours=timeout_hours, blocked_domains=blocked_domains
        )
    else:
        raise ValueError(
            f"Invalid environment. Must be one of 'web', 'ubuntu', or 'windows'. Received: {environment}"
        )

    stream_url = instance.get_stream_url().stream_url

    return {
        "instance_id": instance.id,
        "stream_url": stream_url,
    }


def create_hyperbrowser_instance(state: CUAState, configuration: Dict[str, Any]):
    hyperbrowser_api_key = configuration.get("hyperbrowser_api_key")
    session_params = configuration.get("session_params", {})
    browser_state = state.get("browser_state")

    if not hyperbrowser_api_key:
        raise ValueError(
            "Hyperbrowser API key not provided. Please provide one in the configurable fields, "
            "or set it as an environment variable (HYPERBROWSER_API_KEY)"
        )

    client = get_hyperbrowser_client(hyperbrowser_api_key)
    session: SessionDetail = client.sessions.create(params=CreateSessionParams(**session_params))

    if not browser_state:
        p = sync_playwright().start()
        browser = p.chromium.connect_over_cdp(f"{session.ws_endpoint}&keepAlive=true")
        curr_page = browser.contexts[0].pages[0]
        if curr_page.url == "about:blank":
            curr_page.goto("https://www.google.com", timeout=15000, wait_until="domcontentloaded")
        browser_state = {
            "browser": browser,
            "current_page": curr_page,
        }

    return {
        "instance_id": session.id,
        "stream_url": session.live_url,
        "browser_state": browser_state,
    }


def create_vm_instance(state: CUAState, config: RunnableConfig):
    instance_id = state.get("instance_id")

    if instance_id is not None:
        # If the instance_id already exists in state, do nothing.
        return {}

    configuration = get_configuration_with_defaults(config)
    provider = configuration.get("provider")

    if provider == Provider.Scrapybara:
        return create_scrapybara_instance(configuration)
    elif provider == Provider.Hyperbrowser:
        return create_hyperbrowser_instance(state, configuration)
    else:
        raise ValueError(
            f"Invalid provider. Must be one of 'scrapybara' or 'hyperbrowser'. Received: {provider}"
        )
