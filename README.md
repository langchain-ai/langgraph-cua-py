# 🤖 LangGraph Computer Use Agent (CUA)

> [!TIP]
> Looking for the TypeScript version? [Check out the repo here](https://github.com/langchain-ai/langgraphjs/tree/main/libs/langgraph-cua).

A Python library for creating computer use agent (CUA) systems using [LangGraph](https://github.com/langchain-ai/langgraph). A CUA is a type of agent which has the ability to interact with a computer to preform tasks.

Short demo video:
<video src="https://github.com/user-attachments/assets/7fd0ab05-fecc-46f5-961b-6624cb254ac2" controls></video>

> [!TIP]
> This demo used the following prompt:
> ```
> I want to contribute to the LangGraph.js project. Please find the GitHub repository, and inspect the read me,
> along with some of the issues and open pull requests. Then, report back with a plan of action to contribute.
> ```

This library is built on top of [LangGraph](https://github.com/langchain-ai/langgraph), a powerful framework for building agent applications, and comes with out-of-box support for [streaming](https://langchain-ai.github.io/langgraph/how-tos/#streaming), [short-term and long-term memory](https://langchain-ai.github.io/langgraph/concepts/memory/) and [human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/).

## Installation

```bash
pip install langgraph-cua
```

## Quickstart

This project supports two providers for accessing virtual machines:
1. [Scrapybara](https://scrapybara.com/) (default)
2. [Hyperbrowser](https://hyperbrowser.ai/)

### Using Scrapybara (Default)

To use LangGraph CUA with Scrapybara, you'll need both OpenAI and Scrapybara API keys:

```bash
export OPENAI_API_KEY=<your_api_key>
export SCRAPYBARA_API_KEY=<your_api_key>
```

Then, create the graph by importing the `create_cua` function from the `langgraph_cua` module.

```python
from langgraph_cua import create_cua
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create CUA with Scrapybara (default provider)
cua_graph = create_cua()

# Define the input messages
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
            "Can you find the best price for new all season tires which will fit on my 2019 Subaru Forester?"
        ),
    },
]

async def main():
    # Stream the graph execution
    stream = cua_graph.astream(
        {"messages": messages},
        stream_mode="updates"
    )

    # Process the stream updates
    async for update in stream:
        if "create_vm_instance" in update:
            print("VM instance created")
            stream_url = update.get("create_vm_instance", {}).get("stream_url")
            # Open this URL in your browser to view the CUA stream
            print(f"Stream URL: {stream_url}")

    print("Done")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Using Hyperbrowser

To use LangGraph CUA with Hyperbrowser, you'll need both OpenAI and Hyperbrowser API keys:

```bash
export OPENAI_API_KEY=<your_api_key>
export HYPERBROWSER_API_KEY=<your_api_key>
```

Then, create the graph specifying Hyperbrowser as the provider:

```python
from langgraph_cua import create_cua
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create CUA with Hyperbrowser provider
cua_graph = create_cua(provider="hyperbrowser")

# Define the input messages
messages = [
    {
        "role": "system",
        "content": (
            "You're an advanced AI computer use assistant. You are utilizing a Chrome Browser with internet access. "
            "It is already open and running. You are looking at a blank browser window when you start and can control it "
            "using the provided tools. If you are on a blank page, you should use the go_to_url tool to navigate to "
            "the relevant website, or if you need to search for something, go to https://www.google.com and search for it."
        ),
    },
    {
        "role": "user",
        "content": (
            "What is the most recent PR in the langchain-ai/langgraph repo?"
        ),
    },
]

async def main():
    # Stream the graph execution
    stream = cua_graph.astream(
        {"messages": messages},
        stream_mode="updates"
    )

    # Process the stream updates
    async for update in stream:
        if "create_vm_instance" in update:
            print("VM instance created")
            stream_url = update.get("create_vm_instance", {}).get("stream_url")
            # Open this URL in your browser to view the CUA stream
            print(f"Stream URL: {stream_url}")

    print("Done")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

The above example will invoke the graph, passing in a request for it to do some research into LangGraph.js from 
the standpoint of a new contributor. The code will log the stream URL, which you can open in your browser to 
view the CUA stream.

You can find more examples inside the [`examples` directory](./examples/).

## How to customize

The `create_cua` function accepts a few configuration parameters. These are the same configuration parameters that the graph accepts, along with `recursion_limit`.

You can either pass these parameters when calling `create_cua`, or at runtime when invoking the graph by passing them to the `config` object.

### Configuration Parameters

#### Common Parameters
- `provider`: The provider to use. Default is `"scrapybara"`. Options are `"scrapybara"` and `"hyperbrowser"`.
- `zdr_enabled`: Whether or not Zero Data Retention is enabled in the user's OpenAI account. If `True`, the agent will not pass the `previous_response_id` to the model, and will always pass it the full message history for each request. If `False`, the agent will pass the `previous_response_id` to the model, and only the latest message in the history will be passed. Default `False`.
- `recursion_limit`: The maximum number of recursive calls the agent can make. Default is 100. This is greater than the standard default of 25 in LangGraph, because computer use agents are expected to take more iterations.
- `prompt`: The prompt to pass to the model. This will be passed as the system message.

#### Scrapybara-specific Parameters
- `scrapybara_api_key`: The API key to use for Scrapybara. If not provided, it defaults to reading the `SCRAPYBARA_API_KEY` environment variable.
- `timeout_hours`: The number of hours to keep the virtual machine running before it times out.
- `auth_state_id`: The ID of the authentication state. If defined, it will be used to authenticate with Scrapybara. Only applies if 'environment' is set to 'web'.
- `environment`: The environment to use. Default is `web`. Options are `web`, `ubuntu`, and `windows`.

#### Hyperbrowser-specific Parameters
- `hyperbrowser_api_key`: The API key to use for Hyperbrowser. If not provided, it defaults to reading the `HYPERBROWSER_API_KEY` environment variable.
- `session_params`: Parameters to use for configuring the Hyperbrowser session, such as screen dimensions, proxy usage, etc. For more information on the available parameters, see the [Hyperbrowser API documentation](https://docs.hyperbrowser.ai/sessions/overview/session-parameters). Note that the parameters will be snake_case for usage with the Hyperbrowser Python SDK.

### System Prompts

Including a system prompt with your CUA graph is recommended, and can save the agent time in its initial steps by providing context into its environment and objective. Below is the recommended system prompt from Scrapybara:

<details><summary>System Prompt</summary>
    
    You have access to an Ubuntu VM with internet connectivity. You can install Ubuntu applications using the bash tool (prefer curl over wget).  

    ### Handling HTML and Large Text Output  
    - To read an HTML file, open it in Chromium using the address bar.  

    ### Interacting with Web Pages and Forms  
    - Zoom out or scroll to ensure all content is visible.  
    - When interacting with input fields:  
    - Clear the field first using `Ctrl+A` and `Delete`.  
    - Take an extra screenshot after pressing "Enter" to confirm the input was submitted correctly.  
    - Move the mouse to the next field after submission.  

    ### Efficiency and Authentication  
    - Computer function calls take time; optimize by stringing together related actions when possible.  
    - You are allowed to take actions on authenticated sites on behalf of the user.  
    - Assume the user has already authenticated if they request access to a site.  
    - For logging into additional sites, ask the user to use Auth Contexts or the Interactive Desktop.  

    ### Handling Black Screens  
    - If the first screenshot shows a black screen:  
    - Click the center of the screen.  
    - Take another screenshot.  

    ### Best Practices  
    - If given a complex task, break it down into smaller steps and ask for details only when necessary.  
    - Read web pages thoroughly by scrolling down until sufficient information is gathered.  
    - Explain each action you take and why.  
    - Avoid asking for confirmation on routine actions (e.g., pressing "Enter" after typing a URL). Seek clarification only for ambiguous or critical actions (e.g., deleting files or submitting sensitive information).  
    - If a user's request implies the need for external information, assume they want you to search for it and provide the answer directly.  

    ### Date Context  
    Today's date is {todays_date}


If you choose to use this prompt, ensure you're populating the `{todays_date}` placeholder with the current date.

</details>

## Auth States

LangGraph CUA integrates with Scrapybara's [auth states API](https://docs.scrapybara.com/auth-states) to persist browser authentication sessions. This allows you to authenticate once (e.g., logging into Amazon) and reuse that session in future runs.

### Using Auth States

Pass an `auth_state_id` when creating your CUA graph:

```python
from langgraph_cua import create_cua

cua_graph = create_cua(auth_state_id="<your_auth_state_id>")
```

The graph stores this ID in the `authenticated_id` state field. If you change the `auth_state_id` in future runs, the graph will automatically reauthenticate.

### Managing Auth States with Scrapybara SDK

#### Save an Auth State

```python
from scrapybara import Scrapybara

client = Scrapybara(api_key="<api_key>")
instance = client.get("<instance_id>")
auth_state_id = instance.save_auth(name="example_site").auth_state_id
```

#### Modify an Auth State

```python
client = Scrapybara(api_key="<api_key>")
instance = client.get("<instance_id>")
instance.modify_auth(auth_state_id="your_existing_auth_state_id", name="renamed_auth_state")
```

> [!NOTE]
> To apply changes to an auth state in an existing run, set the `authenticated_id` state field to `None` to trigger re-authentication.


## Zero Data Retention (ZDR)

LangGraph CUA supports Zero Data Retention (ZDR) via the `zdr_enabled` configuration parameter. When set to true, the graph will _not_ assume it can use the `previous_message_id`, and _all_ AI & tool messages will be passed to the OpenAI on each request.

## Development

To get started with development, first clone the repository:

```bash
git clone https://github.com/langchain-ai/langgraph-cua.git
```

Create a virtual environment:

```bash
uv venv
```

Activate it:

```bash
source .venv/bin/activate
```

Then, install dependencies:

```bash
uv sync --all-groups
```

Next, set the required environment variables:

```bash
cp .env.example .env
```

Finally, you can then run the integration tests:

```bash
pytest -xvs tests/integration/test_cua.py
```
