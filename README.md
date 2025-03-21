# 🤖 LangGraph Computer Use Agent (CUA)

> [!WARNING]
> **THIS REPO IS A WORK IN PROGRESS AND NOT INTENDED FOR USE YET**

A Python library for creating computer use agent (CUA) systems using [LangGraph](https://github.com/langchain-ai/langgraph). A CUA is a type of agent which has the ability to interact with a computer to preform tasks.

## Features

- **ADD FEATURES HERE**

This library is built on top of [LangGraph](https://github.com/langchain-ai/langgraph), a powerful framework for building agent applications, and comes with out-of-box support for [streaming](https://langchain-ai.github.io/langgraph/how-tos/#streaming), [short-term and long-term memory](https://langchain-ai.github.io/langgraph/concepts/memory/) and [human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/).

## Installation

```bash
pip install langgraph-cua langgraph langchain-core langchain-openai
```

## Quickstart

```bash
export SCRAPYBARA_API_KEY=<your_api_key>
```

```py
# TODO: Add examples
```

## How to add memory

TODO: Add how to add memory section

## How to customize

TODO: Add how to customize section

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

Finally, you can then run the tests:

```bash
pytest -xvs tests/test_cua.py
```
