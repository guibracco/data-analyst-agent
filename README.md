# DataAnalystAgent

A lightweight orchestration layer that chains two existing LangGraph projects into a single pipeline:

1. **`data-cleaning-agent`**: LLM-driven data cleaning
2. **`eda-workflow`**: automated first-pass exploratory data analysis

Flow: **raw CSV → clean data → EDA report**

## Why this project exists

`DataAnalystAgent` demonstrates agent-to-agent orchestration without rewriting either sub-project. The parent graph handles only state passing and conditional routing (e.g. skipping EDA when cleaning fails).

## Setup

### Prerequisites
- Python 3.10 or 3.11
- Poetry
- OpenAI API key

### Install
From this folder:

```bash
poetry install
```

Copy the example environment file and fill in your key:

```bash
cp .env.example .env
```

Then edit `.env` and set your OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-key-here
```

## Run example

```bash
poetry run python example_usage.py
```

## Project structure

```text
data-analyst-agent/
├── data_analyst_agent/
│   ├── __init__.py
│   └── orchestrator.py
├── .env.example
├── example_usage.py
├── pyproject.toml
└── README.md
```

## Notes
- Uses `data_cleaning_agent` and `EDAWorkflow` internals as-is.
- If cleaning fails, EDA is skipped.
- Keep this project thin; add complexity only when needed (for example, richer routing or observability).
