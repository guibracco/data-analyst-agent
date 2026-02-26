# Data Analyst Agent

A lightweight orchestration layer that chains two existing LangGraph projects into a single pipeline:

1. **[`data-cleaning-agent`](https://github.com/guibracco/data-cleaning-agent)**: LLM-driven data cleaning
2. **[`eda-workflow`](https://github.com/guibracco/eda-workflow)**: automated first-pass exploratory data analysis

Flow: **raw CSV → PII guardrail → clean data → EDA report**

## Why this project exists

`Data Analyst Agent` demonstrates agent-to-agent orchestration without rewriting either sub-project. The parent graph handles state passing, input guardrails (PII detection), and conditional routing (e.g. blocking the pipeline when PII is found, or skipping EDA when cleaning fails).

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
├── data/
│   ├── cafe_data.csv
│   ├── factory_iot_telemetry.csv
│   └── saas_customer_health.csv
├── data_analyst_agent/
│   ├── __init__.py
│   ├── guardrails.py
│   └── orchestrator.py
├── .env.example
├── example_usage.py
├── pyproject.toml
└── README.md
```

- **`orchestrator.py`** — Student version with TODOs to complete.
- **`orchestrator_reference.py`** — Complete solution for reference.
- **`guardrails.py`** — PII column detection guardrail.

## Graph visualization

Running `example_usage.py` generates a `graph.png` diagram of the orchestration graph.

## LangSmith (optional)

To enable tracing, set the LangSmith variables in your `.env` file. If they are not set, the pipeline runs normally without tracing.

## Notes
- Both sub-projects ([`data-cleaning-agent`](https://github.com/guibracco/data-cleaning-agent) and [`eda-workflow`](https://github.com/guibracco/eda-workflow)) are linked as **local path dependencies** in `pyproject.toml`. This means they are expected to live in sibling directories (e.g. `../data-cleaning-agent` and `../eda-workflow`). When you run `poetry install`, Poetry resolves them from those local paths rather than from PyPI.
- A PII guardrail runs before any LLM call and blocks the pipeline if sensitive columns are detected.
- If cleaning fails, EDA is skipped.
