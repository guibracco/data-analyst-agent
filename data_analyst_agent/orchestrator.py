"""Orchestrator that chains data-cleaning and EDA sub-graphs."""

import logging
from typing import Optional, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from data_cleaning_agent import make_lightweight_data_cleaning_agent
from eda_workflow.eda_workflow import make_eda_baseline_workflow

logger = logging.getLogger(__name__)
AGENT_NAME = "data_analyst_agent"


class DataAnalystAgent:
    """Thin orchestration layer for data cleaning and EDA.

    Compiles a LangGraph parent graph that runs the cleaning sub-graph
    first and, on success, feeds the cleaned data into the EDA sub-graph.

    Parameters
    ----------
    model : langchain_core.language_models.BaseChatModel
        The chat model used by both sub-graphs.
    checkpointer : object, optional
        LangGraph checkpointer for state persistence.

    Attributes
    ----------
    response : dict or None
        Raw output from the last ``invoke_workflow`` call.
    """

    def __init__(self, model, checkpointer: Optional[object] = None) -> None:
        self.model = model
        self.checkpointer = checkpointer
        self.response = None
        self._compiled_graph = make_data_analyst_agent(
            model=model,
            checkpointer=checkpointer,
        )

    def invoke_workflow(
        self,
        filepath: str,
        user_instructions: Optional[str] = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ) -> None:
        """Read a CSV and run the full cleaning-then-EDA pipeline."""
        df = pd.read_csv(filepath)

        response = self._compiled_graph.invoke(
            {
                "data_raw": df.to_dict(),
                "user_instructions": user_instructions,
                "max_retries": max_retries,
                "retry_count": retry_count,
                "data_cleaned": None,
                "cleaning_response": {},
                "eda_response": {},
            },
            **kwargs,
        )

        self.response = response
        return None

    def get_data_cleaned(self) -> Optional[pd.DataFrame]:
        """Return the cleaned DataFrame, or ``None`` if unavailable."""
        if self.response and self.response.get("data_cleaned"):
            return pd.DataFrame(self.response.get("data_cleaned"))
        return None

    def get_eda_summary(self) -> Optional[str]:
        """Return the EDA summary text, or ``None`` if unavailable."""
        if self.response:
            return self.response.get("eda_response", {}).get("summary")
        return None

    def get_eda_recommendations(self) -> Optional[list]:
        """Return the list of EDA recommendations, or ``None`` if unavailable."""
        if self.response:
            return self.response.get("eda_response", {}).get("recommendations")
        return None

    def get_eda_results(self) -> Optional[dict]:
        """Return the raw EDA results dict, or ``None`` if unavailable."""
        if self.response:
            return self.response.get("eda_response", {}).get("results")
        return None


def make_data_analyst_agent(model, checkpointer: Optional[object] = None):
    """Build a parent graph that orchestrates existing cleaning and EDA graphs."""

    cleaning_graph = make_lightweight_data_cleaning_agent(
        model=model,
        checkpointer=checkpointer,
    )
    eda_graph = make_eda_baseline_workflow(
        model=model,
        checkpointer=checkpointer,
    )

    class OrchestrationState(TypedDict):
        data_raw: dict
        user_instructions: Optional[str]
        max_retries: int
        retry_count: int
        data_cleaned: Optional[dict]
        cleaning_response: dict
        eda_response: dict

    def clean_data_node(state: OrchestrationState) -> dict:
        """Invoke the cleaning sub-graph and return cleaned data."""
        logger.info("Running cleaning graph")

        cleaning_response = cleaning_graph.invoke(
            {
                "user_instructions": state.get("user_instructions"),
                "data_raw": state.get("data_raw", {}),
                "max_retries": state.get("max_retries", 3),
                "retry_count": state.get("retry_count", 0),
            }
        )

        return {
            "data_cleaned": cleaning_response.get("data_cleaned"),
            "cleaning_response": cleaning_response,
        }

    def run_eda_node(state: OrchestrationState) -> dict:
        """Invoke the EDA sub-graph on the cleaned data."""
        logger.info("Running EDA graph")

        eda_response = eda_graph.invoke(
            {
                "dataframe": state.get("data_cleaned", {}),
                "results": {},
                "observations": {},
                "current_step": "",
                "summary": "",
                "recommendations": [],
            }
        )

        return {"eda_response": eda_response}

    def route_after_cleaning(state: OrchestrationState) -> str:
        """Route to EDA if cleaning succeeded, otherwise end."""
        cleaned = state.get("data_cleaned")
        cleaning_error = state.get("cleaning_response", {}).get(
            "data_cleaner_error",
        )
        if cleaned and cleaning_error is None:
            return "run_eda"
        return "end"

    workflow = StateGraph(OrchestrationState)
    workflow.add_node("clean_data", clean_data_node)
    workflow.add_node("run_eda", run_eda_node)

    workflow.set_entry_point("clean_data")
    workflow.add_conditional_edges(
        "clean_data",
        route_after_cleaning,
        {"run_eda": "run_eda", "end": END},
    )
    workflow.add_edge("run_eda", END)

    return workflow.compile(checkpointer=checkpointer, name=AGENT_NAME)


# Backward-compatible aliases for earlier naming
DataCleaningEDAAgent = DataAnalystAgent
make_data_cleaning_eda_agent = make_data_analyst_agent
