"""Orchestrator that chains data-cleaning and EDA sub-graphs.

This file does not perform data cleaning or EDA itself.
Instead, it acts like a traffic controller:
1) check for risky columns (PII),
2) run the cleaning workflow,
3) run the EDA workflow,
4) expose easy-to-read outputs.
"""

import logging
from typing import Optional, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from data_cleaning_agent import make_lightweight_data_cleaning_agent
from eda_workflow.eda_workflow import make_eda_baseline_workflow

from data_analyst_agent.guardrails import check_pii_columns

logger = logging.getLogger(__name__)
# Name for this compiled graph instance.
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
        # Save constructor inputs for reuse and debugging.
        self.model = model
        self.checkpointer = checkpointer
        # Holds the latest full workflow output after invoke_workflow() runs.
        self.response = None
        # Build the orchestration graph once up front.
        # This avoids rebuilding it every time the user runs the pipeline.
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

        # Run the compiled graph with an initial state dictionary containing:
        # - data_raw: original table data
        # - user_instructions: optional cleaning preferences
        # - retry fields: how many times cleaning can retry
        # - pii_flagged_columns: initialized empty, filled by PII node if risky columns are found
        # - data_cleaned: initialized None, filled by cleaning node on success
        # - cleaning_response: initialized empty, filled with full cleaning graph output (including errors)
        # - eda_response: initialized empty, filled with full EDA graph output (summary, recommendations, detailed results)
        # The graph's nodes read from and write to this shared state as the workflow executes.
        response = self._compiled_graph.invoke(
            {
                "data_raw": df.to_dict(),
                "user_instructions": user_instructions,
                "max_retries": max_retries,
                "retry_count": retry_count,
                "pii_flagged_columns": [],
                "data_cleaned": None,
                "cleaning_response": {},
                "eda_response": {},
            },
            # Allow caller to pass runtime options (for example, LangGraph config).
            **kwargs,
        )

        # Save full workflow result so getter methods can read from it later.
        self.response = response
        return None

    def get_data_cleaned(self) -> Optional[pd.DataFrame]:
        """Return the cleaned DataFrame, or ``None`` if unavailable."""
        # The workflow stores cleaned data as a dictionary.
        # Convert it back to a DataFrame for convenience.
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

    def get_pii_flags(self) -> list:
        """Return column names flagged as potential PII, or empty list."""
        if self.response:
            return self.response.get("pii_flagged_columns", [])
        return []


def make_data_analyst_agent(model, checkpointer: Optional[object] = None):
    """Build the parent graph that controls the full pipeline, orchestrating the cleaning and EDA sub-graphs.

    In plain language, this function creates a flowchart:
    - Start at PII check.
    - If safe, run cleaning.
    - If cleaning succeeds, run EDA.
    - Then stop.
    """

    # Compile each sub-graph once so they can be invoked as nodes.
    # These are the two "specialist workers" this orchestrator delegates to.
    # Each one is a prebuilt graph in another package.
    cleaning_graph = make_lightweight_data_cleaning_agent(
        model=model,
        checkpointer=checkpointer,
    )
    eda_graph = make_eda_baseline_workflow(
        model=model,
        checkpointer=checkpointer,
    )

    # Shared state that flows through every node in the parent graph.
    # This TypedDict defines the shared "memory" passed between nodes.
    # Every node reads from and writes to this shared state.
    class OrchestrationState(TypedDict):
        # Raw input data loaded from CSV.
        data_raw: dict
        # Optional user guidance for how cleaning should behave.
        user_instructions: Optional[str]
        # Retry controls for the cleaning sub-graph.
        max_retries: int
        retry_count: int
        # Names of columns that look like personal/sensitive information.
        pii_flagged_columns: list
        # Cleaned data output from the cleaning graph.
        data_cleaned: Optional[dict]
        # Full cleaning graph output (not just cleaned data).
        cleaning_response: dict
        # Full EDA graph output (summary, recommendations, detailed results).
        eda_response: dict

    def pii_check_node(state: OrchestrationState) -> dict:
        """Flag columns that look like PII before any LLM call."""
        logger.info("Running PII guardrail")
        # PII detection here is column-name based.
        # Example: names like "email" or "ssn" should be flagged early.
        columns = list(state.get("data_raw", {}).keys())
        flagged = check_pii_columns(columns)
        if flagged:
            logger.warning("PII guardrail flagged columns: %s", flagged)
        # Return only the piece of state this node is responsible for.
        return {"pii_flagged_columns": flagged}

    def route_after_pii_check(state: OrchestrationState) -> str:
        """Block the pipeline if PII columns were detected."""
        # If anything is flagged, stop immediately.
        # Otherwise move forward to cleaning.
        if state.get("pii_flagged_columns"):
            return "end"
        return "clean_data"

    def clean_data_node(state: OrchestrationState) -> dict:
        """Invoke the cleaning sub-graph and return cleaned data."""
        logger.info("Running cleaning graph")

        # Map parent state keys to the cleaning sub-graph's expected inputs.
        cleaning_response = cleaning_graph.invoke(
            {
                "user_instructions": state.get("user_instructions"),
                "data_raw": state.get("data_raw", {}),
                "max_retries": state.get("max_retries", 3),
                "retry_count": state.get("retry_count", 0),
            }
        )

        # Keep both:
        # - the cleaned data for downstream EDA
        # - the full cleaning response for diagnostics/routing
        return {
            "data_cleaned": cleaning_response.get("data_cleaned"),
            "cleaning_response": cleaning_response,
        }

    def run_eda_node(state: OrchestrationState) -> dict:
        """Invoke the EDA sub-graph on the cleaned data."""
        logger.info("Running EDA graph")

        # Map parent state keys to the EDA sub-graph's expected inputs.
        # Note: EDA expects the table under key "dataframe" (not "data_cleaned").
        eda_response = eda_graph.invoke(
            {
                "dataframe": state.get("data_cleaned", {}),
                # Initialize empty containers required by the EDA workflow.
                # The EDA graph fills these as it executes each analysis step.
                "results": {},
                "observations": {},
                "current_step": "",
                "summary": "",
                "recommendations": [],
            }
        )

        # Store the full EDA response so helper methods can expose key parts later.
        return {"eda_response": eda_response}

    def route_after_cleaning(state: OrchestrationState) -> str:
        """Route to EDA if cleaning succeeded, otherwise end."""
        # Cleaning workflow reports runtime errors in this field.
        cleaning_error = state.get("cleaning_response", {}).get("data_cleaner_error")
        data_cleaned = state.get("data_cleaned")
        # Proceed only if:
        # 1) no cleaning error was reported, and
        # 2) cleaned output exists.
        if cleaning_error is None and data_cleaned is not None:
            return "run_eda"
        return "end"

    # Build the parent flowchart object.
    workflow = StateGraph(OrchestrationState)
    # Register each node function with a stable node name.
    workflow.add_node("pii_check", pii_check_node)
    workflow.add_node("clean_data", clean_data_node)
    workflow.add_node("run_eda", run_eda_node)

    # The pipeline always starts at the PII guardrail.
    workflow.set_entry_point("pii_check")

    # First decision point:
    # - safe data -> cleaning
    # - flagged data -> end
    workflow.add_conditional_edges(
        "pii_check",
        route_after_pii_check,
        {
            "clean_data": "clean_data",
            "end": END,
        },
    )
    # Second decision point:
    # - cleaned successfully -> EDA
    # - cleaning failed -> end
    workflow.add_conditional_edges(
        "clean_data",
        route_after_cleaning,
        {
            "run_eda": "run_eda",
            "end": END,
        },
    )
    # After EDA completes, the workflow ends.
    workflow.add_edge("run_eda", END)

    # Compile the graph into an executable object returned to the caller.
    return workflow.compile(checkpointer=checkpointer, name=AGENT_NAME)
