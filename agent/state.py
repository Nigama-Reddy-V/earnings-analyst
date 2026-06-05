from typing import TypedDict, List, Optional


class AgentState(TypedDict):
    """
    Shared state that flows through every node in the graph.
    Each node reads from and writes to this state.
    """
    # Input
    query: str                          # original user question
    ticker: Optional[str]               # company ticker e.g. "AAPL"

    # Router output
    mode: Optional[str]                 # "single_quarter" or "multi_quarter"
    quarters: Optional[List[str]]       # ["2024-09-30"] or two quarters

    # Retriever output
    retrieved_chunks: Optional[List[dict]]  # chunks from Qdrant

    # Analyzer output
    analysis: Optional[dict]            # structured JSON from finetuned model

    # Report output
    report: Optional[str]               # final markdown report shown to user

    # Metadata
    error: Optional[str]                # error message if something fails
    trace_url: Optional[str]            # LangSmith trace URL