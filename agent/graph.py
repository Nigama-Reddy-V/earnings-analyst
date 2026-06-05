from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import router_node, retriever_node, analyzer_node, report_node


def should_continue(state: dict) -> str:
    """
    Conditional edge function.
    If there's an error after retrieval, go straight to report.
    Otherwise continue to analyzer.
    """
    if state.get("error") or not state.get("retrieved_chunks"):
        return "report"
    return "analyzer"


def build_graph():
    """Build and compile the LangGraph agent."""

    graph = StateGraph(AgentState)

    # Add all four nodes
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("report", report_node)

    # Define the flow
    graph.set_entry_point("router")
    graph.add_edge("router", "retriever")

    # Conditional edge after retriever
    # If retrieval failed → skip to report
    # If retrieval succeeded → go to analyzer
    graph.add_conditional_edges(
        "retriever",
        should_continue,
        {
            "analyzer": "analyzer",
            "report": "report"
        }
    )

    graph.add_edge("analyzer", "report")
    graph.add_edge("report", END)

    return graph.compile()


# Build the graph once at module level
agent = build_graph()