import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agent.graph import agent


def test_query(query: str, ticker: str = None):
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"Ticker: {ticker or 'all companies'}")
    print('='*60)

    initial_state = {
        "query": query,
        "ticker": ticker,
        "mode": None,
        "quarters": None,
        "retrieved_chunks": None,
        "analysis": None,
        "report": None,
        "error": None,
        "trace_url": None
    }

    result = agent.invoke(initial_state)
    print("\n--- FINAL REPORT ---")
    print(result["report"])
    return result


if __name__ == "__main__":
    test_query(
        "What did management say about gross margins?",
        ticker=None
    )

    print("\nWaiting 30 seconds before next query...")
    time.sleep(30)

    test_query(
        "What risks did management highlight?",
        ticker=None
    )