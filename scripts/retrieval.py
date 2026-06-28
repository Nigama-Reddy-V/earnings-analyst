import os
from dotenv import load_dotenv
from hf_embedder import HuggingFaceEmbedder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

COLLECTION_NAME = "earnings_transcripts"

embedder = HuggingFaceEmbedder('sentence-transformers/all-MiniLM-L6-v2')
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)


def retrieve(
    query: str,
    ticker: str = None,
    session_id: str = None,
    top_k: int = 5
) -> list[dict]:
    """
    Retrieve top-k relevant chunks for a query.
    Optionally filter by ticker or session_id before doing vector search.
    """
    query_vector = embedder.encode(query).tolist()

    # SECURITY: Always enforce session isolation.
    # If no session_id is provided, return empty — never search across all sessions.
    if not session_id:
        print("[Retriever] No session_id provided — refusing to search without isolation.")
        return []

    search_filter = Filter(
        must=[FieldCondition(
            key="session_id",
            match=MatchValue(value=session_id)
        )]
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=top_k,
        with_payload=True
    )

    return [
        {
            **r.payload,
            "score": round(r.score, 4)
        }
        for r in results.points
    ]


if __name__ == "__main__":
    # Test queries — run these and manually check if results are relevant
    test_queries = [
        ("what did management say about gross margins?", "AAPL"),
        ("what was the revenue guidance for next quarter?", "MSFT"),
        ("what risks did management mention?", None),
        ("how did iPhone sales perform?", "AAPL"),
        ("what was said about cloud growth?", "MSFT"),
    ]

    for query, ticker in test_queries:
        filter_label = f"[{ticker}]" if ticker else "[all companies]"
        print(f"\nQuery: '{query}' {filter_label}")
        print("-" * 60)

        results = retrieve(query, ticker=ticker, top_k=3)

        for i, r in enumerate(results):
            print(f"  Result {i+1} | Score: {r['score']} | "
                  f"{r['ticker']} {r['date']} | {r['speaker']}")
            print(f"  {r['text'][:200]}...")
            print()