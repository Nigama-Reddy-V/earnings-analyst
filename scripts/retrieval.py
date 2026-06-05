import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

COLLECTION_NAME = "earnings_transcripts"

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)


def retrieve(
    query: str,
    ticker: str = None,
    top_k: int = 5
) -> list[dict]:
    """
    Retrieve top-k relevant chunks for a query.
    Optionally filter by ticker before doing vector search.
    """
    query_vector = embedder.encode(query).tolist()

    # Build optional metadata filter
    search_filter = None
    if ticker:
        search_filter = Filter(
            must=[FieldCondition(
                key="ticker",
                match=MatchValue(value=ticker.upper())
            )]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=top_k,
        with_payload=True
    )

    return[
        {
            "text": r.payload['text'],
            "ticker": r.payload['ticker'],
            "date": r.payload['date'],
            "speaker": r.payload['speaker'],
            "section": r.payload['section'],
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