import sys
import os
sys.path.append(os.path.dirname(__file__))

from retrieval import retrieve

# 10 test queries with what you expect to find
# Rate each result yourself: 3 = perfect, 2 = relevant, 1 = weak, 0 = wrong
test_cases = [
    {
        "query": "what did management say about gross margins?",
        "ticker": None,
        "expected_keywords": ["margin", "gross", "basis points", "percent"]
    },
    {
        "query": "iPhone revenue growth this quarter",
        "ticker": None,
        "expected_keywords": ["iphone", "revenue", "growth", "billion"]
    },
    {
        "query": "what is the guidance for next quarter?",
        "ticker": None,
        "expected_keywords": ["guidance", "quarter", "expect", "outlook"]
    },
    {
        "query": "cloud revenue performance and growth rate",
        "ticker": None,
        "expected_keywords": ["cloud", "azure", "revenue", "growth"]
    },
    {
        "query": "what risks and headwinds did management mention?",
        "ticker": None,
        "expected_keywords": ["risk", "headwind", "challenge", "pressure"]
    },
    {
        "query": "operating expenses and cost management",
        "ticker": None,
        "expected_keywords": ["operating", "expense", "cost", "billion"]
    },
    {
        "query": "analyst question about capital allocation",
        "ticker": None,
        "expected_keywords": ["buyback", "dividend", "capital", "return"]
    },
    {
        "query": "services revenue and growth",
        "ticker": None,
        "expected_keywords": ["services", "revenue", "growth", "app store"]
    },
    {
        "query": "what did the CFO say about cash flow?",
        "ticker": None,
        "expected_keywords": ["cash", "flow", "operating", "billion"]
    },
    {
        "query": "management comments on AI investments",
        "ticker": None,
        "expected_keywords": ["ai", "artificial intelligence", "investment", "copilot"]
    },
]

def keyword_hit_rate(text: str, keywords: list[str]) -> float:
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)


print("=" * 70)
print("RETRIEVAL QUALITY EVALUATION")
print("=" * 70)

total_score = 0
total_queries = len(test_cases)

for i, tc in enumerate(test_cases):
    results = retrieve(tc['query'], ticker=tc.get('ticker'), top_k=3)
    ticker_label = tc.get('ticker') or 'all'

    if not results:
        print(f"\nQ{i+1}: '{tc['query']}' [{ticker_label}]")
        print("  NO RESULTS RETURNED — check your Qdrant connection")
        continue

    # Score based on keyword hits across top 3 results
    combined_text = ' '.join([r['text'] for r in results])
    hit_rate = keyword_hit_rate(combined_text, tc['expected_keywords'])
    avg_score = sum(r['score'] for r in results) / len(results)
    total_score += hit_rate

    status = "GOOD" if hit_rate >= 0.5 else "WEAK"
    print(f"\nQ{i+1} [{status}]: '{tc['query']}' [{ticker_label}]")
    print(f"  Keyword hit rate: {hit_rate:.0%} | Avg similarity: {avg_score:.3f}")
    print(f"  Top result: {results[0]['ticker']} {results[0]['date']} "
          f"| {results[0]['speaker']} | score={results[0]['score']}")
    print(f"  Preview: {results[0]['text'][:150]}...")

print("\n" + "=" * 70)
print(f"OVERALL: {total_score/total_queries:.0%} keyword hit rate across 10 queries")
if total_score/total_queries >= 0.5:
    print("Retrieval quality is GOOD. Proceed to Phase 2.")
else:
    print("Retrieval quality is WEAK. See fixes below before proceeding.")
print("=" * 70)