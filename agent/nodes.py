import os
import sys
import json
import requests
import time 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

# Add scripts folder to path so we can import retrieval.py
sys.path.append(os.path.join(os.path.dirname(__file__), "../scripts"))
from retrieval import retrieve

# Initialize Gemini - used for Router and Report nodes
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "Nigama-11/mistral-7b-earnings-analyst"


def _call_groq_with_retry(prompt: str, max_retries: int = 3) -> str:
    """Call Groq with retry on rate limit errors."""
    for attempt in range(max_retries):
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait_time = 10 * (attempt + 1)
                print(f"[Groq] Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
    return None



# ── NODE 1: ROUTER ──────────────────────────────────────────────────────────

def router_node(state: dict) -> dict:
    
    """
    Classifies the query as single_quarter or multi_quarter.
    Uses Gemini for this simple classification task.
    """
    query = state["query"]

    prompt = f"""You are classifying a financial analyst question.

Question: "{query}"

Classify this as ONE of:
- "single_quarter": asks about one specific period, company performance, tone, risks, guidance
- "multi_quarter": explicitly compares two periods, asks about changes over time, uses words like "compare", "changed", "last quarter vs", "trend"

Respond with ONLY a JSON object, nothing else:
{{"mode": "single_quarter", "reasoning": "brief reason"}}
or
{{"mode": "multi_quarter", "reasoning": "brief reason"}}"""

    try:
        text = _call_groq_with_retry(prompt)
        if text is None:
            return {**state, "mode": "single_quarter"}

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        result = json.loads(text)
        mode = result.get("mode", "single_quarter")

        print(f"[Router] Mode: {mode} | Reason: {result.get('reasoning', '')}")

        return {
            **state,
            "mode": mode
        }

    except Exception as e:
        print(f"[Router] Error: {e}, defaulting to single_quarter")
        return {**state, "mode": "single_quarter"}


# ── NODE 2: RETRIEVER ───────────────────────────────────────────────────────

def retriever_node(state: dict) -> dict:
    query = state["query"]
    mode = state.get("mode", "single_quarter")
    # Don't filter by ticker - dataset is multi-company
    ticker = None

    top_k = 8 if mode == "multi_quarter" else 5

    try:
        chunks = retrieve(query, ticker=ticker, top_k=top_k)

        if not chunks:
            return {
                **state,
                "retrieved_chunks": [],
                "error": "No relevant chunks found."
            }

        filtered_chunks, warning = _validate_chunks(chunks, query)

        print(f"[Retriever] Retrieved {len(filtered_chunks)} chunks")
        for i, c in enumerate(filtered_chunks):
            print(f"  [{i+1}] {c['date']} | score={c['score']}")

        return {
            **state,
            "retrieved_chunks": filtered_chunks,
            "error": warning
        }

    except Exception as e:
        print(f"[Retriever] Error: {e}")
        return {**state, "retrieved_chunks": [], "error": str(e)}


# ── NODE 3: ANALYZER ────────────────────────────────────────────────────────

def analyzer_node(state: dict) -> dict:
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        return {
            **state,
            "analysis": {
                "tone": "unknown",
                "key_claims": [],
                "risks": [],
                "sentiment": "unknown", 
                "beat_miss_signal": "undeterminable",
                "confidence": "low",
                "summary": "No relevant earnings data found for this query."
            }
        }

    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1} | {chunk['date']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)[:3000]

    analysis = _call_groq_analyzer(query, context)
    print(f"[Analyzer] Complete | Sentiment: {analysis.get('sentiment')}")
    return {**state, "analysis": analysis}


def _call_hf_inference(query: str, context: str) -> dict | None:
    """Call finetuned model via HuggingFace Inference API with retries."""

    prompt = f"""### System:
You are an expert financial analyst. Analyze the provided earnings information and answer questions accurately. Use only information present in the context. Express numerical findings precisely and use proper financial terminology.

### Context:
{context}

### Question:
{query}

### Answer:
"""

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 400,
            "temperature": 0.1,
            "return_full_text": False
        }
    }

    # Retry up to 3 times — model may be cold starting
    for attempt in range(3):
        try:
            print(f"[HF API] Attempt {attempt + 1}/3...")
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{MODEL_ID}",
                headers=headers,
                json=payload,
                timeout=120  # increased to 2 mins for cold start
            )

            if response.status_code == 200:
                result = response.json()
                raw_text = result[0]["generated_text"] if isinstance(result, list) else str(result)
                print(f"[HF API] Success on attempt {attempt + 1}")
                return _parse_analysis_text(raw_text, query)

            elif response.status_code == 503:
                # Model is loading — wait and retry
                wait_time = 20 * (attempt + 1)
                print(f"[HF API] Model loading, waiting {wait_time}s...")
                time.sleep(wait_time)

            else:
                print(f"[HF API] Status {response.status_code}, attempt {attempt + 1}")
                time.sleep(5)

        except Exception as e:
            print(f"[HF API] Exception attempt {attempt + 1}: {e}")
            time.sleep(10)

    return None  # all retries failed, will fall back to Gemini

def _call_groq_analyzer(query: str, context: str) -> dict:
    """Analyzer using Groq."""
    time.sleep(2)

    prompt = f"""You are an expert financial analyst. Analyze this earnings call content and answer the question.

Context:
{context}

Question: {query}

Respond with ONLY a JSON object, no markdown, no backticks:
{{
  "tone": "management tone in one phrase",
  "key_claims": ["claim 1", "claim 2", "claim 3"],
  "risks": ["risk 1", "risk 2"],
  "sentiment": "positive/negative/neutral/mixed",
  "beat_miss_signal": "beat/miss/in-line/undeterminable",
  "confidence": "high/medium/low",
  "summary": "2-3 sentence analyst summary"
}}"""

    try:
        text = _call_groq_with_retry(prompt)
        if text is None:
            raise Exception("All retries failed")

        # Strip any accidental markdown
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        return json.loads(text)

    except Exception as e:
        print(f"[Groq Analyzer] Error: {e}")
        return {
            "tone": "unknown",
            "key_claims": [],
            "risks": [],
            "sentiment": "unknown",
            "beat_miss_signal": "undeterminable",
            "confidence": "low",
            "summary": "Analysis unavailable."
        }


def _parse_analysis_text(text: str, query: str) -> dict:
    """
    Parse freeform model output into structured dict.
    The finetuned model returns text, not JSON,
    so we extract structure from it.
    """
    return {
        "tone": "see summary",
        "key_claims": [text[:500]] if text else [],
        "risks": [],
        "sentiment": "mixed",
        "beat_miss_signal": "undeterminable",
        "confidence": "medium",
        "summary": text[:600] if text else "No analysis generated."
    }


# ── NODE 4: REPORT ──────────────────────────────────────────────────────────

def report_node(state: dict) -> dict:
    """
    Formats the analysis into a clean markdown report.
    Uses Gemini to produce natural language formatting.
    """
    query = state["query"]
    analysis = state.get("analysis") or {}
    chunks = state.get("retrieved_chunks") or []
    ticker = state.get("ticker", "Unknown")
    mode = state.get("mode", "single_quarter")

    # Build sources section
    sources = []
    for i, chunk in enumerate(chunks[:3]):
        sources.append(
            f"> **[Source {i+1}]** {chunk['ticker']} | "
            f"{chunk['date']} | {chunk['speaker']} "
            f"(relevance: {chunk['score']})\n"
            f"> *\"{chunk['text'][:200]}...\"*"
        )
    sources_text = "\n\n".join(sources)

    # Build the report
    key_claims = "\n".join([f"- {c}" for c in analysis.get("key_claims", [])]) or "- See summary below"
    risks = "\n".join([f"- {r}" for r in analysis.get("risks", [])]) or "- No specific risks identified"

    report = f"""## 📊 Earnings Analysis Report

**Company:** {ticker} | **Query Type:** {mode.replace('_', ' ').title()}

---

### 🔍 Query
*{query}*

---

### 📝 Summary
{analysis.get('summary', analysis.get('key_claims', ['No summary available'])[0] if analysis.get('key_claims') else 'No summary available')}

---

### 🎯 Tone & Sentiment
- **Management Tone:** {analysis.get('tone', 'N/A')}
- **Overall Sentiment:** {analysis.get('sentiment', 'N/A')}
- **Beat/Miss Signal:** {analysis.get('beat_miss_signal', 'N/A')}
- **Confidence:** {analysis.get('confidence', 'N/A')}

---

### 📌 Key Claims
{key_claims}

---

### ⚠️ Risk Factors
{risks}

---

### 📚 Sources
{sources_text}

---
*Analysis powered by finetuned Mistral 7B + LangGraph*
"""

    print("[Report] Report generated successfully")

    return {**state, "report": report}



def _validate_chunks(chunks: list, query: str) -> tuple[list, str | None]:
    if not chunks:
        return [], "No relevant content found for this query."

    # Lower threshold from 0.4 to 0.25 to allow more chunks through
    good_chunks = [c for c in chunks if c['score'] > 0.25]

    if not good_chunks:
        return chunks, "⚠️ Low confidence retrieval — answers may be imprecise."

    if len(good_chunks) < len(chunks):
        warning = f"⚠️ {len(chunks) - len(good_chunks)} low-relevance chunks filtered out."
        return good_chunks, warning

    return good_chunks, None