import os
import sys
import json
import requests
import time 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv(override=False)

# Add scripts folder to path so we can import retrieval.py
sys.path.append(os.path.join(os.path.dirname(__file__), "../scripts"))
from retrieval import retrieve # type: ignore

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
    Classifies the query as single_quarter, multi_quarter, or general.
    Uses Groq for this simple classification task.
    """
    query = state["query"]

    prompt = f"""You are classifying a financial analyst question.

Question: "{query}"

Classify this as ONE of:
- "single_quarter": asks about SPECIFIC data from a SPECIFIC company's earnings call transcript (e.g. "What was Apple's revenue?", "What risks did the CEO mention?", "What was the guidance for next quarter?"). The question must clearly be asking to look up data from a transcript.
- "multi_quarter": explicitly compares two or more time periods, uses words like "compare", "changed", "vs", "trend", "across quarters"
- "general": ANY question that is conceptual, definitional, or educational — even if it uses financial terms like "tone", "margins", "EPS", "guidance". If the question does NOT reference a specific company or transcript, it is general. Examples: "what is management tone", "what is EPS", "explain gross margin", "how do earnings calls work", "what are risk factors"

IMPORTANT: When in doubt between "general" and "single_quarter", choose "general". Only choose "single_quarter" if the question clearly asks to look up specific data from a transcript.

Respond with ONLY a JSON object, nothing else:
{{"mode": "single_quarter", "reasoning": "brief reason"}}
or
{{"mode": "multi_quarter", "reasoning": "brief reason"}}
or
{{"mode": "general", "reasoning": "brief reason"}}"""

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

        # Validate mode
        if mode not in ("single_quarter", "multi_quarter", "general"):
            mode = "single_quarter"

        print(f"[Router] Mode: {mode} | Reason: {result.get('reasoning', '')}")

        return {
            **state,
            "mode": mode
        }

    except Exception as e:
        print(f"[Router] Error: {e}, defaulting to single_quarter")
        return {**state, "mode": "single_quarter"}


# ── NODE 1b: GENERAL CHAT ───────────────────────────────────────────────────

def general_chat_node(state: dict) -> dict:
    """
    Handles general financial knowledge questions that don't require
    earnings transcript retrieval.  Uses the base Groq model directly.
    """
    query = state["query"]

    prompt = f"""You are a helpful financial analyst assistant specializing in
earnings calls, corporate finance, and investment analysis.

Answer this question clearly and helpfully: {query}

If it's about earnings calls, transcripts, or financial analysis, give a
detailed educational answer.  Use proper financial terminology and provide
concrete examples where relevant.  Keep it conversational but informative.

Format your answer in clean markdown with headers and bullet points where
appropriate."""

    try:
        answer = _call_groq_with_retry(prompt)
        if answer is None:
            answer = "Sorry, I couldn't generate a response at this time. Please try again."
    except Exception as e:
        print(f"[GeneralChat] Error: {e}")
        answer = f"An error occurred while generating the response: {e}"

    # Format as a report so the UI can render it with st.markdown()
    report = f"""## 💬 General Q&A

**Query:** *{query}*

---

{answer}

---

*Powered by LangGraph · Groq Llama 3.1*
"""

    print(f"[GeneralChat] Response generated for: {query[:60]}...")
    return {**state, "report": report}

# ── NODE 2: RETRIEVER ───────────────────────────────────────────────────────

def retriever_node(state: dict) -> dict:
    query = state["query"]
    mode = state.get("mode", "single_quarter")
    session_id = state.get("session_id")
    # Don't filter by ticker - dataset is multi-company
    ticker = None

    top_k = 12 if mode == "multi_quarter" else 5

    try:
        chunks = retrieve(query, ticker=ticker, session_id=session_id, top_k=top_k)

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
            "warning": warning
        }

    except Exception as e:
        print(f"[Retriever] Error: {e}")
        return {**state, "retrieved_chunks": [], "error": str(e)}


# ── NODE 3: ANALYZER ────────────────────────────────────────────────────────

def analyzer_node(state: dict) -> dict:
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    model_choice = state.get("model_choice", "groq")

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

    analysis = None

    # Route based on model_choice
    if model_choice == "mistral":
        print("[Analyzer] Trying finetuned Mistral 7B via HuggingFace...")
        analysis = _call_hf_inference(query, context)
        if analysis is None:
            print("[Analyzer] HF model unreachable, falling back to Groq")

    # Default to Groq, or fallback if HF failed
    if analysis is None:
        if model_choice == "mistral":
            print("[Analyzer] Using Groq as fallback")
        analysis = _call_groq_analyzer(query, context)

    print(f"[Analyzer] Complete | Model: {'mistral→groq fallback' if model_choice == 'mistral' and analysis.get('_source') != 'hf' else model_choice} | Sentiment: {analysis.get('sentiment')}")
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

    prompt = f"""You are an expert financial analyst. Analyze the earnings call transcript excerpts below and answer the question thoroughly.

IMPORTANT: Base your analysis ONLY on the provided context. Extract specific numbers, percentages, and quotes where available.

Context from earnings call transcript:
{context}

Question: {query}

Respond with ONLY a valid JSON object, no markdown, no backticks, no extra text:
{{
  "tone": "describe management's tone in one phrase (e.g. cautiously optimistic, defensive, confident)",
  "key_claims": ["specific claim with numbers from the transcript", "another specific claim", "third claim"],
  "risks": ["specific risk mentioned in the call", "another risk factor"],
  "sentiment": "positive/negative/neutral/mixed",
  "beat_miss_signal": "beat/miss/in-line/undeterminable",
  "confidence": "high/medium/low",
  "summary": "Write a detailed 4-5 sentence analyst summary. Include specific revenue figures, growth rates, margins, and management outlook. Cover both strengths and concerns."
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
    Formats the analysis into a comprehensive, visually structured markdown
    report.  Output is a plain markdown string consumed by st.markdown() in
    the Streamlit UI — no Streamlit imports are used here.
    """
    query = state["query"]
    analysis = state.get("analysis") or {}
    chunks = state.get("retrieved_chunks") or []
    mode = state.get("mode", "single_quarter")

    # ── Sources ──────────────────────────────────────────────────────────
    sources_lines: list[str] = []
    for i, chunk in enumerate(chunks[:5]):
        score_pct = f"{chunk['score'] * 100:.0f}%"
        source_id = chunk.get("filename") or f"{chunk['ticker']} · {chunk['date']}"
        sources_lines.append(
            f"> **[Source {i+1}]** {source_id} · {chunk['speaker']} "
            f"(relevance {score_pct})\n"
            f"> *\"{chunk['text'][:250]}…\"*"
        )
    sources_text = "\n\n".join(sources_lines) if sources_lines else "> _No sources available._"

    # ── Key Claims ───────────────────────────────────────────────────────
    claims = analysis.get("key_claims", [])
    if claims:
        key_claims_md = "\n".join(f"- {c}" for c in claims)
    else:
        key_claims_md = "- _No specific claims extracted — see summary below._"

    # ── Risk Factors ─────────────────────────────────────────────────────
    risks = analysis.get("risks", [])
    if risks:
        risks_md = "\n".join(f"- ⚠️ {r}" for r in risks)
    else:
        risks_md = "- _No notable risk factors identified in the retrieved context._"

    # ── Expanded multi-paragraph summary ─────────────────────────────────
    raw_summary = analysis.get("summary", "No summary available.")

    # Build a richer synthesis that weaves in claims + risks
    summary_parts: list[str] = [raw_summary]

    if claims:
        claims_narrative = (
            "**Key findings from the call include:** "
            + "; ".join(claims[:4])
            + "."
        )
        summary_parts.append(claims_narrative)

    if risks:
        risk_narrative = (
            "**On the risk side, management flagged the following concerns:** "
            + "; ".join(risks[:3])
            + ". Investors should weigh these factors when evaluating "
            "forward guidance and medium-term positioning."
        )
        summary_parts.append(risk_narrative)

    # Confidence caveat
    confidence = analysis.get("confidence", "low")
    if confidence == "low":
        summary_parts.append(
            "_**Note:** The analysis confidence is **low** — the retrieved "
            "context may not fully cover the question. Consider refining "
            "the query or uploading more transcript data._"
        )
    elif confidence == "medium":
        summary_parts.append(
            "_**Note:** Analysis confidence is **medium**. The retrieved "
            "excerpts partially address the question; additional context "
            "could improve precision._"
        )

    expanded_summary = "\n\n".join(summary_parts)

    # ── Tone & Signals table ─────────────────────────────────────────────
    tone = analysis.get("tone", "N/A")
    sentiment = analysis.get("sentiment", "N/A")
    beat_miss = analysis.get("beat_miss_signal", "N/A")

    # Emoji badges for quick visual scanning
    sentiment_badge = {
        "positive": "🟢 Positive",
        "negative": "🔴 Negative",
        "neutral": "🟡 Neutral",
        "mixed": "🟠 Mixed",
    }.get(sentiment.lower() if isinstance(sentiment, str) else "", sentiment)

    beat_miss_badge = {
        "beat": "✅ Beat",
        "miss": "❌ Miss",
        "in-line": "➖ In-Line",
        "undeterminable": "❓ Undeterminable",
    }.get(beat_miss.lower() if isinstance(beat_miss, str) else "", beat_miss)

    confidence_badge = {
        "high": "🟢 High",
        "medium": "🟡 Medium",
        "low": "🔴 Low",
    }.get(confidence.lower() if isinstance(confidence, str) else "", confidence)

    # ── Assemble final report ────────────────────────────────────────────
    mode_label = mode.replace("_", " ").title()

    report = f"""## 📊 Earnings Analysis Report

**Query:** *{query}*
**Analysis Mode:** {mode_label}

---

### 📝 Summary & Synthesis

{expanded_summary}

---

### 🎯 Tone & Sentiment

| Metric | Result |
|:-------|:-------|
| **Management Tone** | {tone} |
| **Overall Sentiment** | {sentiment_badge} |
| **Beat / Miss Signal** | {beat_miss_badge} |
| **Confidence Level** | {confidence_badge} |

---

### 📌 Key Claims

{key_claims_md}

---

### ⚠️ Risk Factors

{risks_md}

---

### 📚 Sources Referenced

{sources_text}

---

*Powered by LangGraph · Qdrant · Groq Llama 3.1*
"""

    print("[Report] Report generated successfully")
    return {**state, "report": report}



def _validate_chunks(chunks: list, query: str) -> tuple[list, str | None]:
    if not chunks:
        return [], "No relevant content found for this query."

    # Use a low threshold so broad/summary queries still get enough context
    good_chunks = [c for c in chunks if c['score'] > 0.15]

    if not good_chunks:
        return chunks, "⚠️ Low confidence retrieval — answers may be imprecise."

    if len(good_chunks) < len(chunks):
        warning = f"⚠️ {len(chunks) - len(good_chunks)} low-relevance chunks filtered out."
        return good_chunks, warning

    return good_chunks, None