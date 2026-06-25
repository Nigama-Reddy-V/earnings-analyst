import sys
import os

# Prevent transformers from importing TensorFlow/Keras (not used in this project)
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(__file__))

from agent.graph import agent

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Earnings Analyst Agent",
    page_icon="📊",
    layout="wide"
)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.title("📊 Earnings Call Analyst Agent")
st.markdown(
    "Ask natural language questions about earnings call transcripts. "
    "Powered by **LangGraph** + **Finetuned Mistral 7B** + **RAG**."
)
st.divider()

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Selection")
    model_option = st.radio(
        "Choose the analysis model",
        options=["Groq Llama 3.1", "Finetuned Mistral 7B"],
        index=0,
        help="Mistral 7B is the finetuned model hosted on HuggingFace. "
             "If it's unreachable (cold start), the system falls back to Groq automatically."
    )
    model_choice = "mistral" if "Mistral" in model_option else "groq"

    if model_choice == "mistral":
        st.caption("⚡ _Mistral may take ~30s on first call (cold start). "
                   "Falls back to Groq if unreachable._")

    ticker = None

    st.divider()
    st.markdown("### 💡 Example queries")
    examples = [
        "What was management's tone on gross margins?",
        "Did the company beat earnings expectations?",
        "What risks did management highlight?",
        "How did cloud revenue perform vs guidance?",
        "What is the guidance for next quarter?",
        "How did margins change compared to last quarter?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.query_input = ex

    st.divider()
    st.markdown("### 🏗️ Architecture")
    st.markdown("""
    1. **Router** — classifies query type
    2. **Retriever** — fetches relevant chunks from Qdrant
    3. **Analyzer** — Groq or Finetuned Mistral 7B
    4. **Report** — formats structured output
    """)


# Initialize session ID in st.session_state
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# ── FILE UPLOAD ──────────────────────────────────────────────────────────────
st.subheader("📂 Upload Your Own Transcript")
uploaded_file = st.file_uploader(
    "Upload an earnings call transcript (TXT or PDF)",
    type=["txt", "pdf"]
)

# Track if file has been cleared from the uploader
if uploaded_file:
    st.session_state["uploaded_filename"] = uploaded_file.name
else:
    st.session_state["has_uploaded_file"] = False
    st.session_state["uploaded_filename"] = None

if uploaded_file:
    if st.button("📥 Ingest Transcript"):
        with st.spinner(f"Ingesting {uploaded_file.name}..."):
            try:
                sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
                from ingest_upload import ingest_uploaded_transcript, extract_text
                text = extract_text(uploaded_file)
                num_chunks = ingest_uploaded_transcript(
                    text, uploaded_file.name, session_id=st.session_state.session_id
                )
                st.success(f"✅ Ingested {num_chunks} chunks from {uploaded_file.name}. Now ask questions below.")
                st.session_state["last_action"] = "ingested"
                st.session_state["has_uploaded_file"] = True
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

st.divider()

# ── MAIN INPUT ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input(
        "Ask a question about earnings calls",
        value=st.session_state.get("query_input", ""),
        placeholder="e.g. What did management say about gross margins?",
        key="query_input"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.button("🔍 Analyze", type="primary", use_container_width=True)

# ── AGENT EXECUTION ──────────────────────────────────────────────────────────
if submit and query:

    # All queries now flow through the LangGraph agent.
    # The router_node classifies into single_quarter / multi_quarter / general
    # and routes accordingly (general bypasses retriever + analyzer).
    initial_state = {
        "query": query,
        "ticker": None,
        "model_choice": model_choice,
        "session_id": st.session_state.session_id if st.session_state.get("has_uploaded_file") else None,
        "mode": None,
        "quarters": None,
        "retrieved_chunks": None,
        "analysis": None,
        "report": None,
        "error": None,
        "trace_url": None
    }

    with st.spinner("🤖 Agent analyzing..."):
        try:
            result = agent.invoke(initial_state)
        except Exception as e:
            st.error(f"Agent error: {e}")
            st.stop()

    if result.get("report"):
        st.markdown(result["report"])

    chunks = result.get("retrieved_chunks") or []
    if chunks:
        with st.expander(f"📚 View {len(chunks)} sources", expanded=False):
            for i, chunk in enumerate(chunks):
                st.markdown(f"**Source {i+1}** | `{chunk['date']}` | Score: `{chunk['score']}`")
                st.markdown(f"> {chunk['text'][:300]}...")
                st.divider()

    # ── LANGSMITH TRACE ──────────────────────────────────────────────────────
    # st.divider()
    # col1, col2 = st.columns([3, 1])
    # with col1:
    #     st.caption(f"Query mode: `{result.get('mode', 'unknown')}` | "
    #               f"Chunks retrieved: `{len(chunks)}`")
    # with col2:
    #     st.link_button(
    #         "🔗 View LangSmith Trace",
    #         "https://smith.langchain.com",
    #         use_container_width=True
    #     )

elif submit and not query:
    st.warning("Please enter a question first.")

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with LangGraph · Qdrant · Finetuned Mistral 7B · "
    "SEC EDGAR · LangSmith | "
    "[GitHub](https://github.com/Nigama-11/earnings-analyst)"
)