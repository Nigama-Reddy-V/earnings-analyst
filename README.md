# 📊 Earnings Call Analyst Agent

An AI-powered financial analyst that answers natural language questions about SEC earnings call transcripts using **Graph RAG**, **LangGraph**, and a **domain-finetuned Mistral 7B** model.

## 🎯 What it does

Ask questions like:
- *"What was Apple's management tone on gross margins in Q3 2024?"*
- *"How did Microsoft's cloud guidance change compared to last quarter?"*
- *"Did Google beat earnings expectations?"*

The agent retrieves relevant transcript sections, analyzes them with a finetuned financial model, and returns a structured analyst-style report with cited sources.

## 🏗️ Architecture

```
User Query
    ↓
Router Node (Gemini) — classifies single vs multi-quarter
    ↓
Retriever Node — semantic search over Qdrant vector store
    ↓
Analyzer Node (Finetuned Mistral 7B) — domain analysis
    ↓
Report Node (Gemini) — structured markdown output
```

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| Agent orchestration | LangGraph |
| Vector database | Qdrant |
| Embeddings | all-MiniLM-L6-v2 |
| Base LLM | Gemini 1.5 Flash |
| Finetuned model | Mistral 7B (LoRA, QLoRA via Unsloth) |
| Training data | finance-alpaca (68k examples) |
| Observability | LangSmith |
| Data source | SEC EDGAR API |
| UI | Streamlit |

## 📈 Evaluation

**Retrieval Quality:** 65% keyword hit rate across 10 queries

**Manual Agent Evaluation (5 test cases):**
- Gross margin query: ✅ Returned specific margin percentages and management tone
- Risk factors query: ✅ Identified operational and macroeconomic risks  
- Guidance query: ✅ Provided forward-looking statements
- Cloud revenue query: ✅ Returned growth metrics
- Beat/miss query: ✅ Compared performance against expectations

**Finetuned Model Evaluation:**
- Quantitative reasoning: 4/5
- Qualitative analysis: 2/5
- Overall: 2.8/5

## 🚀 Setup

```bash
git clone https://github.com/Nigama-11/earnings-analyst
cd earnings-analyst
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your API keys to `.env`:
```
QDRANT_URL=...
QDRANT_API_KEY=...
GEMINI_API_KEY=...
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=earnings-analyst
HF_TOKEN=...
```

```bash
streamlit run app.py
```

## 🎥 Demo

[Loom demo link here]

## 🤖 Finetuned Model

Model: [Nigama-11/mistral-7b-earnings-analyst](https://huggingface.co/Nigama-11/mistral-7b-earnings-analyst)

Finetuned on finance-alpaca dataset using QLoRA (rank=16) via Unsloth on Google Colab T4 GPU. Training time: ~50 minutes.