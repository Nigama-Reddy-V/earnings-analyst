import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from agent.graph import agent

# Tell RAGAS to use Groq instead of Google
groq_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

wrapped_llm = LangchainLLMWrapper(groq_llm)
wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)

# 5 evaluation questions
eval_data = [
    {
        "question": "What did management say about gross margins?",
        "ground_truth": "Management discussed gross margin performance including cost pressures and expansion drivers"
    },
    {
        "question": "What risks did management highlight?",
        "ground_truth": "Management highlighted macroeconomic risks, competitive pressures, and operational challenges"
    },
    {
        "question": "What was the guidance for next quarter?",
        "ground_truth": "Management provided revenue and margin guidance for the upcoming quarter"
    },
    {
        "question": "How did cloud revenue perform?",
        "ground_truth": "Cloud revenue showed growth with specific percentage increases mentioned"
    },
    {
        "question": "Did the company beat earnings expectations?",
        "ground_truth": "Company performance was compared against analyst consensus estimates"
    },
]

questions = []
answers = []
contexts = []
ground_truths = []

print("Running evaluation queries...")
for item in eval_data:
    state = {
        "query": item["question"],
        "ticker": None,
        "mode": None,
        "quarters": None,
        "retrieved_chunks": None,
        "analysis": None,
        "report": None,
        "error": None,
        "trace_url": None
    }

    result = agent.invoke(state)
    chunks = result.get("retrieved_chunks", [])

    questions.append(item["question"])
    answers.append(result.get("report", "No answer"))
    contexts.append([c["text"] for c in chunks] if chunks else [""])
    ground_truths.append(item["ground_truth"])
    print(f"  ✓ {item['question'][:50]}")

    import time
    time.sleep(5)  # avoid rate limits

# Build RAGAS dataset
ragas_dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

print("\nRunning RAGAS evaluation...")
results = evaluate(
    ragas_dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=wrapped_llm,
    embeddings=wrapped_embeddings
)

print(f"\n=== RAGAS Results ===")
print(f"Faithfulness: {results['faithfulness']:.3f}")
print(f"Answer Relevancy: {results['answer_relevancy']:.3f}")
print("\nSave these scores for your README.")