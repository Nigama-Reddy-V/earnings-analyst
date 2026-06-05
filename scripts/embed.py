import os
import json
import sys
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

load_dotenv()

COLLECTION_NAME = "earnings_transcripts"
VECTOR_SIZE = 384

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded.")

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)


def setup_collection():
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print("Cleared old collection")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    print(f"Created collection: {COLLECTION_NAME}")


def embed_and_upload(chunks: list[dict], batch_size: int = 32):
    print(f"Uploading {len(chunks)} chunks to Qdrant...")
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        texts = [c['text'] for c in batch]
        vectors = embedder.encode(texts, show_progress_bar=False).tolist()
        points = []
        for chunk, vector in zip(batch, vectors):
            points.append(PointStruct(
                id=abs(hash(chunk['chunk_id'])) % (2**63),
                vector=vector,
                payload=chunk
            ))
        client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Upload complete. Total: {client.count(collection_name=COLLECTION_NAME).count}")


if __name__ == "__main__":
    # Load the earnings QA dataset
    data_path = os.path.join(os.path.dirname(__file__), "../data/earnings_qa.json")
    
    with open(data_path, 'r') as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} examples from dataset")

    # Convert to chunks
    chunks = []
    for i, item in enumerate(raw_data):
        transcript = item.get('transcript', '').strip()
        question = item.get('question', '').strip()
        answer = item.get('answer', '').strip()
        date = item.get('date', 'unknown')

        # Skip empty or useless answers
        if not transcript or not answer or len(answer) < 30:
            continue
        if 'I do not know' in answer and len(answer) < 100:
            continue

        # Create a rich chunk combining transcript context + Q&A
        text = f"Q: {question}\nA: {answer}\nContext: {transcript[:500]}"

        chunks.append({
            "chunk_id": f"earningsqa_{i}",
            "ticker": "MULTI",  # mixed companies
            "date": date,
            "speaker": "analyst_qa",
            "section": "qa",
            "text": text,
            "token_count": len(text.split()),
            "question": question,
            "answer": answer
        })

    print(f"Created {len(chunks)} valid chunks")

    setup_collection()
    embed_and_upload(chunks)