import sys
import os
sys.path.append(os.path.dirname(__file__))

from chunk import split_by_speaker, chunk_segments, split_by_paragraph
from embed import embedder, client, COLLECTION_NAME
from qdrant_client.models import PointStruct


def extract_text(uploaded_file) -> str:
    """Extract text from uploaded TXT or PDF file."""
    if uploaded_file.name.endswith(".pdf"):
        import pypdf
        reader = pypdf.PdfReader(uploaded_file)
        return "\n".join(page.extract_text() for page in reader.pages)
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")


def ingest_uploaded_transcript(text: str, filename: str) -> int:
    """
    Takes raw transcript text, chunks it, embeds it,
    and upserts into Qdrant — available immediately for querying.
    """
    ticker = filename.split("_")[0].upper() if "_" in filename else "UPLOAD"
    date = filename.replace(".txt", "").replace(".pdf", "")

    segments = split_by_speaker(text)
    chunks = chunk_segments(segments, ticker=ticker, date=date)

    if not chunks:
        print("No chunks created — falling back to paragraph splitting")
        segments = split_by_paragraph(text)
        chunks = chunk_segments(segments, ticker=ticker, date=date)

    texts = [c['text'] for c in chunks]
    vectors = embedder.encode(texts, batch_size=32).tolist()

    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(PointStruct(
            id=abs(hash(chunk['chunk_id'] + filename)) % (2**63),
            vector=vector,
            payload=chunk
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Ingested {len(chunks)} chunks from {filename}")
    return len(chunks)