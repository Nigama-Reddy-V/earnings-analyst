import sys
import os
sys.path.append(os.path.dirname(__file__))

import importlib.util

# Load the local chunk.py explicitly to avoid collision with Python's
# built-in `chunk` module.
_chunk_path = os.path.join(os.path.dirname(__file__), "chunk.py")
_spec = importlib.util.spec_from_file_location("local_chunk", _chunk_path)
_chunk_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_chunk_mod)
split_by_speaker = _chunk_mod.split_by_speaker
chunk_segments = _chunk_mod.chunk_segments
split_by_paragraph = _chunk_mod.split_by_paragraph

from embed import embedder, client, COLLECTION_NAME
from qdrant_client.models import PointStruct


def extract_text(uploaded_file) -> str:
    """Extract text from uploaded TXT or PDF file."""
    name = getattr(uploaded_file, "name", None) or getattr(uploaded_file, "filename", "")
    if name.endswith(".pdf"):
        import pypdf
        # If it's a FastAPI UploadFile, PdfReader wants the file object.
        # Otherwise for Streamlit it can read the UploadedFile direct wrapper.
        file_obj = getattr(uploaded_file, "file", None) or uploaded_file
        reader = pypdf.PdfReader(file_obj)
        return "\n".join(page.extract_text() for page in reader.pages)
    else:
        # Check underlying sync file object first to avoid reading async coroutine in FastAPI
        if hasattr(uploaded_file, "file") and hasattr(uploaded_file.file, "read"):
            content = uploaded_file.file.read()
        elif hasattr(uploaded_file, "read"):
            content = uploaded_file.read()
        else:
            raise ValueError("Invalid file object provided to extract_text")
        
        # If content is bytes, decode it
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="ignore")
        return str(content)


def ingest_uploaded_transcript(text: str, filename: str, session_id: str = None) -> int:
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
        chunk['filename'] = filename
        if session_id:
            chunk['session_id'] = session_id

        points.append(PointStruct(
            id=abs(hash(chunk['chunk_id'] + filename)) % (2**63),
            vector=vector,
            payload=chunk
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Ingested {len(chunks)} chunks from {filename} with session_id {session_id}")
    return len(chunks)