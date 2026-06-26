import os
import sys
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Add workspace directory to path so we can import modules properly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.graph import agent
from scripts.ingest_upload import ingest_uploaded_transcript, extract_text

app = FastAPI(
    title="Earnings Analyst Backend",
    description="FastAPI Backend for Earnings Call Analyst Agent",
    version="1.0.0"
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev simplicity, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    model_choice: str = "groq"
    session_id: Optional[str] = None

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    Ingest a PDF or TXT transcript.
    Tags chunks in Qdrant with session_id if provided.
    """
    filename = file.filename
    if not filename.endswith((".txt", ".pdf")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a .txt or .pdf file."
        )
    
    try:
        text = extract_text(file)
        num_chunks = ingest_uploaded_transcript(text, filename, session_id=session_id)
        return {
            "success": True,
            "filename": filename,
            "num_chunks": num_chunks,
            "message": f"Successfully ingested {num_chunks} chunks from {filename}."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Query the Earnings Call Analyst agent.
    If session_id is active, retrieves exclusively from the uploaded document.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    initial_state = {
        "query": request.query,
        "ticker": None,
        "model_choice": request.model_choice,
        "session_id": request.session_id,
        "mode": None,
        "quarters": None,
        "retrieved_chunks": None,
        "analysis": None,
        "report": None,
        "error": None,
        "trace_url": None
    }
    
    try:
        result = agent.invoke(initial_state)
        return {
            "query": request.query,
            "mode": result.get("mode", "single_quarter"),
            "report": result.get("report"),
            "retrieved_chunks": result.get("retrieved_chunks") or [],
            "error": result.get("error")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/debug")
def debug_check():
    import os
    from huggingface_hub import InferenceClient
    
    hf_token = os.getenv("HF_TOKEN")
    hf_token_masked = f"{hf_token[:5]}...{hf_token[-5:]}" if hf_token else "NOT_SET"
    
    # Test InferenceClient
    client_status = "unknown"
    client_preview = ""
    try:
        client = InferenceClient(token=hf_token)
        res = client.feature_extraction(
            model="sentence-transformers/all-MiniLM-L6-v2",
            text="test text for embeddings"
        )
        # Convert memoryview/numpy/list to list representation for preview
        import numpy as np
        if isinstance(res, np.ndarray):
            res_list = res.tolist()
        elif isinstance(res, memoryview):
            res_list = list(res)
        else:
            res_list = res
            
        client_status = "success"
        client_preview = f"Type: {type(res)}, Length: {len(res_list)}, Preview: {str(res_list[:5])}"
    except Exception as e:
        client_status = f"Failed: {str(e)}"
        
    return {
        "HF_TOKEN_configured": hf_token is not None,
        "HF_TOKEN_masked": hf_token_masked,
        "InferenceClient_status": client_status,
        "InferenceClient_preview": client_preview,
        "QDRANT_URL_configured": os.getenv("QDRANT_URL") is not None,
        "GROQ_API_KEY_configured": os.getenv("GROQ_API_KEY") is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
