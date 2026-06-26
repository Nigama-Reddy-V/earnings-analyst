import os
import time
from huggingface_hub import InferenceClient

class HuggingFaceEmbedder:
    """
    Drop-in replacement for SentenceTransformer that uses Hugging Face's 
    official InferenceClient, using 0MB of local RAM.
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.hf_token = os.getenv("HF_TOKEN")
        self.client = InferenceClient(token=self.hf_token)

    def encode(self, sentences, batch_size=32, show_progress_bar=False):
        is_single = isinstance(sentences, str)
        if is_single:
            sentences_list = [sentences]
        else:
            sentences_list = sentences

        embeddings = []

        for i in range(0, len(sentences_list), batch_size):
            batch = sentences_list[i:i + batch_size]
            
            success = False
            for attempt in range(5):
                try:
                    res = self.client.feature_extraction(
                        model=self.model_name,
                        text=batch
                    )
                    
                    # Convert to standard Python lists
                    if hasattr(res, "tolist"):
                        res_list = res.tolist()
                    elif isinstance(res, memoryview):
                        res_list = list(res)
                    else:
                        res_list = res

                    # If it's a batch of 1 and returned a flat list of numbers (single vector),
                    # wrap it in a list to keep a list-of-lists structure.
                    if len(batch) == 1:
                        if len(res_list) > 0 and not isinstance(res_list[0], (list, memoryview)):
                            res_list = [res_list]
                            
                    embeddings.extend(res_list)
                    success = True
                    break
                except Exception as e:
                    print(f"[HF Embedder] Attempt {attempt} failed: {e}")
                    if "loading" in str(e).lower() or "503" in str(e) or "555" in str(e):
                        wait_time = (attempt + 1) * 6
                        print(f"[HF Embedder] Model loading, waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        time.sleep(2)

            if not success:
                raise RuntimeError("Failed to generate embeddings from HuggingFace Inference API")

        if is_single:
            data = embeddings[0] if len(embeddings) > 0 else []
        else:
            data = embeddings

        class EmbedResult:
            def __init__(self, d):
                self.d = d
            def tolist(self):
                return self.d

        return EmbedResult(data)
