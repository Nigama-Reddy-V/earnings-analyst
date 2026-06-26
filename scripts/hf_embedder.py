import os
import requests
import time

class HuggingFaceEmbedder:
    """
    Drop-in replacement for SentenceTransformer that uses Hugging Face's 
    free Inference API, using 0MB of local RAM.
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.hf_token = os.getenv("HF_TOKEN")
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"

    def encode(self, sentences, batch_size=32, show_progress_bar=False):
        is_single = isinstance(sentences, str)
        if is_single:
            sentences_list = [sentences]
        else:
            sentences_list = sentences

        headers = {}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        embeddings = []

        for i in range(0, len(sentences_list), batch_size):
            batch = sentences_list[i:i + batch_size]
            payload = {"inputs": batch}
            
            success = False
            for attempt in range(5):
                try:
                    response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
                    if response.status_code == 200:
                        res = response.json()
                        if isinstance(res, list):
                            embeddings.extend(res)
                            success = True
                            break
                    elif response.status_code == 503 or response.status_code == 555:
                        # Model is loading
                        wait_time = (attempt + 1) * 6
                        print(f"[HF Embedder] Model loading, waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"[HF Embedder] Error: {response.status_code} - {response.text}")
                        time.sleep(2)
                except Exception as e:
                    print(f"[HF Embedder] Exception: {e}")
                    time.sleep(2)

            if not success:
                raise RuntimeError("Failed to generate embeddings from HuggingFace Inference API")

        # Return mock that implements .tolist()
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
