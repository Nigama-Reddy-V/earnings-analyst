import os
import torch
from dotenv import load_dotenv

load_dotenv()

FINETUNED_MODEL_ID = "Nigama-11/mistral-7b-earnings-analyst"


def load_finetuned_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model_id = "unsloth/mistral-7b-bnb-4bit"
    adapter_model_id = "Nigama-11/mistral-7b-earnings-analyst"

    print("Loading base model on CPU (no GPU on Mac)...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="cpu",
        dtype=torch.float32
    )

    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(model, adapter_model_id)
    model.eval()

    print("Finetuned model loaded successfully.")
    return model, tokenizer


def generate_analysis(
    model,
    tokenizer,
    question: str,
    context: str,
    max_new_tokens: int = 400
) -> str:
    """
    Generate financial analysis using the finetuned model.
    This is what the Analyzer node in LangGraph will call.
    """
    prompt = f"""### System:
You are an expert financial analyst. Analyze the provided earnings information and answer questions accurately. Use only information present in the context. Express numerical findings precisely and use proper financial terminology.

### Context:
{context}

### Question:
{question}

### Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


if __name__ == "__main__":
    # Quick test to verify model loads and runs
    model, tokenizer = load_finetuned_model()

    test_response = generate_analysis(
        model,
        tokenizer,
        question="Did the company beat earnings expectations?",
        context="Microsoft reported EPS of $3.12 versus analyst consensus of $2.94. Revenue came in at $64.7 billion against expectations of $60.9 billion."
    )

    print("\n=== Test Response ===")
    print(test_response)