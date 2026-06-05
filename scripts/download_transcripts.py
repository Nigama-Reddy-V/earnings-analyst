from datasets import load_dataset
import json
import os
import sys
sys.path.append(os.path.dirname(__file__))

print("Loading earnings call dataset...")
dataset = load_dataset("lamini/earnings-calls-qa", split="train")
print(f"Loaded {len(dataset)} examples")

# See what companies are in here
from collections import Counter
# Extract company hints from questions
print("\nSample dates:", set(list(d['date'] for d in dataset)[:20]))
print("\nTotal examples:", len(dataset))

# Save raw dataset for inspection
output_path = os.path.join(os.path.dirname(__file__), "../data/earnings_qa.json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save first 5000 examples
examples = []
for i, item in enumerate(dataset):
    if i >= 10000:
        break
    examples.append(item)

with open(output_path, 'w') as f:
    json.dump(examples, f)

print(f"\nSaved {len(examples)} examples to data/earnings_qa.json")