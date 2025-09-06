MODEL_PATH = "./code"

import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from tqdm import tqdm

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    truncation=True
)

with open("anomalous_packets.jsonl", "r") as f:
    lines = [json.loads(line) for line in f]

results = []
for line in tqdm(lines, desc="Classifying"):
    result = classifier(line["text"])
    results.append(result[0])

with open("classification_results.jsonl", "w") as out_f:
    for res in results:
        out_f.write(json.dumps(res) + "\n")

