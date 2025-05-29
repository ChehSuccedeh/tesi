import pandas as pd

with open("./data/test_dataset.jsonl", "r") as f:
    df = pd.read_json(f, lines=True)
    df = df.rename(columns={"code": "text", "language": "language"})
    df.sample(1000).to_json("./data/random_concept_dataset.jsonl", orient="records", lines=True)
    
    