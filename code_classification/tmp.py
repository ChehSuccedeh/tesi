import json

input_file = "./code_classification/packets.txt"
output_file = "output.jsonl"

with open(input_file, "r") as f, open(output_file, "w") as out:
    current_attack = None
    for line in f:
        line = line.strip()
        if line.startswith("-----") and line.endswith("-----"):
            # Nuova sezione di attacco
            current_attack = line.strip("- ").lower()
        elif line:
            # Entry del pacchetto
            out.write(json.dumps({
                "attack_type": current_attack,
                "packet": line
            }, ensure_ascii=False) + "\n")