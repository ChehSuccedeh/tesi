import json

input_path = "d:\\fabio\\tesi\\packet_inspection\\sae\\results.txt"
output_path = "d:\\fabio\\tesi\\packet_inspection\\sae\\results_new.txt"

import re

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    content = infile.read()
    # Usa una regex per trovare tutti gli oggetti JSON (assumendo che siano {} separati)
    json_strings = re.findall(r'\{.*?\}(?=\{|$)', content)
    for js in json_strings:
        js = js.strip()
        if js:
            try:
                obj = json.loads(js)
                outfile.write(json.dumps(obj, ensure_ascii=False))
                outfile.write("\n")
            except json.JSONDecodeError:
                outfile.write(js + "\n")