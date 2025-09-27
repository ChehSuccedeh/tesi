import json
import re

def parse_json_lines(file_path):
    parsed_data = []

    # Espressione regolare per trovare il primo JSON in una riga
    with open(file_path, "r") as f:
        text = f.read()
        
    # print(text)

    json_objects = re.findall(r'\{.*?\}', text, re.DOTALL)
    print(json_objects[0])
    results = []
    for obj in json_objects:
        # print(obj)
        results.append(json.loads(obj))
    return results

# Esempio di utilizzo
# file_paths = ["./packet_inspection/results/tcav_auto", "./packet_inspection/results/tcav_fixed", "./packet_inspection/results/tcav_avg"]
file_paths = ["./results_c_auto", "./results_c_fixed", "./results_c_avg", "./results_p_auto", "./results_p_fixed", "./results_p_avg"]
for f in file_paths:
    dizionario = parse_json_lines(f+".txt")

    fo = open(f+"_parsed.json", "w")
    fo.write(json.dumps(dizionario))
        
    fo.close()