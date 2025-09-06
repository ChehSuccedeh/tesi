import json
import re

def parse_json_lines(file_path):
    parsed_data = []

    # Espressione regolare per trovare il primo JSON in una riga
    with open(file_path, "r") as f:
        text = f.read()
        
    # print(text)

    json_objects = re.findall(r'\{.*?\}', text, re.DOTALL)
    # print(json_objects)
    results = [json.loads(obj) for obj in json_objects]

    return results

# Esempio di utilizzo
file_paths = ["./packet_inspection/results/tcav_auto", "./packet_inspection/results/tcav_fixed", "./packet_inspection/results/tcav_avg"]
for f in file_paths:
    dizionario = parse_json_lines(f+".txt")
    
    fo = open(f+"_parsed.txt", "w")
    for x in dizionario:
        print(x)
        fo.write(json.dumps(x)+"\n")
        
    fo.close()