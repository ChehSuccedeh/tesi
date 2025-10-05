import ast

lime_results = "d:\\fabio\\tesi\\packet_inspection\\lime\\lime_results_fixed.txt"
model_results = "d:\\fabio\\tesi\\packet_inspection\\lime\\extracted_labels.txt"

lime = []
with open(lime_results, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            lime.append(ast.literal_eval(line))

print(lime[0]["words"])

model = []
with open(model_results, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            model.append(ast.literal_eval(line))

print(model[0])

correct = 0
total = len(lime)

for i in range(total):
    lime_class = lime[i]["class"]
    model_label = model[i][0]["label"]
    if lime_class == model_label:
        correct += 1

accuracy = correct / total if total > 0 else 0
print(f"Accuracy: {accuracy:.4f}")

if correct < total:
    for i in range(len(lime)):
        lime_class = lime[i]["class"]
        model_label = model[i][0]["label"]
        if lime_class != model_label:
            words_dict = lime[i]["words"]
            if model_label in words_dict:
                # Ordina le tuple (parola, valore) in base al valore, decrescente
                sorted_words = sorted(words_dict[model_label], key=lambda x: x[1], reverse=True)
                top_5 = sorted_words[:5]
                print(f"Sample {i}: class in lime='{lime_class}', label in model='{model_label}' [NON COMBACIANO]")
                print("  Top 5 words for class '{}':".format(model_label))
                for word, value in top_5:
                    print(f"    {word}: {value}")
            else:
                print(f"Sample {i}: class in lime='{lime_class}', label in model='{model_label}' [NON COMBACIANO]")
                print(f"  No words found for class '{model_label}'")
        else:
            print(f"Sample {i}: class in lime='{lime_class}', label in model='{model_label}' [COMBACIANO]")
            words_dict = lime[i]["words"]
            if model_label in words_dict:
                sorted_words = sorted(words_dict[model_label], key=lambda x: x[1], reverse=True)
                top_5 = sorted_words[:5]
                print("  Top 5 words for class '{}':".format(model_label))
                for word, value in top_5:
                    print(f"    {word}: {value}")
            else:
                print(f"  No words found for class '{model_label}'")
            
output_file = "d:\\fabio\\tesi\\packet_inspection\\lime\\accuracy_results.txt"
with open(output_file, "w", encoding="utf-8") as out_f:
    out_f.write(f"Accuracy: {accuracy:.4f}\n\n")
    for i in range(len(lime)):
        lime_class = lime[i]["class"]
        model_label = model[i][0]["label"]
        if lime_class != model_label:
            words_dict = lime[i]["words"]
            out_f.write(f"Sample {i}: class in lime='{lime_class}', label in model='{model_label}' [NON COMBACIANO]\n")
            if model_label in words_dict:
                sorted_words = sorted(words_dict[model_label], key=lambda x: x[1], reverse=True)
                top_5 = sorted_words[:5]
                out_f.write(f"  Top 5 words for class '{model_label}':\n")
                for word, value in top_5:
                    out_f.write(f"    {word}: {value}\n")
            else:
                out_f.write(f"  No words found for class '{model_label}'\n")
        else:
            out_f.write(f"Sample {i}: class in lime='{lime_class}', label in model='{model_label}' [COMBACIANO]\n")
            words_dict = lime[i]["words"]
            if model_label in words_dict:
                sorted_words = sorted(words_dict[model_label], key=lambda x: x[1], reverse=True)
                top_5 = sorted_words[:5]
                out_f.write(f"  Top 5 words for class '{model_label}':\n")
                for word, value in top_5:
                    out_f.write(f"    {word}: {value}\n")
            else:
                out_f.write(f"  No words found for class '{model_label}'\n")