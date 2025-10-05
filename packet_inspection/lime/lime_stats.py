import ast
from collections import defaultdict

input_path = "d:\\fabio\\tesi\\packet_inspection\\lime\\lime_results_fixed.txt"

dizionari = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            dizionari.append(ast.literal_eval(line))

# Accumula valori delle parole per ogni classe
class_word_values = defaultdict(lambda: defaultdict(list))

for d in dizionari:
    for cls, word_list in d['words'].items():
        for word, value in word_list:
            class_word_values[cls][word].append(value)

# print(class_word_values)
# Calcola la media e ordina
for cls, words in class_word_values.items():
    avg_values = {word: (sum(vals)/len(vals), len(vals)) for word, vals in words.items()}
    top_words = sorted(avg_values.items(), key=lambda x: x[1][0], reverse=True)[:10]
    print(f"Classe: {cls}")
    for word, (avg, count) in top_words:
        print(f"\t- {word}: {avg:.4f} (count: {count})")

output_path = "d:\\fabio\\tesi\\packet_inspection\\lime\\lime_stats_results.txt"
with open(output_path, "w", encoding="utf-8") as out_file:
    for cls, words in class_word_values.items():
        avg_values = {word: (sum(vals)/len(vals), len(vals)) for word, vals in words.items()}
        top_words = sorted(avg_values.items(), key=lambda x: x[1][0], reverse=True)[:10]
        out_file.write(f"Classe: {cls}\n")
        print(f"Classe: {cls}")
        for word, (avg, count) in top_words:
            out_file.write(f"\t- {word}: {avg:.4f} (count: {count})\n")
            print(f"\t- {word}: {avg:.4f} (count: {count})")

top_words_path = "d:\\fabio\\tesi\\packet_inspection\\lime\\lime_stats_top_words.txt"
with open(top_words_path, "w", encoding="utf-8") as top_file:
    for cls, words in class_word_values.items():
        # Trova i 10 valori massimi tra tutte le occorrenze delle parole
        all_word_values = []
        for word, vals in words.items():
            for v in vals:
                all_word_values.append((word, v))
        top_word_values = sorted(all_word_values, key=lambda x: x[1], reverse=True)[:10]
        top_file.write(f"Classe: {cls}\n")
        print(f"Classe: {cls}")
        for word, value in top_word_values:
            top_file.write(f"\t- {word}: {value:.4f}\n")
            print(f"\t- {word}: {value:.4f}")