import ast
from collections import defaultdict
DIR = "./packet_inspection/lime/"
input_path = DIR + "lime_results_fixed.txt"

dizionari = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            dizionari.append(ast.literal_eval(line))
# print(dizionari[0])
# Accumula valori delle parole per ogni classe
class_word_values = defaultdict(lambda: defaultdict(list))

for d in dizionari:
    for cls, word_list in d['words'].items():
        for word, value in word_list:
            class_word_values[cls][word].append(value)

# class_word_values: dict[class][word] = [valori]

# Calcola la media e ordina
for cls, words in class_word_values.items():
    avg_values = {word: (sum(vals)/len(vals), len(vals)) for word, vals in words.items()}
    top_words = sorted(avg_values.items(), key=lambda x: x[1][0], reverse=True)[:10]
    print(f"Classe: {cls}")
    for word, (avg, count) in top_words:
        print(f"\t- {word}: {avg:.4f} (count: {count})")

output_path = DIR + "lime_stats_results.txt"
with open(output_path, "w", encoding="utf-8") as out_file:
        # Calcola e stampa la media dei valori delle parole per ogni classe
    print("\nMedia dei valori delle parole per ogni classe:")
    out_file.write("\nMedia dei valori delle parole per ogni classe:\n")
    for cls, words in class_word_values.items():
        total = 0
        count = 0
        for vals in words.values():
            total += sum(vals)
            count += len(vals)
            
        print(f"Total: {total}, Count: {count}")
        media = total / count if count > 0 else 0
        print(f"Classe: {cls} - Media valori parole: {media:.4f}")
        out_file.write(f"Classe: {cls} - Media valori parole: {media:.4f}\n")
        

    for cls, words in class_word_values.items():
        avg_values = {word: (sum(vals)/len(vals), len(vals)) for word, vals in words.items()}
        top_words = sorted(avg_values.items(), key=lambda x: x[1][0], reverse=True)[:10]
        out_file.write(f"Classe: {cls}\n")
        print(f"Classe: {cls}")
        for word, (avg, count) in top_words:
            out_file.write(f"\t- {word}: {avg:.4f} (count: {count})\n")
            print(f"\t- {word}: {avg:.4f} (count: {count})")

top_words_path = DIR + "lime_stats_top_words.txt"
with open(top_words_path, "w", encoding="utf-8") as top_file:
    for cls, words in class_word_values.items():
        # Trova i 10 valori massimi tra tutte le occorrenze delle parole, includendo l'indice originale
        all_word_values = []
        for idx, d in enumerate(dizionari):
            if cls in d['words']:
                for word, value in d['words'][cls]:
                    all_word_values.append((word, value, idx))
        top_word_values = sorted(all_word_values, key=lambda x: x[1], reverse=True)[:10]
        top_file.write(f"Classe: {cls}\n")
        print(f"Classe: {cls}")
        for word, value, idx in top_word_values:
            top_file.write(f"\t- {word}: {value:.4f} (indice dizionario: {idx})\n")
            print(f"\t- {word}: {value:.4f} (indice dizionario: {idx})")
            
