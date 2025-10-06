import numpy as np
from collections import defaultdict
DIR = "./packet_inspection/shap/"
data = np.load(DIR + "results.npz", allow_pickle=True)


# Per vedere le chiavi disponibili
print(data.files)


array1 = data['words_0']
array2 = data['values_0']

print(array1)
print(array2)

# Raccogli tutte le (parola, valore) per ogni chiave tra tutti i dizionari
dizionari = []

for i in range(len(data.files) // 2):
    words = data[f'words_{i}'].item()
    values = data[f'values_{i}'].item()
    sample_dict = {}
    for key in words:
        word_list = words[key]
        value_list = values[key]
        for word, value in zip(word_list, value_list):
            sample_dict[word] = value
    dizionari.append(sample_dict)
print(dizionari[0])
class_word_values = defaultdict(lambda: defaultdict(list))

for d in dizionari:
    for cls, word_list in d.items():
        print(word_list)

        for word, value in word_list.items():
            class_word_values[cls][word].append(value)

# Per ogni chiave, trova i primi 10 valori massimi e le relative parole
with open(DIR +"top_words_per_key.txt", "w", encoding="utf-8") as f:
    for key in class_word_values:
        word_value_pairs = class_word_values[key]
        # Ordina per valore decrescente
        word_value_pairs.sort(key=lambda x: x[1], reverse=True)
        f.write(f"Key '{key}': Top 10 words with max values:\n")
        for word, value in word_value_pairs[:10]:
            f.write(f"  '{word}' (Value: {value})\n")
            
for i in range(len(data.files) // 2):
    words = data[f'words_{i}'].item()
    values = data[f'values_{i}'].item()
    for cls in words:
        word_list = words[cls]
        value_list = values[cls]
        for word, value in zip(word_list, value_list):
            class_word_values[cls][word].append(value)

# Calcola la media e ordina
for cls, words in class_word_values.items():
    avg_values = {word: (sum(vals)/len(vals), len(vals)) for word, vals in words.items()}
    top_words = sorted(avg_values.items(), key=lambda x: x[1][0], reverse=True)[:10]
    print(f"Classe: {cls}")
    for word, (avg, count) in top_words:
        print(f"\t- {word}: {avg:.4f} (count: {count})")

with open(DIR + "shap_stats_results.txt", "w", encoding="utf-8") as out_file:
    out_file.write("\nMedia dei valori delle parole per ogni classe:\n")
    for cls, words in class_word_values.items():
        total = 0
        count = 0
        for vals in words.values():
            total += sum(vals)
            count += len(vals)
        media = total / count if count > 0 else 0
        out_file.write(f"Classe: {cls} - Media valori parole: {media:.4f}\n")
    for cls, words in class_word_values.items():
        avg_values = {word: (sum(vals)/len(vals), len(vals)) for word, vals in words.items()}
        top_words = sorted(avg_values.items(), key=lambda x: x[1][0], reverse=True)[:10]
        out_file.write(f"Classe: {cls}\n")
        for word, (avg, count) in top_words:
            out_file.write(f"\t- {word}: {avg:.4f} (count: {count})\n")

with open(DIR + "shap_stats_top_words.txt", "w", encoding="utf-8") as top_file:
    for cls, words in class_word_values.items():
        all_word_values = []
        for i in range(len(data.files) // 2):
            w = data[f'words_{i}'].item()
            v = data[f'values_{i}'].item()
            if cls in w:
                for word, value in zip(w[cls], v[cls]):
                    all_word_values.append((word, value, i))
        top_word_values = sorted(all_word_values, key=lambda x: x[1], reverse=True)[:10]
        top_file.write(f"Classe: {cls}\n")
        for word, value, idx in top_word_values:
            top_file.write(f"\t- {word}: {value:.4f} (indice dizionario: {idx})\n")