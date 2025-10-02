import numpy as np
from collections import defaultdict

data = np.load("d:\\fabio\\tesi\\packet_inspection\\shap\\results.npz", allow_pickle=True)

# Per vedere le chiavi disponibili
print(data.files)


array1 = data['words_0']
array2 = data['values_0']

print(array1)
print(array2)

# Raccogli tutte le parole e i valori per ogni chiave tra tutti i dizionari
all_words = defaultdict(list)
all_values = defaultdict(list)

for i in range(len(data.files) // 2):
    words = data[f'words_{i}'].item()
    values = data[f'values_{i}'].item()
    for key in words:
        word_list = words[key]
        value_list = values[key]
        all_words[key].extend(word_list)
        all_values[key].extend(value_list)

# Per ogni chiave, trova i primi 10 valori massimi e le relative parole
with open("top_words_per_key.txt", "w", encoding="utf-8") as f:
    for key in all_words:
        word_value_pairs = list(zip(all_words[key], all_values[key]))
        # Ordina per valore decrescente
        word_value_pairs.sort(key=lambda x: x[1], reverse=True)
        f.write(f"Key '{key}': Top 10 words with max values:\n")
        for word, value in word_value_pairs[:10]:
            f.write(f"  '{word}' (Value: {value})\n")