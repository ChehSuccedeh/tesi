
import ast
import re
import numpy as np

def npstr_to_str(match):
    # Prende il contenuto tra le parentesi di np.str_('...') o np.str_("...")
    s = match.group(2)
    return repr(s)

with open("packet_inspection/lime/results.txt", "r", encoding="utf-8") as f:
    risultati = [line.strip().split(":", 1)[-1].strip() for line in f if ":" in line]
all_words = []
all_values = []
for r in risultati:
    print("Original:", r)
    # Regex robusta che gestisce sia singoli che doppi apici
    content = re.sub(r"np\.str_\((['\"])(.*?)\1\)", npstr_to_str, r)
    try:
        data = ast.literal_eval(content)
        words = np.array([t[0] for t in data])
        values = np.array([t[1] for t in data], dtype=float)
        all_words.append(words)
        all_values.append(values)
        print(words)
        print(values)
    except Exception as e:
        print(f"Errore nel parsing: {e}")
    
with open("d:\\fabio\\tesi\\packet_inspection\\lime\\classes.txt", "r", encoding="utf-8") as f:
    content = f.read()
    classes = ast.literal_eval(content)

print(classes)
unique_classes = set(classes)
print("Unique classes:", unique_classes)

words_for_class = {}
for cls in unique_classes:
    words_for_class[cls] = []
for i,cls in enumerate(classes):
    for j, words in enumerate(all_words[i]):
        words_for_class[cls].extend((str(all_words[i][j]), float(all_values[i][j])))
    
print(words_for_class)

top_words_per_class = {}
for cls, words_vals in words_for_class.items():
    # words_vals is a flat list: [word1, value1, word2, value2, ...]
    words = words_vals[::2]
    values = words_vals[1::2]
    # Trova l'indice del valore massimo
    if values:
        max_idx = np.argmax(values)
        top_words_per_class[cls] = (words[max_idx], values[max_idx])
    else:
        top_words_per_class[cls] = None

import matplotlib.pyplot as plt

print("Parole con significato maggiore per classe:")
for cls, top in top_words_per_class.items():
    print(f"{cls}: {top}")

# Visualizzazione grafica
labels = []
words = []
values = []
for cls, top in top_words_per_class.items():
    if top is not None:
        labels.append(str(cls))
        words.append(top[0])
        values.append(top[1])

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, values, color='skyblue')
plt.ylabel("Valore")
plt.title("Parola con significato maggiore per classe")

# Annotazioni con la parola sopra ogni barra
for bar, word in zip(bars, words):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), word, ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=90)  # Etichette verticali

plt.tight_layout()
plt.show()

# Salva le 10 parole con significato maggiore per ogni classe in un file
top_10_words_per_class = {}
for cls, words_vals in words_for_class.items():
    words = np.array(words_vals[::2])
    values = np.array(words_vals[1::2], dtype=float)
    if len(values) > 0:
        top_indices = np.argsort(values)[-10:][::-1]
        top_10 = [(words[i], values[i]) for i in top_indices]
        top_10_words_per_class[cls] = top_10
    else:
        top_10_words_per_class[cls] = []

with open("d:\\fabio\\tesi\\packet_inspection\\lime\\top10_words_per_class.txt", "w", encoding="utf-8") as f:
    for cls, top_words in top_10_words_per_class.items():
        f.write(f"Classe: {cls}\n")
        for word, value in top_words:
            f.write(f"{word}: {value}\n")
        f.write("\n")