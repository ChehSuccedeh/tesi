import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_path = "d:\\fabio\\tesi\\packet_inspection\\sae\\results_new.txt"

# Carica i dati
data = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            data.append(obj)

# Estrai tutti i layer
layers = set(d['layer'] for d in data)
betas = sorted(set(d['beta'] for d in data))
lrs = sorted(set(d['lr'] for d in data))

for layer in layers:
    # Filtra i dati per layer
    layer_data = [d for d in data if d['layer'] == layer]
    # Crea una matrice vuota
    matrix = pd.DataFrame(index=betas, columns=lrs)
    for d in layer_data:
        matrix.loc[d['beta'], d['lr']] = d['avg_loss']
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix.astype(float), annot=True, fmt=".4f", cmap="viridis")
    plt.title(f"Layer {layer} - avg_loss")
    plt.xlabel("lr")
    plt.ylabel("beta")
    plt.tight_layout()
    plt.savefig(f"./packet_inspection/sae/figures/layer_{layer}_avg_loss.png")
    plt.close()

print("Grafici generati per ogni layer (file PNG).")