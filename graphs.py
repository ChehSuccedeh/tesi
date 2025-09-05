# %%
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Parsing del file
def full_graphs(df, suffix):


    # Adatta i nomi delle colonne se necessario
    bottlenecks = df['bottleneck'].unique()
    n_bottlenecks = len(bottlenecks)

    for concept in df['concept'].unique():
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
        fig.suptitle(f'Boxplot per concept: {concept}', fontsize=16)
        subset = df[df['concept'] == concept]
        # Palette di colori per i bottleneck
        palette = sns.color_palette('Set2', n_colors=n_bottlenecks)
        bottleneck_colors = {bn: palette[i] for i, bn in enumerate(bottlenecks)}
        for i, bottleneck in enumerate(bottlenecks):
            ax = axes[i // 3, i % 3]
            data_bottleneck = subset[subset['bottleneck'] == bottleneck]
            # Applica il colore specifico per il bottleneck
            sns.boxplot(
                data=data_bottleneck,
                x='class',
                y='sensitivity',
                ax=ax,
                boxprops=dict(facecolor=bottleneck_colors[bottleneck], color=bottleneck_colors[bottleneck]),
                medianprops=dict(color='black'),
                whiskerprops=dict(color=bottleneck_colors[bottleneck]),
                capprops=dict(color=bottleneck_colors[bottleneck]),
                flierprops=dict(markerfacecolor=bottleneck_colors[bottleneck], markeredgecolor=bottleneck_colors[bottleneck])
            )
            ax.set_title(bottleneck)
            ax.set_xlabel('Classe')
            ax.set_ylabel('Valore TCAV')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'./figures/boxplot_{concept}_{suffix}.png')
        # plt.show()
# %%
import matplotlib.pyplot as plt

def summary_graphs(df, suffix):
    for concept in df['concept'].unique():
        subset = df[df['concept'] == concept]
        # Calcola media, min e max per ogni bottleneck e classe
        agg_df = subset.groupby(['bottleneck', 'class'])['sensitivity'].agg(['mean', 'min', 'max']).reset_index()
        plt.figure(figsize=(10, 6))
        # Palette per le classi
        classi = agg_df['class'].unique()
        palette = sns.color_palette('tab10', n_colors=len(classi))
        class_colors = {cls: palette[i] for i, cls in enumerate(classi)}
        for class_name in classi:
            class_data = agg_df[agg_df['class'] == class_name]
            # Linea media
            plt.plot(
                class_data['bottleneck'],
                class_data['mean'],
                marker='o',
                label=class_name,
                color=class_colors[class_name]
            )
            # Linea min
            plt.plot(
                class_data['bottleneck'],
                class_data['min'],
                marker=None,
                linestyle='--',
                color=sns.set_hls_values(class_colors[class_name], l=0.6),
                alpha=0.7
            )
            # Linea max
            plt.plot(
                class_data['bottleneck'],
                class_data['max'],
                marker=None,
                linestyle='--',
                color=sns.set_hls_values(class_colors[class_name], l=0.6),
                alpha=0.7
            )
        plt.title(f'Andamento media/min/max TCAV per concept: {concept}')
        plt.xlabel('Bottleneck')
        plt.ylabel('Valore TCAV')
        plt.legend(title='Classe')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'./figures/summary_{concept}_{suffix}.png')
        # plt.show()
# %%

files = ["./results/tcav_auto_parsed.txt", "./results/tcav_fixed_parsed.txt", "./results/tcav_avg_parsed.txt"]
suffix = ["auto", "fixed", "avg"]
dfs = []
for i in range(len(files)):
    with open(files[i], "r") as f:
        text = f.read()
    json_objects = re.findall(r'\{.*?\}', text, re.DOTALL)
    data = [json.loads(obj) for obj in json_objects]

    df = pd.DataFrame(data)
    full_graphs(df, suffix[i])
    summary_graphs(df, suffix[i])
    df["tipo"] = suffix[i]
    dfs.append(df)
    
    
# %%

df_all = pd.concat(dfs, ignore_index=True)
# print(df_all['tipo'].value_counts())
# print(df_all['class'].value_counts())

# Per ogni concetto, grafico a linee: x=bottleneck, y=media sensibilità, una linea per ogni file per ogni classe (3*6 linee)
for concept in df_all['concept'].unique():
    subset = df_all[df_all['concept'] == concept]
    plt.figure(figsize=(12, 7))
    classi = subset['class'].unique()
    tipi = subset['tipo'].unique() if 'tipo' in subset.columns else ['auto', 'fixed', 'avg']
    bottlenecks = sorted(df_all['bottleneck'].unique())
    palette = sns.color_palette('tab10', n_colors=len(classi))
    # Stili di linea e marker per i tipi
    line_styles = ['-', '--', ':']
    markers = ['o', 's', 'D']
    tipo_style = {tipo: (line_styles[i % len(line_styles)], markers[i % len(markers)]) for i, tipo in enumerate(tipi)}
    for i, class_name in enumerate(classi):
        for j, tipo in enumerate(tipi):
            tipo_df = subset[(subset['class'] == class_name) & (subset['tipo'] == tipo)]
            mean_df = tipo_df.groupby('bottleneck')['sensitivity'].mean()
            mean_df = mean_df.reindex(bottlenecks)  # assicura tutti i bottleneck
            linestyle, marker = tipo_style[tipo]
            plt.plot(
                mean_df.index,
                mean_df.values,
                marker=marker,
                linestyle=linestyle,
                label=f'{tipo} - {class_name}',
                color=palette[i]
            )
    plt.title(f'Confronto media TCAV per concept: {concept}')
    plt.xlabel('Bottleneck')
    plt.ylabel('Valore medio TCAV')
    plt.legend(title='Tipo - Classe', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# %%
