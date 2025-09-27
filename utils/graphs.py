# %%
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CARTELLA = "code_classification"
CARTELLA = "packet_inspection"
# Parsing del file
def full_graphs(df, suffix):


    # Adatta i nomi delle colonne se necessario
    bottlenecks = df['bottleneck'].unique()
    n_bottlenecks = len(bottlenecks)

    for concept in df['concept'].unique():
        fig, axes = plt.subplots(2, 3, figsize=(18, 18), sharey=True)
        fig.suptitle(f'Boxplot per concept: {concept}', fontsize=16)
        subset = df[df['concept'] == concept]
        classi = sorted(df['class'].unique())
        # Palette da 10 colori
        palette = sns.color_palette('tab10',  n_colors=len(classi))
        for i, bottleneck in enumerate(bottlenecks):
            ax = axes[i // 3, i % 3]
            data_bottleneck = subset[subset['bottleneck'] == bottleneck]
            sns.boxplot(
            data=data_bottleneck,
            x='class',
            y='sensitivity',
            ax=ax,
            medianprops=dict(color='black'),
            palette=palette[:len(classi)]
            )
            ax.set_title(bottleneck)
            ax.set_xlabel('Classe')
            ax.set_ylabel('Valore TCAV')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)  # Etichette verticali
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'./{CARTELLA}/figures/boxplot_{concept}_{suffix}.png')
        # plt.show()
# %%
import matplotlib.pyplot as plt

def summary_graphs(df, suffix):
    for concept in df['concept'].unique():
        subset = df[df['concept'] == concept]
        # Calcola media, min e max per ogni bottleneck e classe
        agg_df = subset.groupby(['bottleneck', 'class'])['sensitivity'].agg(['mean', 'min', 'max']).reset_index()
        plt.figure(figsize=(18, 10))
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
        plt.savefig(f'./{CARTELLA}/figures/summary_{concept}_{suffix}.png')
        # plt.show()
# %%
if not os.path.exists(f'./{CARTELLA}/figures'):
    os.makedirs(f'./{CARTELLA}/figures')
# files = [f"./{CARTELLA}/results/tcav_auto_parsed.txt", f"./{CARTELLA}/results/tcav_fixed_parsed.txt", f"./{CARTELLA}/results/tcav_avg_parsed.txt"]
# files = [f"./{CARTELLA}/results/results_c_auto_parsed.json", f"./{CARTELLA}/results/results_c_fixed_parsed.json", f"./{CARTELLA}/results/results_c_avg_parsed.json"]
files = [f"./{CARTELLA}/results/results_p_auto_parsed.json", f"./{CARTELLA}/results/results_p_fixed_parsed.json", f"./{CARTELLA}/results/results_p_avg_parsed.json"]
suffix = ["auto", "fixed", "avg"]
dfs = []
for i in range(len(files)):
    with open(files[i], "r") as f:
        text = f.read()
    json_objects = json.loads(text)
    # data = [json.loads(obj) for obj in json_objects]
    data = json_objects

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
    classi = subset['class'].unique()
    tipi = subset['tipo'].unique() if 'tipo' in subset.columns else ['auto', 'fixed', 'avg']
    bottlenecks = sorted(df_all['bottleneck'].unique())
    n_classi = len(classi)
    n_cols = min(5, n_classi)
    n_rows = (n_classi + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), sharey=True)
    axes = axes.flatten() if n_classi > 1 else [axes]
    palette = sns.color_palette('Set1', n_colors=len(tipi))
    tipo_colors = {tipo: palette[i] for i, tipo in enumerate(tipi)}
    line_styles = ['-', '--', ':']
    markers = ['o', 's', 'D']
    for idx, class_name in enumerate(classi):
        ax = axes[idx]
        for j, tipo in enumerate(tipi):
            tipo_df = subset[(subset['class'] == class_name) & (subset['tipo'] == tipo)]
            mean_df = tipo_df.groupby('bottleneck')['sensitivity'].mean()
            mean_df = mean_df.reindex(bottlenecks)
            linestyle = line_styles[j % len(line_styles)]
            marker = markers[j % len(markers)]
            ax.plot(
                mean_df.index,
                mean_df.values,
                marker=marker,
                linestyle=linestyle,
                label=f'{tipo}',
                color=tipo_colors[tipo]
            )
        ax.set_title(f'Classe: {class_name}')
        ax.set_xlabel('Bottleneck')
        if idx % n_cols == 0:
            ax.set_ylabel('Valore medio TCAV')
        ax.legend(title='Tipo')
        ax.grid(True)
    for idx in range(len(classi), len(axes)):
        fig.delaxes(axes[idx])
    fig.suptitle(f'Confronto media TCAV per concept: {concept}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    plt.savefig(f'./{CARTELLA}/figures/confronto_medie_tcav_{concept}.png')
    # plt.show()
# %%
