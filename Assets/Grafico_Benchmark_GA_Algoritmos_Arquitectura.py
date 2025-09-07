import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.15)

df = pd.read_csv("Resultados/Benchmark_GA_Tiempos_Relevantes.csv")

# Diccionario: clave = nombre de línea, valor = mostrar cifra en marcador (True/False)
config_to_show = {
    # "XGBoost CPU (8 hilos)":    False,
    # "XGBoost CPU (16 hilos)":   False,
    "XGBoost CPU (32 hilos)":   True,
    "XGBoost GPU":              True,
    # "MLP scikit-learn":         True,
    "MLP PyTorch GPU":          True,
}

config_map = {
    # "XGB_CPU_4": "XGBoost CPU (4 hilos)",
    # "XGB_CPU_8": "XGBoost CPU (8 hilos)",
    # "XGB_CPU_16": "XGBoost CPU (16 hilos)",
    "XGB_CPU_32": "XGBoost CPU (32 hilos)",
    "XGB_GPU": "XGBoost GPU",
    # "MLP_SKLEARN": "MLP scikit-learn",
    # "MLP_TORCH_CPU": "MLP PyTorch CPU",
    "MLP_TORCH_GPU": "MLP PyTorch GPU"
}
df['Configuración'] = df['Configuración'].map(config_map).fillna(df['Configuración'])

custom_palette = {
    # "XGBoost CPU (4 hilos)": "#2ca02c",
    # "XGBoost CPU (8 hilos)": "#1f77b4",
    # "XGBoost CPU (16 hilos)": "#ff7f0e",
    "XGBoost CPU (32 hilos)": "#d62728",
    "XGBoost GPU": "#3b993b",
    # "MLP scikit-learn": "#e377c2",
    # "MLP PyTorch CPU": "#8c564b",
    "MLP PyTorch GPU": "#17becf",
}

plt.figure(figsize=(11,7))

for config, show_label in config_to_show.items():
    color = custom_palette.get(config, "#333333")
    sub = df[df['Configuración'] == config].sort_values('Tamaño_dataset')
    plt.plot(sub['Tamaño_dataset'], sub['Tiempo_segundos'],
             marker='o', label=config, color=color, linewidth=2.2, markersize=9)
    if show_label:
        for x, y in zip(sub['Tamaño_dataset'], sub['Tiempo_segundos']):
            plt.text(x, y+0.03*max(df['Tiempo_segundos']), f"{y:.1f}", fontsize=10, color=color, ha='center')

plt.title('Tiempos de entrenamiento de una generación de GA (25 individuos)\npor modelo y configuración hardware', fontsize=17, fontweight='bold', pad=15)
plt.xlabel('Tamaño del dataset de entrenamiento (filas)', fontsize=14)
plt.ylabel('Tiempo total (segundos)', fontsize=14)
plt.xticks(sorted(df['Tamaño_dataset'].unique()), fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Configuración', fontsize=11, title_fontsize=13, loc='best', frameon=True)
plt.grid(axis='both', alpha=0.25, linestyle='--')
plt.tight_layout()
plt.gcf().set_facecolor('white')
plt.show()
