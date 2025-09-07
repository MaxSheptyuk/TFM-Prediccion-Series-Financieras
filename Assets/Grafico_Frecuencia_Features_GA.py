import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap

# 1. Cargar el CSV
df = pd.read_csv("Resultados/GA_Frecuencia_Features.csv", sep=';')

# 2. Procesar la columna Selected_Features como lista
features_list = df['Selected_Features'].astype(str).apply(
    lambda x: [f.strip() for f in x.split(',') if f.strip()]
)
all_features = [feat for feats in features_list for feat in feats]

# 3. Contar frecuencia de cada feature individual
feature_counts = Counter(all_features)
df_feature_counts = pd.DataFrame.from_dict(feature_counts, orient='index', columns=['Frecuencia'])
df_feature_counts = df_feature_counts.sort_values('Frecuencia', ascending=False).head(20)

# 4. Recortar nombres largos
MAX_LEN = 18
df_feature_counts_short = df_feature_counts.copy()
df_feature_counts_short.index = df_feature_counts_short.index.map(
    lambda x: x if len(x) <= MAX_LEN else x[:MAX_LEN] + "…"
)

# 5. Colores (azul oscuro → azul medio, sin llegar a blanco)
norm = plt.Normalize(df_feature_counts_short['Frecuencia'].min(),
                     df_feature_counts_short['Frecuencia'].max())
blue_custom = ListedColormap(cm.Blues(np.linspace(0.3, 1, 256)))  # 0.3 = tono más oscuro que el blanco
colors = [blue_custom(norm(val)) for val in df_feature_counts_short['Frecuencia'][::-1]]

# 6. Gráfico
fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(
    df_feature_counts_short.index[::-1],
    df_feature_counts_short['Frecuencia'][::-1],
    color=colors, edgecolor="#222", linewidth=1.5, alpha=0.98
)

# Etiquetas
for bar in bars:
    ax.text(
        bar.get_width() + 0.5,
        bar.get_y() + bar.get_height()/2,
        f"{int(bar.get_width())}",
        va='center', fontsize=11, fontweight='bold'
    )

# Título centrado
ax.set_title("Top 20 indicadores individuales más seleccionados por el GA",
             fontsize=15, fontweight='bold', pad=15, loc='center')

# Ejes y estilo
ax.set_xlabel("Frecuencia de selección", fontsize=13)
ax.set_ylabel("Indicador", fontsize=13)
ax.grid(axis='x', linestyle='--', alpha=0.3)
ax.tick_params(axis='both', labelsize=11)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.show()
