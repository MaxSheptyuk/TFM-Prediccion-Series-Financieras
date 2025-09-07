# -*- coding: utf-8 -*-
"""
Grafico_Mejora_Tuning_SNS.py
De: Resultados/Comparativa_Tuning_2020_2024.csv (sep=';')
Hace un barplot ordenado por mejora %, con estilo profesional.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

CSV_IN   = "Resultados/Comparativa_Tuning_2020_2024.csv"   # columnas: Stock;RMSE_Base;RMSE_Tuned;Mejora_%;BaselineMode
PNG_OUT  = "Resultados/Grafico_Mejora_Tuning_SNS.png"

# Opción para resaltar algunos stocks clave (colores distintos y etiqueta bold)
HIGHLIGHT = {"NVDA", "ADI"}   # añade/quita a tu gusto

# --- Cargar y preparar ---
df = pd.read_csv(CSV_IN, sep=';')
df = df.sort_values("Mejora_%", ascending=False).reset_index(drop=True)

# Métricas para el título o pie
mean_gain = df["Mejora_%"].mean()
std_gain  = df["Mejora_%"].std()

# --- Estilo seaborn profesional ---
sns.set_theme(style="whitegrid", context="talk")  # 'talk' = tamaño agradable para docs
# Paletas: positivo vs negativo + resaltados
palette_pos = sns.color_palette("crest", n_colors=max(3, (df["Mejora_%"] > 0).sum()))
palette_neg = sns.color_palette("flare", n_colors=max(3, (df["Mejora_%"] <= 0).sum()))

colors = []
i_pos = i_neg = 0
for _, row in df.iterrows():
    stock = row["Stock"]
    if stock in HIGHLIGHT:
        colors.append(sns.color_palette("deep", 8)[3])  # un azul distinto para destacar
    elif row["Mejora_%"] >= 0:
        colors.append(palette_pos[i_pos % len(palette_pos)])
        i_pos += 1
    else:
        colors.append(palette_neg[i_neg % len(palette_neg)])
        i_neg += 1

# --- Plot ---
plt.figure(figsize=(14, 7))
ax = sns.barplot(
    data=df, x="Stock", y="Mejora_%", palette=colors, edgecolor="none"
)

# Línea base 0
ax.axhline(0, ls="--", lw=1, color="gray", alpha=0.8)

# Etiquetas encima de cada barra (1 decimal, con signo)
for p, val in zip(ax.patches, df["Mejora_%"].tolist()):
    h = p.get_height()
    ax.annotate(f"{val:.1f}%",
                (p.get_x() + p.get_width()/2, h),
                xytext=(0, 6 if h >= 0 else -16),
                textcoords="offset points",
                ha="center", va="bottom" if h>=0 else "top",
                fontsize=12, color="#222222")

# Resaltar etiquetas de stocks clave en negrita
xticklabels = []
for lab in ax.get_xticklabels():
    txt = lab.get_text()
    if txt in HIGHLIGHT:
        lab.set_fontweight("bold")
    xticklabels.append(txt)

# Títulos y ejes
ax.set_title(f"Mejora porcentual (RMSE) con tuning vs. baseline · Test 2020–2024\n"
             f"Media = {mean_gain:.2f}% ",
             pad=14)
ax.set_xlabel("Activo financiero")
ax.set_ylabel("Mejora % (↑ mejor)")

# Límites y layout
ymin = min(-3, df["Mejora_%"].min()*1.15)
ymax = df["Mejora_%"].max()*1.10
ax.set_ylim(ymin, ymax)
plt.xticks(rotation=45, ha="right")
sns.despine(left=True, bottom=False)
plt.tight_layout()

plt.show() 

# Guardar (Opcionalmente)
#os.makedirs(os.path.dirname(PNG_OUT), exist_ok=True)
#plt.savefig(PNG_OUT, dpi=200)
#print(f"Guardado: {PNG_OUT}")
