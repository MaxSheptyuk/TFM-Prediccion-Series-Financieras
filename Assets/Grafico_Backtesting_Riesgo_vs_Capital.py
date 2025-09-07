import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Carga de métricas ===
df = pd.read_csv("Resultados/Resumen_Metricas_Trading.csv", sep=";")

# Parsear W y H del target
def parse_wh(s):
    s = str(s)
    m = re.search(r"[Ww](\d+)[Hh](\d+)", s)
    if m: return int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+)\D+(\d+)$", s)
    if m: return int(m.group(1)), int(m.group(2))
    return np.nan, np.nan

df["W"], df["H"] = zip(*df["Target"].map(parse_wh))
df["DDpct"] = df["Max_Drawdown_%"].astype(float)

# === Barras agrupadas por H ===
Ws = sorted(df["W"].unique())
Hs = sorted(df["H"].unique())
fig, ax = plt.subplots(figsize=(10,5))

bar_width = 0.22
x = np.arange(len(Hs))

for k, w in enumerate(Ws):
    vals = []
    for h in Hs:
        row = df[(df["W"]==w) & (df["H"]==h)]
        vals.append(row["DDpct"].iloc[0] if not row.empty else np.nan)
    ax.bar(x + k*bar_width, vals, width=bar_width, label=f"W{w}")
    # etiquetas encima
    for xi, yi in zip(x + k*bar_width, vals):
        if not np.isnan(yi):
            ax.text(xi, yi + 0.2, f"{yi:.1f}%", ha="center", va="bottom", fontsize=9)

# Ajustar ticks
ax.set_xticks(x + bar_width*(len(Ws)-1)/2)
ax.set_xticklabels(Hs)
ax.set_xlabel("Horizonte H (días hacia el futuro)")
ax.set_ylabel("Max Drawdown (%)")
ax.set_title("Riesgo (Max Drawdown %) — Barras agrupadas por Horizonte")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(title="Ventana W")

# === Líneas verticales divisorias entre grupos ===
for i in range(len(Hs)-1):
    xpos = (x[i] + bar_width*len(Ws))  # borde derecho del grupo
    ax.axvline(x=xpos, color="gray", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
