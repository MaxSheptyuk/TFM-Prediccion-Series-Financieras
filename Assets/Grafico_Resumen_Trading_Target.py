import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ----------------------------
# Config
# ----------------------------
CSV_PATH = os.path.join("Resultados", "Resumen_Metricas_Trading.csv")
FIG_OUT  = os.path.join("Resultados", "Relieve_Capital_WxH.png")

# ----------------------------
# Utilidades
# ----------------------------
def parse_target(t: str) -> tuple[int, int]:
    # Ej: TARGET_TREND_ANG_10_15 -> (W=10, H=15)
    parts = t.split("_")
    return int(parts[-2]), int(parts[-1])

def fmt_miles_punto(x, pos=None):
    # 123456.78 -> '123.457' (sin decimales, separador miles '.')
    try:
        return f"{x:,.0f}".replace(",", ".")
    except Exception:
        return str(x)

def bilinear_interpolate(W_vals, H_vals, Zmat, W_new, H_new):
    # Interp. bilineal en dos pasos sin SciPy
    Z_H = np.array([np.interp(H_new, H_vals, Zmat[i, :]) for i in range(len(W_vals))])
    Z_fine = np.array([np.interp(W_new, W_vals, Z_H[:, j]) for j in range(len(H_new))]).T
    return Z_fine  # shape (len(W_new), len(H_new))

# ----------------------------
# Carga y preparación
# ----------------------------
df = pd.read_csv(CSV_PATH, sep=";")

# Extraer W y H del nombre de Target
df[["W","H"]] = df["Target"].apply(lambda s: pd.Series(parse_target(s)))

# Rejilla original 3x3 y matriz de capital ganado
Z_pivot = df.pivot(index="W", columns="H", values="Capital_ganado").sort_index().sort_index(axis=1)
Ws = Z_pivot.index.to_numpy()
Hs = Z_pivot.columns.to_numpy()
Z  = Z_pivot.to_numpy()  # Capital ganado ($)

# Interpolación a malla fina
W_new = np.linspace(Ws.min(), Ws.max(), 120)
H_new = np.linspace(Hs.min(), Hs.max(), 120)
Z_smooth = bilinear_interpolate(Ws, Hs, Z, W_new, H_new)

# ----------------------------
# Gráfico: Relieve (contourf)
# ----------------------------
fig, ax = plt.subplots(figsize=(8.5, 6.5))

# Malla para contourf (X=H, Y=W)
X, Y = np.meshgrid(H_new, W_new)
levels = 15

cs = ax.contourf(X, Y, Z_smooth, levels=levels)
ax.contour(X, Y, Z_smooth, levels=10, linewidths=0.5, colors='k')

# Puntos reales y etiquetas abreviadas
ax.scatter(df["H"], df["W"], s=40, edgecolor="black", facecolor="white", zorder=3)
for _, r in df.iterrows():
    ax.text(r["H"]+0.15, r["W"]+0.10, f"W{int(r['W'])}H{int(r['H'])}", fontsize=9)

# Títulos y ejes
ax.set_title("Mapa de contornos interpolado: Capital ganado vs W (ventana) y H (horizonte) de predicción", fontsize=16, pad=18)
ax.set_xlabel("Horizonte H (días)", fontsize=12)
ax.set_ylabel("Ventana W (días)", fontsize=12)
ax.grid(True, alpha=0.25)

# Colorbar desplazado a la derecha y con miles '.'
# Ampliamos margen derecho para que no choque con el eje Y
plt.subplots_adjust(right=0.86)
cbar = fig.colorbar(cs, ax=ax, pad=0.08, fraction=0.05)
cbar.set_label("Capital ganado (USD, 2020–2024)")
cbar.formatter = FuncFormatter(fmt_miles_punto)
cbar.update_ticks()

plt.tight_layout()
#fig.savefig(FIG_OUT, dpi=200)
plt.show()

#print(f"[OK] Figura guardada en: {FIG_OUT}")
