# -*- coding: utf-8 -*-
# ROI individual por símbolo y año (2020–2024)
# Barras horizontales apiladas (positivos a la derecha, negativos a la izquierda)
# Orden por suma de ROI positivos (longitud visible) y top arriba.

import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ========= Parámetros =========
CAPITAL_INICIAL_X_STOCK = 10_000.0
SYMBOLS_CARTERA = [
    'NVDA','AAPL','AMZN','LRCX','SBUX','REGN','KLAC','BKNG','AMD','VRTX',
    'MAR','CDNS','CAT','INTU','GILD','MU','EBAY','AXP','AMAT','COST','MSFT',
    'ORCL','ADI','MS','NKE'
]
ANIOS = [2020, 2021, 2022, 2023, 2024]
F_INI, F_FIN = "2020-01-01", "2024-12-31"

PATH_PRECIOS_TODOS = "DATA/AllStocksHistoricalData.csv"
PATH_LOG = "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_5_5.csv"  # XGB W5H5

# ========= Carga =========
if not os.path.exists(PATH_PRECIOS_TODOS):
    raise FileNotFoundError(f"No existe dataset de precios: {PATH_PRECIOS_TODOS}")
if not os.path.exists(PATH_LOG):
    raise FileNotFoundError(f"No existe log: {PATH_LOG}")

dfp = pd.read_csv(PATH_PRECIOS_TODOS, sep=";", parse_dates=["Fecha"])
timeline = (dfp[(dfp["Fecha"] >= F_INI) & (dfp["Fecha"] <= F_FIN)]
            .sort_values("Fecha")["Fecha"].drop_duplicates())

log = pd.read_csv(PATH_LOG, sep=";", parse_dates=["Fecha"])
log = log[log["Symbol"].isin(SYMBOLS_CARTERA)].copy()
log = log[log["Accion"].astype(str).str.startswith("SELL", na=False)].copy()
log = log[(log["Fecha"] >= F_INI) & (log["Fecha"] <= F_FIN)].copy()

# ========= Equity diario por símbolo =========
equities = {}
for sym in SYMBOLS_CARTERA:
    pnl_daily = (log[log["Symbol"] == sym]
                 .groupby("Fecha", as_index=False)["Profit"].sum()
                 .set_index("Fecha")["Profit"]
                 .reindex(timeline).fillna(0.0).cumsum())
    equity = CAPITAL_INICIAL_X_STOCK + pnl_daily
    equity.index.name = "Fecha"
    equities[sym] = equity.sort_index()

# ========= ROI anual por símbolo (sobre equity de inicio de año) =========
rows = []
for sym, eq in equities.items():
    for y in ANIOS:
        eq_y = eq[eq.index.year == y]
        if len(eq_y) == 0:
            roi_y = 0.0
        else:
            start = float(eq_y.iloc[0])
            end   = float(eq_y.iloc[-1])
            roi_y = ((end / start) - 1.0) * 100.0 if start != 0 else 0.0
        rows.append({"Symbol": sym, "Año": y, "ROI_%": roi_y})

df_roi = pd.DataFrame(rows)
pivot = df_roi.pivot(index="Symbol", columns="Año", values="ROI_%").fillna(0.0)

# ========= Orden por suma de positivos (coherente con longitud visible) =========
sum_pos = pivot.clip(lower=0).sum(axis=1)
order = sum_pos.sort_values(ascending=False).index  # descendente
pivot = pivot.loc[order]

symbols_plot = pivot.index.tolist()
ypos = np.arange(len(symbols_plot))

# ========= Plot =========
fig, ax = plt.subplots(figsize=(12, max(7, 0.42 * len(symbols_plot))))

year_colors = {2020:"#4c78a8", 2021:"#f58518", 2022:"#e45756", 2023:"#72b7b2", 2024:"#54a24b"}

base_pos = np.zeros(len(symbols_plot))
base_neg = np.zeros(len(symbols_plot))

# Barras apiladas: positivos a derecha, negativos a izquierda
for j, y in enumerate(ANIOS):
    vals = pivot[y].values
    pos = np.clip(vals, 0, None)
    neg = np.clip(vals, None, 0)

    ax.barh(ypos, pos, left=base_pos, color=year_colors[y],
            edgecolor="white", linewidth=0.6)
    base_pos += pos

    ax.barh(ypos, neg, left=base_neg, color=year_colors[y],
            edgecolor="white", linewidth=0.6)
    base_neg += neg

# Línea 0 %
ax.axvline(0, color="#666", linewidth=1.15, linestyle=":")

# Límites X
min_left  = float(base_neg.min())
max_right = float(base_pos.max())
margin = max(10.0, 0.05 * max(abs(min_left), abs(max_right)))
xmin, xmax = min_left - margin, max_right + margin
ax.set_xlim(xmin, xmax)

# ---- Ticks de rejilla
neg_floor = -5 * math.ceil(abs(xmin)/5.0)           # negativos cada -5
ticks_neg = np.arange(-5, neg_floor-1, -5)
pos_ceil  = 25 * math.ceil(max(0.0, xmax)/25.0)     # positivos cada 25
ticks_pos = np.arange(0, pos_ceil+1, 25)
ticks = np.unique(np.concatenate([ticks_neg, ticks_pos])).astype(float)
ax.set_xticks(ticks)

# ---- Formatter: negativos sin '%' (solo múltiplos de 10); 0 y positivos con '%'
def sparse_percent_formatter(v, pos):
    v_int = int(round(v))
    if v_int < 0:
        return f"{v_int:d}" if (abs(v_int) % 10 == 0) else ""
    return f"{v_int:d}%"
ax.xaxis.set_major_formatter(mticker.FuncFormatter(sparse_percent_formatter))

# Líneas verticales en los ticks (menos el 0)
for t in ticks:
    if t == 0:
        continue
    ax.axvline(t, color="#999", linestyle="--", linewidth=0.6, alpha=0.35)

# ======= Etiquetas (derecha = suma positivos; izquierda = suma negativos opcional) =======
sum_pos_vals = pivot.clip(lower=0).sum(axis=1).values
sum_neg_abs = -pivot.clip(upper=0).sum(axis=1).values
end_right = base_pos
end_left  = base_neg

for i in range(len(symbols_plot)):
    # Derecha (suma de positivos)
    ax.text(end_right[i] + 4, i, f"{sum_pos_vals[i]:.0f}%", va="center", ha="left",
            fontsize=9.2, fontweight="bold")
    # Izquierda (suma de negativos, si existe)
    if sum_neg_abs[i] > 0.1:
        ax.text(end_left[i] - 4, i, f"-{sum_neg_abs[i]:.0f}%", va="center", ha="right",
                fontsize=8.5, color="#666")

# Estética
ax.set_yticks(ypos)
ax.set_yticklabels(symbols_plot)
ax.invert_yaxis()  # 👉 mayor arriba

ax.set_xlabel("ROI individual anual por activo 2020–2024 (%)")
ax.set_title("ROI % individual por activo y año 2020-2024\nXGBoost Regressor, ventana 5 y horizonte 5")
ax.grid(axis="y", alpha=0.18)

# Leyenda manual por año
handles = [plt.Line2D([0],[0], color=year_colors[a], lw=8) for a in ANIOS]
ax.legend(handles, [str(a) for a in ANIOS], title="Año", loc="lower right", frameon=True)

plt.tight_layout()
plt.show()

# ========= Export opcional (comentado) =========
# os.makedirs("Resultados", exist_ok=True)
# plt.savefig("Resultados/Figura_ROI_Ind_Simbolo_Anios_XGB_W5H5_top_arriba.png",
#             dpi=220, bbox_inches="tight")
# out = pivot.copy()
# out["Sum_Pos_%"] = sum_pos_vals
# out["Sum_Neg_%"] = -sum_neg_abs
# out.reset_index().to_csv("Resultados/Tabla_ROI_Ind_Simbolo_Anios_XGB_W5H5_top_arriba.csv",
#                          sep=";", index=False)
