# -*- coding: utf-8 -*-
# ROI medio (%) por periodo y modelo (2020–2024, escala logarítmica)
# - Barras agrupadas verticales (periodos como colores)
# - Leyenda dentro (arriba-derecha)
# - Barras más anchas + separadores verticales
# - Escala log/symlog automática
# - Ticks forzados (incluye 0,5 %) y formateo inteligente de porcentajes

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
from collections import OrderedDict

# ========= Parámetros base =========
CAPITAL_INICIAL_X_STOCK = 10_000.0
N_SIMBOLOS = 25
CAPITAL_INICIAL = CAPITAL_INICIAL_X_STOCK * N_SIMBOLOS

SYMBOLS_CARTERA = [
    'NVDA','AAPL','AMZN','LRCX','SBUX','REGN','KLAC','BKNG','AMD','VRTX',
    'MAR','CDNS','CAT','INTU','GILD','MU','EBAY','AXP','AMAT','COST','MSFT',
    'ORCL','ADI','MS','NKE'
]

F_INI = "2020-01-01"
F_FIN = "2024-12-31"
PATH_PRECIOS_TODOS = "DATA/AllStocksHistoricalData.csv"

# ========= Modelos a incluir =========
LOGS = OrderedDict({
    # XGB
    "XGB W5H15":  "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_5_15.csv",
    "XGB W15H15": "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_15_15.csv",
    "XGB W10H15": "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_10_15.csv",
    "XGB W5H5":   "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_5_5.csv",
    "XGB W10H5":  "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_10_5.csv",
    # MLP
    "MLP (128,)":  "Resultados/Trading_Log_AllStocks_MLP_OHLCV_128.csv",
    "MLP (64,32)": "Resultados/Trading_Log_AllStocks_MLP_OHLCV_64_32.csv",
    "MLP (32,)":   "Resultados/Trading_Log_AllStocks_MLP_OHLCV_32.csv",
    # ARIMA
    "ARIMA (1,1,0)": "Resultados/Trading_Log_ARIMA_AllStocks.csv",
})

# ========= Periodos y colores =========
PERIODOS = [
    ("Semanal",   "W"),
    ("Mensual",   "M"),
    ("Trimestral","Q"),
    ("Semestral", "2Q-DEC"),
    ("Anual",     "Y"),
]
COLORS_PERIOD = {
    "Semanal":    "#4c78a8",
    "Mensual":    "#f58518",
    "Trimestral": "#e45756",
    "Semestral":  "#72b7b2",
    "Anual":      "#54a24b",
}

# ========= Helpers =========
def build_timeline(df_prices_all: pd.DataFrame) -> pd.DatetimeIndex:
    tl = (df_prices_all[(df_prices_all["Fecha"] >= F_INI) &
                        (df_prices_all["Fecha"] <= F_FIN)]
          .sort_values("Fecha")[["Fecha"]].drop_duplicates()
          .set_index("Fecha").index)
    return tl

def equity_from_log_profit(path_log: str, timeline: pd.DatetimeIndex) -> pd.Series:
    """Equity diario de cartera (25 símbolos) usando PnL realizado en SELL."""
    df = pd.read_csv(path_log, sep=";", parse_dates=["Fecha"])
    df = df[df["Symbol"].isin(SYMBOLS_CARTERA)].copy()
    sell = df[df["Accion"].astype(str).str.startswith("SELL", na=False)].copy()
    daily = sell.groupby("Fecha", as_index=False)["Profit"].sum().sort_values("Fecha")
    equity = pd.Series(CAPITAL_INICIAL, index=timeline, name="Equity")
    if len(daily):
        pnl = daily.set_index("Fecha")["Profit"].reindex(timeline).fillna(0.0)
        equity = (CAPITAL_INICIAL + pnl.cumsum()).rename("Equity")
    return equity

def roi_periodic_series(equity: pd.Series, freq: str) -> pd.Series:
    """Retornos % por periodo freq: (cierre_periodo / apertura_periodo − 1) * 100."""
    close = equity.resample(freq).last()
    open_  = equity.resample(freq).first()
    mask_valid = (~open_.isna()) & (~close.isna()) & (open_ != 0)
    rois = (close[mask_valid] / open_[mask_valid] - 1.0) * 100.0
    return rois.dropna()

def roi_mean_by_periods(equity: pd.Series) -> dict:
    out = {}
    for nombre, freq in PERIODOS:
        r = roi_periodic_series(equity, freq)
        out[nombre] = float(r.mean()) if len(r) else 0.0
    return out

# ========= Cargar datos base =========
if not os.path.exists(PATH_PRECIOS_TODOS):
    raise FileNotFoundError(f"No existe dataset de precios: {PATH_PRECIOS_TODOS}")
df_prices_all = pd.read_csv(PATH_PRECIOS_TODOS, sep=";", parse_dates=["Fecha"])
timeline = build_timeline(df_prices_all)

# ========= Calcular ROI medio por periodo y modelo =========
rows = []
for modelo, ruta in LOGS.items():
    if not os.path.exists(ruta):
        print(f"[WARN] Falta: {ruta}")
        continue
    eq = equity_from_log_profit(ruta, timeline)
    rows.append({"Modelo": modelo, **roi_mean_by_periods(eq)})

df = pd.DataFrame(rows)

# Orden por ROI medio Mensual (ajusta si prefieres otra métrica)
if "Mensual" in df.columns:
    df = df.sort_values("Mensual", ascending=False).reset_index(drop=True)

# ========= Plot: Barras agrupadas (verticales) =========
labels = df["Modelo"].tolist()
period_names = [p[0] for p in PERIODOS if p[0] in df.columns]
num_models  = len(labels)
num_periods = len(period_names)

x = np.arange(num_models)            # centros de grupo
width = min(0.18, 0.88 / max(1, num_periods))  # barras más anchas

fig, ax = plt.subplots(figsize=(max(12, 0.9 * num_models), 6))

# Barras por periodo
for i, pname in enumerate(period_names):
    vals = df[pname].values
    offsets = x - ((num_periods - 1) * width / 2) + i * width
    ax.bar(offsets, vals, width,
           label=pname,
           color=COLORS_PERIOD.get(pname, None),
           edgecolor="white", linewidth=0.6)

# Etiquetas X
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right")

# Leyenda dentro (arriba-derecha)
leg = ax.legend(title="Periodo", loc="upper right",
                frameon=True, facecolor="white", edgecolor="gray", fontsize=9)
leg.get_frame().set_alpha(0.85)

ax.set_ylabel("ROI medio (%)")
ax.set_title("ROI medio (%) por periodo y modelo (2020–2024, escala logarítmica)")

# ===== Escala log/symlog automática =====
all_vals = df[period_names].values.flatten()
if np.all(all_vals > 0):
    ax.set_yscale("log")
else:
    ax.set_yscale("symlog", linthresh=0.1)

# ===== Ticks forzados (incluye 0,5%) + formateo inteligente =====
ax.set_yticks([0.1, 0.5, 1, 2, 5, 10, 20, 30])

def smart_percent(v, _):
    if v == 0:
        return "0%"
    if v < 1:              # 0.1%, 0.5%, etc.
        return f"{v:.1f}%"
    if v < 10:             # 1%..9%
        return f"{v:.0f}%"
    return f"{int(round(v))}%"

ax.yaxis.set_major_formatter(FuncFormatter(smart_percent))
ax.set_axisbelow(True)

# Separadores verticales entre grupos
for gx in np.arange(-0.5, num_models - 0.5, 1):
    ax.axvline(gx + 1, linestyle="--", color="#999999", alpha=0.15, linewidth=0.8)

# Grid horizontal (log/symlog)
ax.grid(axis="y", which="both", alpha=0.25, linestyle="--")

plt.tight_layout()
os.makedirs("Resultados", exist_ok=True)
plt.savefig("Resultados/Figura_Barras_Agrupadas_ROI_Medio_PorPeriodo_tuned.png",
            dpi=220, bbox_inches="tight")
plt.show()

# (Opcional) Exportar tabla
df.to_csv("Resultados/Tabla_ROI_Medio_PorPeriodo.csv", sep=";", index=False)
