# Barras horizontales apiladas de ROI%: 1 barra por modelo/config,
# segmentos por año (2020..2024), ordenadas por ROI total desc.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
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

F_FECHA_DESDE = "2020-01-01"
F_FECHA_HASTA = "2024-12-31"
PATH_PRECIOS_TODOS = "DATA/AllStocksHistoricalData.csv"

# ========= MODELOS A INCLUIR =========
# Puedes añadir tantos como quieras (XGB, MLP, ARIMA...). El gráfico escala solo.
LOGS = OrderedDict({
    # --- XGBoost (ejemplos) ---
    "XGB W5H5 (baseline)": "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_5_5.csv",
    "XGB W5H10":           "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_5_10.csv",
    "XGB W5H15":           "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_5_15.csv",
    "XGB W10H5":           "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_10_5.csv",
    "XGB W10H10":          "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_10_10.csv",
    "XGB W10H15":          "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_10_15.csv",
    "XGB W15H5":           "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_15_5.csv",
    "XGB W15H10":          "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_15_10.csv",
    "XGB W15H15":          "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_15_15.csv",
    # --- MLP ---
    "MLP (32,)":           "Resultados/Trading_Log_AllStocks_MLP_OHLCV_32.csv",
    "MLP (64,)":           "Resultados/Trading_Log_AllStocks_MLP_OHLCV_64.csv",
    "MLP (64,32)":         "Resultados/Trading_Log_AllStocks_MLP_OHLCV_64_32.csv",
    "MLP (128,)":          "Resultados/Trading_Log_AllStocks_MLP_OHLCV_128.csv",
    "MLP (128,64)":        "Resultados/Trading_Log_AllStocks_MLP_OHLCV_128_64.csv",
    # --- ARIMA ---
    "ARIMA (1,1,0)":       "Resultados/Trading_Log_ARIMA_AllStocks.csv",
})

ANIOS = [2020, 2021, 2022, 2023, 2024]

# ========= Helpers =========
def build_timeline(df_prices_all: pd.DataFrame) -> pd.DatetimeIndex:
    tl = (df_prices_all[(df_prices_all["Fecha"] >= F_FECHA_DESDE) &
                        (df_prices_all["Fecha"] <= F_FECHA_HASTA)]
          .sort_values("Fecha")[["Fecha"]].drop_duplicates()
          .set_index("Fecha").index)
    return tl

def equity_from_log_profit(path_log: str, timeline: pd.DatetimeIndex) -> pd.Series:
    df = pd.read_csv(path_log, sep=";", parse_dates=["Fecha"])
    df = df[df["Symbol"].isin(SYMBOLS_CARTERA)].copy()
    sell = df[df["Accion"].astype(str).str.startswith("SELL", na=False)].copy()
    daily = (sell.groupby("Fecha", as_index=False)["Profit"].sum()
                  .sort_values("Fecha"))
    equity = pd.Series(CAPITAL_INICIAL, index=timeline, name="Equity")
    if len(daily):
        pnl = daily.set_index("Fecha")["Profit"].reindex(timeline).fillna(0.0)
        equity = (CAPITAL_INICIAL + pnl.cumsum()).rename("Equity")
    return equity

def roi_anual_por_equity(equity: pd.Series, year: int) -> float:
    """ROI% del año = (FinAño - IniAño) / IniAño * 100."""
    # subserie del año
    ymask = (equity.index.year == year)
    if not ymask.any():
        return 0.0
    eq_year = equity[ymask]
    # si faltan días al inicio de año, tomar el valor de equity del último día anterior disponible
    prev_idx = equity.index < pd.Timestamp(f"{year}-01-01")
    start = equity[prev_idx][-1] if prev_idx.any() else eq_year.iloc[0]
    end = eq_year.iloc[-1]
    if start == 0:
        return 0.0
    return (end - start) / start * 100.0

# ========= Cálculo de ROI por año y total =========
if not os.path.exists(PATH_PRECIOS_TODOS):
    raise FileNotFoundError(f"No existe dataset de precios: {PATH_PRECIOS_TODOS}")
df_prices_all = pd.read_csv(PATH_PRECIOS_TODOS, sep=";", parse_dates=["Fecha"])
timeline = build_timeline(df_prices_all)

rows = []
for label, ruta in LOGS.items():
    if not os.path.exists(ruta):
        print(f"[WARN] No existe: {ruta}")
        continue
    eq = equity_from_log_profit(ruta, timeline)
    rois = [roi_anual_por_equity(eq, y) for y in ANIOS]
    roi_total = (eq.iloc[-1] - eq.iloc[0]) / eq.iloc[0] * 100.0
    rows.append([label, *rois, roi_total])

cols = ["Modelo"] + [f"ROI_{y}%" for y in ANIOS] + ["ROI_Total%"]
df = pd.DataFrame(rows, columns=cols)

# Ordenar por ROI total desc
df = df.sort_values("ROI_Total%", ascending=True).reset_index(drop=True)

# ========= Plot (barras horizontales apiladas) =========
fig, ax = plt.subplots(figsize=(14, max(6, 0.45*len(df))))

# Paleta fija por año (colores distintos, consistentes en todas las barras)
year_colors = {
    2020: "#4c78a8",
    2021: "#f58518",
    2022: "#e45756",
    2023: "#72b7b2",
    2024: "#54a24b",
}

y_pos = np.arange(len(df))
left = np.zeros(len(df))

for y in ANIOS:
    vals = df[f"ROI_{y}%"].values
    ax.barh(y_pos, vals, left=left, height=0.6,
            label=str(y), color=year_colors[y], edgecolor="white")
    left += vals  # apilar ROI% (positivo/negativo funciona)

# Etiquetas de modelos
ax.set_yticks(y_pos)
ax.set_yticklabels(df["Modelo"].values)

ax.set_xlabel("ROI anual apilado (%)")
ax.set_title("Retorno de inversión ROI (%) por modelo y año")

# Formato eje X
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}%"))

# Líneas guía verticales cada 50%
ax.xaxis.set_major_locator(mticker.MultipleLocator(50))

# Etiqueta ROI total al final de cada barra
for i, total in enumerate(df["ROI_Total%"].values):
    xtext = left[i] if total >= 0 else left[i]  # después del último segmento
    ax.text(xtext + (3 if total>=0 else -3), i,
            f"{total:.0f}%",
            va="center", ha="left" if total>=0 else "right", fontsize=9)

ax.legend(title="Año", ncol=5, loc="lower right", frameon=True, fontsize=9)
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()

# Guardar figura lista para el TFM
os.makedirs("Resultados", exist_ok=True)
plt.savefig("Resultados/Figura_ROI_Stacked_Modelos_por_Año.png", dpi=220, bbox_inches="tight")
plt.show()

# (Opcional) Exportar tabla ordenada a CSV para la memoria
df.to_csv("Resultados/Tabla_ROI_Stacked_Modelos_por_Año.csv", index=False, sep=";")
