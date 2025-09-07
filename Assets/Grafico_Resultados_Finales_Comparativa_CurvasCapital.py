# -*- coding: utf-8 -*-
# Curva de capital comparativa (2020–2024) - 9 curvas
# Todas las líneas: sólidas, finas (1px), misma estética
# Diferencia solo por color de grupo

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib import cm
from collections import OrderedDict

# ========= Parámetros =========
CAPITAL_INICIAL_X_STOCK = 10_000.0
N_SIMBOLOS = 25
CAPITAL_INICIAL = CAPITAL_INICIAL_X_STOCK * N_SIMBOLOS
SYMBOLS_CARTERA = [
    'NVDA','AAPL','AMZN','LRCX','SBUX','REGN','KLAC','BKNG','AMD','VRTX',
    'MAR','CDNS','CAT','INTU','GILD','MU','EBAY','AXP','AMAT','COST','MSFT',
    'ORCL','ADI','MS','NKE'
]
F_FECHA_DESDE, F_FECHA_HASTA = "2020-01-01", "2024-12-31"
PATH_PRECIOS_TODOS = "DATA/AllStocksHistoricalData.csv"

# ========= Selección reducida =========
LOGS_XGB = OrderedDict({
    "XGB W5H5 (baseline)": "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_5_5.csv",
    "XGB W5H15":           "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_5_15.csv",
    "XGB W10H15":          "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_10_15.csv",
    "XGB W15H15":          "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_15_15.csv",
    "XGB W10H5":           "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_10_5.csv",
})
LOGS_MLP = OrderedDict({
    "MLP (128,)":   "Resultados/Trading_Log_AllStocks_MLP_OHLCV_128.csv",
    "MLP (64,32)":  "Resultados/Trading_Log_AllStocks_MLP_OHLCV_64_32.csv",
    "MLP (32,)":    "Resultados/Trading_Log_AllStocks_MLP_OHLCV_32.csv",
})
INCLUIR_ARIMA, PATH_ARIMA, LABEL_ARIMA = True, "Resultados/Trading_Log_ARIMA_AllStocks.csv", "ARIMA (1,1,0)"

# ========= Helpers =========
def build_timeline(df_prices_all: pd.DataFrame) -> pd.DatetimeIndex:
    return (df_prices_all[(df_prices_all["Fecha"] >= F_FECHA_DESDE) &
                          (df_prices_all["Fecha"] <= F_FECHA_HASTA)]
            .sort_values("Fecha")[["Fecha"]].drop_duplicates()
            .set_index("Fecha").index)

def equity_from_log_profit(path_log: str, timeline: pd.DatetimeIndex) -> pd.Series:
    df = pd.read_csv(path_log, sep=";", parse_dates=["Fecha"])
    df = df[df["Symbol"].isin(SYMBOLS_CARTERA)]
    sell = df[df["Accion"].astype(str).str.startswith("SELL", na=False)]
    daily = sell.groupby("Fecha", as_index=False)["Profit"].sum().sort_values("Fecha")
    equity = pd.Series(CAPITAL_INICIAL, index=timeline, name="Equity")
    if len(daily):
        pnl = daily.set_index("Fecha")["Profit"].reindex(timeline).fillna(0.0)
        equity = (CAPITAL_INICIAL + pnl.cumsum()).rename("Equity")
    return equity

def shades(cmap_name: str, n: int, start: float = 0.45, end: float = 0.9):
    cmap = cm.get_cmap(cmap_name)
    return [cmap(v) for v in np.linspace(start, end, n)]

# ========= Datos base =========
df_prices_all = pd.read_csv(PATH_PRECIOS_TODOS, sep=";", parse_dates=["Fecha"])
timeline = build_timeline(df_prices_all)

series_xgb, series_mlp, series_extra = OrderedDict(), OrderedDict(), OrderedDict()
for lbl, ruta in LOGS_XGB.items(): series_xgb[lbl] = equity_from_log_profit(ruta, timeline)
for lbl, ruta in LOGS_MLP.items(): series_mlp[lbl] = equity_from_log_profit(ruta, timeline)
if INCLUIR_ARIMA: series_extra[LABEL_ARIMA] = equity_from_log_profit(PATH_ARIMA, timeline)

# ========= Colores =========
colors_xgb = shades("Blues", len(series_xgb))
colors_mlp = shades("Oranges", len(series_mlp))
color_arima = "#7f0bd2"  # violeta oscuro


# ========= Plot =========
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_facecolor("white")

for i, (lbl, serie) in enumerate(series_xgb.items()):
    ax.plot(serie.index, serie.values, label=lbl, color=colors_xgb[i],
            linewidth=1.0, linestyle="-", alpha=0.9)
for i, (lbl, serie) in enumerate(series_mlp.items()):
    ax.plot(serie.index, serie.values, label=lbl, color=colors_mlp[i],
            linewidth=1.0, linestyle="-", alpha=0.9)
for lbl, serie in series_extra.items():
    ax.plot(serie.index, serie.values, label=lbl, color=color_arima,
            linewidth=1.0, linestyle="-", alpha=0.9)

ax.set_title("Evolución de capital de cartera (2020–2024)\n5 XGB (azules) + 3 MLP (naranjas) + ARIMA", pad=10)
ax.set_ylabel("Capital ($)")
ax.set_xlim(pd.to_datetime(F_FECHA_DESDE), pd.to_datetime(F_FECHA_HASTA))
ax.grid(True, alpha=0.3)

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(round(x)):,}".replace(",", ".")))

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, ncol=3, fontsize=9, frameon=True, loc="upper left")

plt.tight_layout()
plt.savefig("Resultados/Figura_Curva_Capital_Comparativa_TOP9_minimal.png", dpi=200)
plt.show()
