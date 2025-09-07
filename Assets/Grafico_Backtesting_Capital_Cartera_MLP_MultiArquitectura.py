# Grafico_Backtesting_Capital_Cartera_Arquitecturas.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

# ===================== CONFIG =====================
CAPITAL_INICIAL = 10_000.0   # por activo
SYMBOLS_CARTERA = [
    'NVDA','AAPL','AMZN','LRCX','SBUX','REGN','KLAC','BKNG','AMD','VRTX',
    'MAR','CDNS','CAT','INTU','GILD','MU','EBAY','AXP','AMAT','COST','MSFT',
    'ORCL','ADI','MS','NKE'
]

F_FECHA_DESDE = "2020-01-01"
F_FECHA_HASTA = "2024-12-31"

PATH_PRECIOS_TODOS = "DATA/AllStocksHistoricalData.csv"

#  Rellena con tus logs por ARQUITECTURA
LOGS = {
    "MLP (64,)":        "Resultados/Trading_Log_AllStocks_MLP_OHLCV_64.csv",
    "MLP (64,32)":        "Resultados/Trading_Log_AllStocks_MLP_OHLCV_64_32.csv",
    "MLP (128)":     "Resultados/Trading_Log_AllStocks_MLP_OHLCV_128.csv",
    "MLP (128,64)":     "Resultados/Trading_Log_AllStocks_MLP_OHLCV_128_64.csv",
}

# ===================== UTILIDADES =====================
def build_timeline(df_prices_all: pd.DataFrame) -> pd.DatetimeIndex:
    timeline = (
        df_prices_all[(df_prices_all["Fecha"] >= F_FECHA_DESDE) &
                      (df_prices_all["Fecha"] <= F_FECHA_HASTA)]
        .sort_values("Fecha")[["Fecha"]]
        .drop_duplicates()
        .set_index("Fecha")
        .index
    )
    return timeline

def curva_cartera_desde_log(path_log: str,
                            df_prices_all: pd.DataFrame,
                            timeline: pd.DatetimeIndex) -> pd.Series:
    if not os.path.exists(path_log):
        raise FileNotFoundError(f"No existe: {path_log}")

    trade_log = pd.read_csv(path_log, sep=";", parse_dates=["Fecha"])
    trade_log = trade_log[trade_log["Symbol"].isin(SYMBOLS_CARTERA)].copy()

    curvas = []
    for sym in SYMBOLS_CARTERA:
        tl_sym = trade_log[trade_log["Symbol"] == sym].copy()
        tl_sym = tl_sym[tl_sym["Accion"].astype(str).str.startswith("SELL")]

        precios_sym = df_prices_all[
            (df_prices_all["Symbol"] == sym) &
            (df_prices_all["Fecha"] >= F_FECHA_DESDE) &
            (df_prices_all["Fecha"] <= F_FECHA_HASTA)
        ].sort_values("Fecha")[["Fecha"]]

        cap_curve = precios_sym.merge(
            tl_sym[["Fecha", "Capital_Actual"]],
            on="Fecha", how="left"
        )
        cap_curve["Capital_Actual"] = (
            cap_curve["Capital_Actual"]
            .ffill()
            .fillna(CAPITAL_INICIAL)
        )
        cap_curve = cap_curve.set_index("Fecha")["Capital_Actual"]

        cap_curve = cap_curve.reindex(timeline).ffill().fillna(CAPITAL_INICIAL)
        curvas.append(cap_curve.rename(sym))

    df_cartera = pd.concat(curvas, axis=1)
    df_cartera["Capital_Total"] = df_cartera.sum(axis=1)
    return df_cartera["Capital_Total"]

# ===================== LECTURA BASE =====================
df_prices_all = pd.read_csv(PATH_PRECIOS_TODOS, sep=";", parse_dates=["Fecha"])
timeline = build_timeline(df_prices_all)

# ===================== CÁLCULO CURVAS =====================
series_por_log = {}
for etiqueta, ruta in LOGS.items():
    series_por_log[etiqueta] = curva_cartera_desde_log(ruta, df_prices_all, timeline)

# ===================== PLOT =====================
fig, ax = plt.subplots(figsize=(16, 6))

for etiqueta, serie in series_por_log.items():
    ax.plot(serie.index, serie.values, linewidth=1.7, label=etiqueta)

ax.set_title("Evolución de capital de cartera — Comparativa por Arquitectura MLP")
ax.set_ylabel("Capital ($)")
ax.set_xlim(pd.to_datetime(F_FECHA_DESDE), pd.to_datetime(F_FECHA_HASTA))
ax.grid(True, alpha=0.3)
ax.legend(ncol=3, frameon=True)

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(round(x)):,}".replace(",", "."))
)

plt.tight_layout()
plt.show()
