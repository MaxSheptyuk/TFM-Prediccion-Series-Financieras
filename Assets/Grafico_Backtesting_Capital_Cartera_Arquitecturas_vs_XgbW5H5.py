# Grafico_Backtesting_Capital_Cartera_Arquitecturas_Highlight.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

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

LOGS_BASE = {
    "MLP (64,)":    "Resultados/Trading_Log_AllStocks_MLP_OHLCV_64.csv",
    "MLP (64,32)":  "Resultados/Trading_Log_AllStocks_MLP_OHLCV_64_32.csv",
    "MLP (128)":    "Resultados/Trading_Log_AllStocks_MLP_OHLCV_128.csv",
    "MLP (128,64)": "Resultados/Trading_Log_AllStocks_MLP_OHLCV_128_64.csv",
}

# NUESTRA CURVA DESTACADA (W=5, H=5)
HIGHLIGHT_LABEL = "XGBoost Regressor (W=5, H=5)"
HIGHLIGHT_PATH  = "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_5_5.csv"

def build_timeline(df_prices_all: pd.DataFrame) -> pd.DatetimeIndex:
    tl = (df_prices_all[(df_prices_all["Fecha"] >= F_FECHA_DESDE) &
                        (df_prices_all["Fecha"] <= F_FECHA_HASTA)]
          .sort_values("Fecha")[["Fecha"]].drop_duplicates()
          .set_index("Fecha").index)
    return tl

# === NUEVO: equity por cumsum de Profit diario (coincide con la tabla) ===
def equity_from_log_profit(path_log: str, timeline: pd.DatetimeIndex) -> pd.Series:
    if not os.path.exists(path_log):
        raise FileNotFoundError(f"No existe: {path_log}")
    df = pd.read_csv(path_log, sep=";", parse_dates=["Fecha"])
    # Filtra símbolos de la cartera (por coherencia)
    df = df[df["Symbol"].isin(SYMBOLS_CARTERA)].copy()
    # Solo SELL para PnL realizado
    sell = df[df["Accion"].astype(str).str.startswith("SELL", na=False)].copy()

    # PnL diario de la CARPETA (suma sobre símbolos)
    daily = (sell.groupby("Fecha", as_index=False)["Profit"].sum()
                  .sort_values("Fecha"))

    equity = pd.Series(CAPITAL_INICIAL, index=timeline, name="Equity")
    if len(daily):
        pnl = daily.set_index("Fecha")["Profit"].reindex(timeline).fillna(0.0)
        equity = (CAPITAL_INICIAL + pnl.cumsum()).rename("Equity")
    return equity

# ===================== LECTURA BASE =====================
df_prices_all = pd.read_csv(PATH_PRECIOS_TODOS, sep=";", parse_dates=["Fecha"])
timeline = build_timeline(df_prices_all)

# ===================== CÁLCULO CURVAS =====================
series_por_log = {}
for etiqueta, ruta in LOGS_BASE.items():
    series_por_log[etiqueta] = equity_from_log_profit(ruta, timeline)

serie_highlight = equity_from_log_profit(HIGHLIGHT_PATH, timeline)

# ===================== PLOT =====================
fig, ax = plt.subplots(figsize=(16, 6))

soft_kwargs = dict(linewidth=1.7, alpha=0.5, zorder=1)
highlight_kwargs = dict(linewidth=2.8, alpha=0.95, zorder=5)

for etiqueta, serie in series_por_log.items():
    ax.plot(serie.index, serie.values, label=etiqueta, **soft_kwargs)

ax.plot(serie_highlight.index, serie_highlight.values,
        label=HIGHLIGHT_LABEL, color="#170DE1", **highlight_kwargs)

ax.set_title("Evolución de capital de cartera — MLP vs XGB (W5H5)")
ax.set_ylabel("Capital ($)")
ax.set_xlim(pd.to_datetime(F_FECHA_DESDE), pd.to_datetime(F_FECHA_HASTA))
ax.grid(True, alpha=0.3)

handles, labels = ax.get_legend_handles_labels()
order = [labels.index(HIGHLIGHT_LABEL)] + [i for i,l in enumerate(labels) if l != HIGHLIGHT_LABEL]
ax.legend([handles[i] for i in order], [labels[i] for i in order], ncol=3, frameon=True)

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(round(x)):,}".replace(",", ".")))

plt.tight_layout()
plt.show()
