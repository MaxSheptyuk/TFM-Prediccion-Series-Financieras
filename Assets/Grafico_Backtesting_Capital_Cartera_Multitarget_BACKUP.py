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

#  Rellena con tus logs. Clave = etiqueta en la leyenda, Valor = ruta CSV
LOGS = {
    "W5_H5":  "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_5_5.csv",
    "W5_H10": "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_5_10.csv",
    "W5_H15": "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_5_15.csv",
    "W10_H5": "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_10_5.csv",
    "W10_H10":"Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_10_10.csv",
    "W10_H15":"Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_10_15.csv",
    "W15_H5": "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_15_5.csv",
    "W15_H10":"Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_15_10.csv",
    "W15_H15":"Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_15_15.csv",
    # Ejemplos alternativos:
    # "XGB_ALLFEATS": "Resultados/Trading_Log_AllStocks_XGB_ALLFEATS.csv",
    # "ARIMA":        "Resultados/Trading_Log_ARIMA_AllStocks.csv",
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
    """
    A partir de un log unificado (todos los symbols) devuelve la curva
    de capital total de la cartera (suma de 25 activos) reindexada al timeline global.
    """
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

        # Reindexar al timeline global (evita huecos entre symbols)
        cap_curve = cap_curve.reindex(timeline).ffill().fillna(CAPITAL_INICIAL)
        curvas.append(cap_curve.rename(sym))

    df_cartera = pd.concat(curvas, axis=1)
    df_cartera["Capital_Total"] = df_cartera.sum(axis=1)
    return df_cartera["Capital_Total"]

# ===================== LECTURA BASE =====================
df_prices_all = pd.read_csv(PATH_PRECIOS_TODOS, sep=";", parse_dates=["Fecha"])
timeline = build_timeline(df_prices_all)

# ===================== CÁLCULO CURVAS =====================
series_por_log = {}  # etiqueta -> pd.Series (Capital_Total)
for etiqueta, ruta in LOGS.items():
    series_por_log[etiqueta] = curva_cartera_desde_log(ruta, df_prices_all, timeline)

# ===================== PLOT =====================
fig, ax = plt.subplots(figsize=(16, 6))

for etiqueta, serie in series_por_log.items():
    ax.plot(serie.index, serie.values, linewidth=1.7, label=etiqueta)

ax.set_title("Evolución de capital de cartera — Comparativa por TARGET_TREND_ANG (W,H)")
ax.set_ylabel("Capital ($)")
ax.set_xlim(pd.to_datetime(F_FECHA_DESDE), pd.to_datetime(F_FECHA_HASTA))
ax.grid(True, alpha=0.3)
ax.legend(ncol=3, frameon=True)

# Eje X: ticks trimestrales
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# Eje Y: miles con punto
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(round(x)):,}".replace(",", "."))
)

plt.tight_layout()
# Guardar si quieres
# plt.savefig("Resultados/Fig_Comparativa_Cartera_TARGET_TREND_ANG.png", dpi=300, bbox_inches="tight")
# plt.savefig("Resultados/Fig_Comparativa_Cartera_TARGET_TREND_ANG.pdf", bbox_inches="tight")
plt.show()
