import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

# ---------------- CONFIG -----------------
CAPITAL_INICIAL = 10000.0    # por activo
SYMBOLS_CARTERA = [
    'NVDA','AAPL','AMZN','LRCX','SBUX','REGN','KLAC','BKNG','AMD','VRTX',
    'MAR','CDNS','CAT','INTU','GILD','MU','EBAY','AXP','AMAT','COST','MSFT',
    'ORCL','ADI','MS','NKE'
]

F_FECHA_DESDE = "2020-01-01"
F_FECHA_HASTA = "2024-12-31"

PATH_LOG_ML    = "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_15_5.csv"
PATH_LOG_ARIMA = "Resultados/Trading_Log_ARIMA_AllStocks.csv"
PATH_PRECIOS   = "DATA/AllStocksHistoricalData.csv"

# ---------------- LECTURA -----------------
trade_log_ml    = pd.read_csv(PATH_LOG_ML, sep=";", parse_dates=["Fecha"])
trade_log_arima = pd.read_csv(PATH_LOG_ARIMA, sep=";", parse_dates=["Fecha"])

df_prices_all = pd.read_csv(PATH_PRECIOS, sep=";", parse_dates=["Fecha"])

# Timeline global (todas las fechas de precios dentro del rango)
timeline = (
    df_prices_all[(df_prices_all["Fecha"] >= F_FECHA_DESDE) &
                  (df_prices_all["Fecha"] <= F_FECHA_HASTA)]
    .sort_values("Fecha")[["Fecha"]]
    .drop_duplicates()
    .set_index("Fecha")
)

# ---------------- FUNCIÓN CURVA -----------------
def curva_capital_global(trade_log, df_prices_all, symbols, f_ini, f_fin, capital_ini):
    curvas = []
    for sym in symbols:
        tl_sym = trade_log[trade_log["Symbol"] == sym].copy()
        tl_sym = tl_sym[tl_sym["Accion"].str.startswith("SELL")]

        precios_sym = df_prices_all[(df_prices_all["Symbol"] == sym) &
                                    (df_prices_all["Fecha"] >= f_ini) &
                                    (df_prices_all["Fecha"] <= f_fin)] \
                                    .sort_values("Fecha")[["Fecha"]]

        cap_curve = precios_sym.merge(
            tl_sym[["Fecha", "Capital_Actual"]],
            on="Fecha", how="left"
        )
        cap_curve["Capital_Actual"] = cap_curve["Capital_Actual"].ffill().fillna(capital_ini)
        cap_curve = cap_curve.set_index("Fecha")["Capital_Actual"]
        cap_curve = cap_curve.reindex(timeline.index).ffill().fillna(capital_ini)

        curvas.append(cap_curve.rename(sym))

    df_cartera = pd.concat(curvas, axis=1)
    df_cartera["Capital_Total"] = df_cartera.sum(axis=1)
    return df_cartera

# ---------------- CALCULAR CURVAS -----------------
df_cartera_ml = curva_capital_global(trade_log_ml, df_prices_all, SYMBOLS_CARTERA,
                                     F_FECHA_DESDE, F_FECHA_HASTA, CAPITAL_INICIAL)
df_cartera_arima = curva_capital_global(trade_log_arima, df_prices_all, SYMBOLS_CARTERA,
                                        F_FECHA_DESDE, F_FECHA_HASTA, CAPITAL_INICIAL)

# ---------------- PLOTEO -----------------
fig, ax = plt.subplots(figsize=(16,6))
ax.plot(df_cartera_ml.index, df_cartera_ml["Capital_Total"], linewidth=2,
        color="blue", label="XGBoost Regressor + GA – Cartera total (25 activos)")
ax.plot(df_cartera_arima.index, df_cartera_arima["Capital_Total"], linewidth=2,
        color="orange", linestyle="--", label="ARIMA – Cartera total (25 activos)")

ax.set_title("Evolución de Capital – Comparación Cartera Completa (25 activos)")
ax.set_ylabel("Capital ($)")
ax.set_xlim(pd.to_datetime(F_FECHA_DESDE), pd.to_datetime(F_FECHA_HASTA))
ax.grid(True, alpha=0.3)
ax.legend()

# Eje X cada 3 meses
quarter_loc = mdates.MonthLocator(interval=3)
date_fmt    = mdates.DateFormatter("%Y-%m")
ax.xaxis.set_major_locator(quarter_loc)
ax.xaxis.set_major_formatter(date_fmt)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# Formato eje Y con puntos de miles
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(round(x)):,}".replace(",", "."))
)

plt.tight_layout()
plt.show()

# (Opcional) Guardar
# plt.savefig("Resultados/Fig_Capital_Cartera_Comparacion.png", dpi=300, bbox_inches='tight')
