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

#PATH_LOG_UNIFICADO = "Resultados/Trading_Log_AllStocks.csv"
PATH_LOG_UNIFICADO = "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_15_5.csv"
#PATH_LOG_UNIFICADO = "Resultados/Trading_Log_AllStocks_XGB_ALLFEATS.csv"
#PATH_LOG_UNIFICADO = "Resultados/Trading_Log_AllStocks_MLP_OHLCV.csv"
#PATH_LOG_UNIFICADO = "Resultados/Trading_Log_AllStocks_MLP_OHLCV_128_64.csv"
#PATH_LOG_UNIFICADO = "Resultados/Trading_Log_AllStocks_MLP_OHLCV_64.csv"
#PATH_LOG_UNIFICADO = "Resultados/Trading_Log_AllStocks_MLP_OHLCV_64_32.csv"
#PATH_LOG_UNIFICADO = "Resultados/Trading_Log_AllStocks_MLP_OHLCV_32.csv"
#PATH_LOG_UNIFICADO = "Resultados/Trading_Log_AllStocks_MLP_OHLCV_128.csv"




PATH_PRECIOS_TODOS = "DATA/AllStocksHistoricalData.csv"

# ---------------- LECTURA -----------------
trade_log = pd.read_csv(PATH_LOG_UNIFICADO, sep=";", parse_dates=["Fecha"])
trade_log = trade_log[trade_log["Symbol"].isin(SYMBOLS_CARTERA)].copy()

# Timeline global (todas las fechas de precios dentro del rango)
df_prices_all = pd.read_csv(PATH_PRECIOS_TODOS, sep=";", parse_dates=["Fecha"])
timeline = (
    df_prices_all[(df_prices_all["Fecha"] >= F_FECHA_DESDE) &
                  (df_prices_all["Fecha"] <= F_FECHA_HASTA)]
    .sort_values("Fecha")[["Fecha"]]
    .drop_duplicates()
    .set_index("Fecha")
)

# ---------------- CURVAS INDIVIDUALES -----------------
curvas = []

for sym in SYMBOLS_CARTERA:
    # Eventos de capital: solo cierres SELL_* (donde cambia el capital)
    tl_sym = trade_log[trade_log["Symbol"] == sym].copy()
    tl_sym = tl_sym[tl_sym["Accion"].str.startswith("SELL")]

    # Fechas de precios de ese símbolo
    precios_sym = df_prices_all[(df_prices_all["Symbol"] == sym) &
                                (df_prices_all["Fecha"] >= F_FECHA_DESDE) &
                                (df_prices_all["Fecha"] <= F_FECHA_HASTA)] \
                                .sort_values("Fecha")[["Fecha"]]

    # Merge precios + eventos de capital y rellenar
    cap_curve = precios_sym.merge(
        tl_sym[["Fecha", "Capital_Actual"]],
        on="Fecha", how="left"
    )
    cap_curve["Capital_Actual"] = cap_curve["Capital_Actual"].ffill().fillna(CAPITAL_INICIAL)
    cap_curve = cap_curve.set_index("Fecha")["Capital_Actual"]

    # Reindexar al timeline global para evitar NaN y arrancar en 10.000
    cap_curve = cap_curve.reindex(timeline.index).ffill().fillna(CAPITAL_INICIAL)

    curvas.append(cap_curve.rename(sym))

# ---------------- SUMA DE CARTERA -----------------
df_cartera = pd.concat(curvas, axis=1)
df_cartera["Capital_Total"] = df_cartera.sum(axis=1)   # debería iniciar en 25 * 10.000 = 250.000

# ---------------- PLOTEO -----------------
fig, ax = plt.subplots(figsize=(16,6))
ax.plot(df_cartera.index, df_cartera["Capital_Total"], linewidth=2,
        color="blue", label="Cartera total (25 activos)")

ax.set_title("Curva de Evolución de Capital – Cartera completa (25 activos)")
ax.set_ylabel("Capital ($)")
ax.set_xlim(pd.to_datetime(F_FECHA_DESDE), pd.to_datetime(F_FECHA_HASTA))
ax.grid(True, alpha=0.3)
ax.legend()

# --- Eje X: ticks cada 3 meses ---
quarter_loc = mdates.MonthLocator(interval=3)   # 3 meses
date_fmt    = mdates.DateFormatter("%Y-%m")     # etiqueta compacta AAAA-MM
ax.xaxis.set_major_locator(quarter_loc)
ax.xaxis.set_major_formatter(date_fmt)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# --- Formato eje Y: miles con punto como separador ---
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(round(x)):,}".replace(",", "."))
)

plt.tight_layout()
plt.show()

# (Opcional) Guardar
# plt.savefig("Resultados/Fig_Capital_Cartera_25.png", dpi=300, bbox_inches='tight')
# plt.savefig("Resultados/Fig_Capital_Cartera_25.pdf", bbox_inches='tight')
