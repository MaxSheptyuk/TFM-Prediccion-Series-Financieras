from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------- CONFIGURACIÓN -----------------
SYMBOL_TEST    = "NVDA"
F_FECHA_DESDE  = "2020-01-01"   # 5 años
F_FECHA_HASTA  = "2020-03-31"

# Si True, los “saltos” de capital solo cambian en cierres (SELL_*).
CAPITAL_SOLO_CIERRES = True
CAPITAL_INICIAL = 10000.0

PATH_LOG_UNIFICADO   = "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_15_5.csv"
PATH_PRECIOS_TODOS   = "DATA/AllStocksHistoricalData.csv"

# ---------------- LECTURA DE DATOS -----------------
trade_log = pd.read_csv(PATH_LOG_UNIFICADO, sep=';', parse_dates=['Fecha'])
trade_log = (trade_log[trade_log['Symbol'] == SYMBOL_TEST]
             .sort_values('Fecha').reset_index(drop=True))

df_prices = pd.read_csv(PATH_PRECIOS_TODOS, sep=';', parse_dates=['Fecha'])
df_prices = (df_prices[df_prices['Symbol'] == SYMBOL_TEST]
             .sort_values('Fecha').reset_index(drop=True))

# ---------------- FILTROS DE FECHA -----------------
fecha_desde = pd.to_datetime(F_FECHA_DESDE)
fecha_hasta = pd.to_datetime(F_FECHA_HASTA)

trade_log = trade_log[(trade_log['Fecha'] >= fecha_desde) & (trade_log['Fecha'] <= fecha_hasta)].copy()
df_prices = df_prices[(df_prices['Fecha'] >= fecha_desde) & (df_prices['Fecha'] <= fecha_hasta)].copy()

# ---------------- CURVA DE CAPITAL ALINEADA -----------------
cap_events = trade_log.copy()
if CAPITAL_SOLO_CIERRES:
    cap_events = cap_events[cap_events['Accion'].str.startswith('SELL')]

timeline = df_prices[['Fecha']].copy()

cap_curve = timeline.merge(cap_events[['Fecha', 'Capital_Actual']], on='Fecha', how='left')
cap_curve['Capital_Actual'] = cap_curve['Capital_Actual'].ffill().fillna(CAPITAL_INICIAL)

# ---------------- PLOTEO -----------------
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(16, 8), sharex=True, gridspec_kw={'height_ratios':[2,1]}
)

# --- CHART 1: Evolución de precios (SIN triángulos) ---
ax1.plot(df_prices['Fecha'], df_prices['Close'], label=f'Precio real {SYMBOL_TEST}')
ax1.set_title(f"Evolución de Precios {SYMBOL_TEST}")
ax1.set_ylabel("Precio ($)")
ax1.legend()
ax1.grid(True, alpha=0.25)

# --- CHART 2: Curva de capital (línea SIN puntos) ---
# Usa drawstyle='steps-post' para escalones; quítalo si la prefieres completamente lisa.
ax2.plot(cap_curve['Fecha'], cap_curve['Capital_Actual'],
         label='Capital acumulado', linewidth=2, drawstyle='steps-post')
ax2.set_title(f"Curva de Capital - Backtesting {SYMBOL_TEST}")
ax2.set_ylabel("Capital ($)")
ax2.grid(True, alpha=0.25)
ax2.legend()

# --- Eje X: MUY POCO texto (anual + trimestral) ---
year_loc   = mdates.YearLocator()          # ticks mayores: 1 por año
quarter_loc= mdates.MonthLocator(interval=3)  # menores: cada 3 meses
date_fmt   = mdates.DateFormatter('%Y-%m')    # etiqueta compacta

for ax in (ax1, ax2):
    ax.xaxis.set_major_locator(year_loc)
    ax.xaxis.set_minor_locator(quarter_loc)
    ax.xaxis.set_major_formatter(date_fmt)
    ax.grid(True, which='major', axis='x', alpha=0.35)
    ax.set_xlim(fecha_desde, fecha_hasta)
    ax.margins(x=0)

plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()

# Guardar listo para TFM (opcional)
# plt.savefig(f"Resultados/Fig_Backtest_{SYMBOL_TEST}_2020_2024.png", dpi=300, bbox_inches='tight')
# plt.savefig(f"Resultados/Fig_Backtest_{SYMBOL_TEST}_2020_2024.pdf", bbox_inches='tight')

plt.show()
