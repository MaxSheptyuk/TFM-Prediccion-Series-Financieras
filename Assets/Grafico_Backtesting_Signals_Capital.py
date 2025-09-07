from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, StrMethodFormatter, MaxNLocator, AutoMinorLocator

# ---------------- CONFIGURACIÓN -----------------
SYMBOL_TEST    = "NVDA"
F_FECHA_DESDE  = "2020-01-01"
F_FECHA_HASTA  = "2020-03-31"

# Colores señales
COLOR_BUY  = "green"
COLOR_SELL = "red"

# Si True, los “saltos” de capital solo cambian en cierres (SELL_*).
CAPITAL_SOLO_CIERRES = True
CAPITAL_INICIAL = 10000.0  # por si el primer SELL cae más tarde

PATH_LOG_UNIFICADO   = "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_15_5.csv"
PATH_PRECIOS_TODOS   = "DATA/AllStocksHistoricalData.csv"

# ---------------- LECTURA DE DATOS -----------------
# Log de trading UNIFICADO -> filtrar por símbolo
trade_log = pd.read_csv(PATH_LOG_UNIFICADO, sep=';', parse_dates=['Fecha'])
trade_log = (
    trade_log[trade_log['Symbol'] == SYMBOL_TEST]
    .sort_values('Fecha')
    .reset_index(drop=True)
)

# Precios históricos -> filtrar por símbolo
df_prices = pd.read_csv(PATH_PRECIOS_TODOS, sep=';', parse_dates=['Fecha'])
df_prices = (
    df_prices[df_prices['Symbol'] == SYMBOL_TEST]
    .sort_values('Fecha')
    .reset_index(drop=True)
)

# Asegurar tipo numérico en precio y limpiar nulos
df_prices['Close'] = pd.to_numeric(df_prices['Close'], errors='coerce').astype(float)
df_prices = df_prices.dropna(subset=['Close'])

# ---------------- FILTROS DE FECHA -----------------
fecha_desde = pd.to_datetime(F_FECHA_DESDE)
fecha_hasta = pd.to_datetime(F_FECHA_HASTA)

trade_log = trade_log[(trade_log['Fecha'] >= fecha_desde) & (trade_log['Fecha'] <= fecha_hasta)].copy()
df_prices = df_prices[(df_prices['Fecha'] >= fecha_desde) & (df_prices['Fecha'] <= fecha_hasta)].copy()

# ---------------- MARCA SEÑALES EN PRECIOS -----------------
df_prices['Buy']  = df_prices['Fecha'].isin(trade_log.loc[trade_log['Accion'] == 'BUY', 'Fecha'])
df_prices['Sell'] = df_prices['Fecha'].isin(trade_log.loc[trade_log['Accion'].str.startswith('SELL'), 'Fecha'])

# ---------------- CURVA DE CAPITAL ALINEADA -----------------
# 1) Eventos de capital (por defecto, solo en cierres SELL)
cap_events = trade_log.copy()
if CAPITAL_SOLO_CIERRES:
    cap_events = cap_events[cap_events['Accion'].str.startswith('SELL')]

# 2) Línea temporal diaria basada en los precios (todas las fechas del gráfico)
timeline = df_prices[['Fecha']].copy()

# 3) Unimos eventos a timeline y rellenamos capital entre eventos
cap_curve = timeline.merge(cap_events[['Fecha', 'Capital_Actual']], on='Fecha', how='left')
cap_curve['Capital_Actual'] = pd.to_numeric(cap_curve['Capital_Actual'], errors='coerce')
cap_curve['Capital_Actual'] = cap_curve['Capital_Actual'].ffill().fillna(CAPITAL_INICIAL)

# ---------------- PLOTEO -----------------
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(16, 8), sharex=True, gridspec_kw={'height_ratios':[2,1]}
)

# --- CHART 1: Evolución de precios con señales ---
ax1.plot(df_prices['Fecha'], df_prices['Close'], label=f'Precio real {SYMBOL_TEST}')

ax1.scatter(df_prices.loc[df_prices['Buy'], 'Fecha'],  df_prices.loc[df_prices['Buy'],  'Close'],
            marker='^', color=COLOR_BUY, label='BUY', s=70, zorder=5)
ax1.scatter(df_prices.loc[df_prices['Sell'], 'Fecha'], df_prices.loc[df_prices['Sell'], 'Close'],
            marker='v', color=COLOR_SELL, label='SELL', s=70, zorder=5)

ax1.set_title(f"Evolución de Precios {SYMBOL_TEST} y Señales de Trading")
ax1.set_ylabel("Precio ($)")
ax1.legend()
ax1.grid(True)

# --- Forzar decimales en eje Y de precios (nada de enteros) ---
ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))  # 2 decimales
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
ax1.yaxis.set_major_locator(MaxNLocator(nbins=8, prune=None))  # ticks “bonitos”

# --- CHART 2: Curva de capital (alineada en fechas) ---
ax2.plot(cap_curve['Fecha'], cap_curve['Capital_Actual'],
         label='Capital acumulado', marker='o', drawstyle='steps-post')
ax2.set_title(f"Curva de Capital - Backtesting {SYMBOL_TEST}")
ax2.set_ylabel("Capital ($)")
ax2.grid(True)
ax2.legend()

# Formato con separador de miles en capital (sin decimales)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", ".")))
ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

# --- Eje X: ticks y rejilla (dd-MM-yyyy) y límites estrictos del rango ---
major_locator = mdates.WeekdayLocator(interval=1)      # semanal
minor_locator = mdates.DayLocator(interval=1)          # diario
date_fmt      = mdates.DateFormatter('%d-%m-%Y')       # dd-MM-yyyy

for ax in (ax1, ax2):
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.xaxis.set_major_formatter(date_fmt)
    ax.grid(True, which='major', axis='x', alpha=0.5)
    ax.grid(True, which='minor', axis='x', alpha=0.2)
    ax.set_xlim(fecha_desde, fecha_hasta)
    ax.margins(x=0)

plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()

# (Opcional) Guardar figura lista para el TFM
# plt.savefig(f"Resultados/Fig_Backtest_{SYMBOL_TEST}_{F_FECHA_DESDE}_{F_FECHA_HASTA}.png",
#             dpi=300, bbox_inches='tight')
# plt.savefig(f"Resultados/Fig_Backtest_{SYMBOL_TEST}_{F_FECHA_DESDE}_{F_FECHA_HASTA}.pdf",
#             bbox_inches='tight')

plt.show()
