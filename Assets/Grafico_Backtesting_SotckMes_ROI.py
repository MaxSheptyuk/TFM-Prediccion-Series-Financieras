import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ================== CONFIG ==================
CAPITAL_INICIAL = 10000.0
F_FECHA_DESDE   = "2020-01-01"
F_FECHA_HASTA   = "2024-12-31"

SYMBOLS_TO_TEST = [
    'NVDA','AAPL','AMZN','LRCX','SBUX','REGN','KLAC','BKNG','AMD','VRTX',
    'MAR','CDNS','CAT','INTU','GILD','MU','EBAY','AXP','AMAT','COST','MSFT',
    'ORCL','ADI','MS','NKE'  # NKE va en la fila inferior (bajo la col. 1)
]

PATH_LOG_UNIFICADO = "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_15_5.csv"
PATH_PRECIOS_TODOS = "DATA/AllStocksHistoricalData.csv"

ADD_TREND = True

# ================== LECTURA ==================
log = pd.read_csv(PATH_LOG_UNIFICADO, sep=";", parse_dates=["Fecha"])
log = log[log["Symbol"].isin(SYMBOLS_TO_TEST)]

prices = pd.read_csv(PATH_PRECIOS_TODOS, sep=";", parse_dates=["Fecha"])
prices = prices[prices["Symbol"].isin(SYMBOLS_TO_TEST)].sort_values(["Symbol","Fecha"])

timeline = (
    prices[(prices["Fecha"] >= F_FECHA_DESDE) & (prices["Fecha"] <= F_FECHA_HASTA)]
    [["Fecha"]].drop_duplicates().sort_values("Fecha").set_index("Fecha")
)

# ================== CAPITAL DIARIO POR STOCK ==================
curvas = []
for sym in SYMBOLS_TO_TEST:
    tl = log[(log["Symbol"] == sym) & (log["Accion"].str.startswith("SELL"))].copy()
    p  = (prices[(prices["Symbol"] == sym) &
                 (prices["Fecha"] >= F_FECHA_DESDE) &
                 (prices["Fecha"] <= F_FECHA_HASTA)]
          [["Fecha"]].drop_duplicates().sort_values("Fecha"))
    if p.empty:
        curvas.append(pd.Series(CAPITAL_INICIAL, index=timeline.index, name=sym))
        continue

    cap = p.merge(tl[["Fecha","Capital_Actual"]], on="Fecha", how="left")
    cap["Capital_Actual"] = cap["Capital_Actual"].ffill().fillna(CAPITAL_INICIAL)
    cap = (cap.set_index("Fecha")["Capital_Actual"]
             .reindex(timeline.index).ffill().fillna(CAPITAL_INICIAL))
    cap.name = sym
    curvas.append(cap)

df_cap = pd.concat(curvas, axis=1)

# ================== CAPITAL MENSUAL (cierre de mes) ==================
cap_monthly = df_cap.resample('M').last()   # ~60 puntos por símbolo

# ================== HELPERS ==================
fmt_thousands = mticker.FuncFormatter(lambda v, _: f"{int(round(v)):,}".replace(",", "."))

# ================== PLOTEO 3×8 + 1 ==================
assert len(SYMBOLS_TO_TEST) == 25
symbols_24  = SYMBOLS_TO_TEST[:24]
symbol_last = SYMBOLS_TO_TEST[24]

fig = plt.figure(figsize=(10, 22), constrained_layout=False)
gs = fig.add_gridspec(
    9, 3,
    height_ratios=[1]*8 + [1],
    hspace=0.85, wspace=0.40
)

def plot_symbol(ax, serie, title):
    s = serie.dropna()
    x = np.arange(1, len(s)+1)
    y = s.values

    # Línea de capital
    ax.plot(x, y, marker='o', markersize=2, linewidth=0.7, color='tab:blue')

    # Tendencia lineal (opcional)
    if ADD_TREND and len(x) > 1:
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m*x + b, color='red', linewidth=0.9)

    # === Escala Y automática con 4 ticks fijos ===
    y_min, y_max = y.min(), y.max()
    span = y_max - y_min
    pad  = max(500, 0.05 * span)  # pequeño margen
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.yaxis.set_major_locator(mticker.LinearLocator(numticks=4))  # <-- 4 ticks
    ax.yaxis.set_major_formatter(fmt_thousands)

    # X simple
    ax.set_xlim(1, len(x))
    ax.set_xticks([1, 24, 48] if len(x) >= 48 else [1, len(x)//2, len(x)])
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', alpha=0.25)
    ax.set_title(title, fontsize=10, pad=6)

# Primeras 24 (3×8)
for i, sym in enumerate(symbols_24):
    r, c = divmod(i, 3)
    ax = fig.add_subplot(gs[r, c])
    plot_symbol(ax, cap_monthly[sym], sym)
    if c == 0:
        ax.set_ylabel("Capital ($)", fontsize=9)
    # En la fila 8, quita 'Mes' en col 1 para no pisar el título abajo
    if r == 7:
        if c == 0:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Mes", fontsize=8, labelpad=0)

# Última fila: solo bajo la columna 1
ax_last = fig.add_subplot(gs[8, 0])
plot_symbol(ax_last, cap_monthly[symbol_last], symbol_last)
ax_last.set_xlabel("Mes", fontsize=9, labelpad=2)
ax_last.set_ylabel("Capital ($)", fontsize=9)

# Vaciar celdas 8,1 y 8,2
fig.add_subplot(gs[8, 1]).axis("off")
fig.add_subplot(gs[8, 2]).axis("off")

fig.suptitle("Evolución mensual de capital por activo (2020–2024)", fontsize=14, y=0.995)
plt.tight_layout()
plt.show()
