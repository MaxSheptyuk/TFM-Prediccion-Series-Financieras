# Aportación por símbolo al ROI% anual (barras verticales apiladas)
# Modelo: XGB W5H5
# - Altura barra = ROI% cartera en el año
# - Segmentos = contribución ROI% por símbolo (PnL anual símbolo / equity inicial del año * 100)
# - Opción de agrupar contribuciones pequeñas en "Otros" (TOP_K)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ========= Parámetros base =========
CAPITAL_INICIAL_X_STOCK = 10_000.0
N_SIMBOLOS = 25
CAPITAL_INICIAL = CAPITAL_INICIAL_X_STOCK * N_SIMBOLOS

SYMBOLS_CARTERA = [
    'NVDA','AAPL','AMZN','LRCX','SBUX','REGN','KLAC','BKNG','AMD','VRTX',
    'MAR','CDNS','CAT','INTU','GILD','MU','EBAY','AXP','AMAT','COST','MSFT',
    'ORCL','ADI','MS','NKE'
]

F_INI = "2020-01-01"
F_FIN = "2024-12-31"
ANIOS = [2020, 2021, 2022, 2023, 2024]

PATH_PRECIOS_TODOS = "DATA/AllStocksHistoricalData.csv"
# Modelo objetivo:
PATH_LOG = "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_5_5.csv"  # XGB W5H5

# Visualización
TOP_K = 8  # cuántos símbolos mostrar individualmente (el resto se agrupa en "Otros")

# ========= Helpers =========
def load_prices_timeline(path_prices: str, f_ini: str, f_fin: str) -> pd.DatetimeIndex:
    if not os.path.exists(path_prices):
        raise FileNotFoundError(f"No existe dataset de precios: {path_prices}")
    dfp = pd.read_csv(path_prices, sep=";", parse_dates=["Fecha"])
    tl = (dfp[(dfp["Fecha"] >= f_ini) & (dfp["Fecha"] <= f_fin)]
          .sort_values("Fecha")[["Fecha"]].drop_duplicates()
          .set_index("Fecha").index)
    return tl

def load_log(path_log: str) -> pd.DataFrame:
    if not os.path.exists(path_log):
        raise FileNotFoundError(f"No existe log: {path_log}")
    df = pd.read_csv(path_log, sep=";", parse_dates=["Fecha"])
    df = df[df["Symbol"].isin(SYMBOLS_CARTERA)].copy()
    # Solo operaciones SELL para PnL realizado
    df = df[df["Accion"].astype(str).str.startswith("SELL", na=False)]
    return df

def equity_diario(df_sell: pd.DataFrame, timeline: pd.DatetimeIndex) -> pd.Series:
    daily = (df_sell.groupby("Fecha", as_index=False)["Profit"].sum()
                    .sort_values("Fecha"))
    equity = pd.Series(CAPITAL_INICIAL, index=timeline, name="Equity")
    if len(daily):
        pnl = daily.set_index("Fecha")["Profit"].reindex(timeline).fillna(0.0)
        equity = (CAPITAL_INICIAL + pnl.cumsum()).rename("Equity")
    return equity

def pnl_anual_por_symbol(df_sell: pd.DataFrame, year: int) -> pd.Series:
    """PnL realizado anual por símbolo (suma de Profit en SELL dentro del año)."""
    mask = df_sell["Fecha"].dt.year == year
    dfy = df_sell[mask]
    ser = dfy.groupby("Symbol")["Profit"].sum()
    # incluye símbolos sin trades como 0
    ser = ser.reindex(SYMBOLS_CARTERA).fillna(0.0)
    return ser

# ========= Cálculo de contribuciones =========
timeline = load_prices_timeline(PATH_PRECIOS_TODOS, F_INI, F_FIN)
df_sell = load_log(PATH_LOG)
eq = equity_diario(df_sell, timeline)

# Equity inicial por año (equity del último día del año previo; 2020 usa capital inicial)
equity_ini_anio = {}
for y in ANIOS:
    if y == ANIOS[0]:
        equity_ini_anio[y] = CAPITAL_INICIAL
    else:
        prev_last = eq[eq.index.year == (y-1)]
        equity_ini_anio[y] = float(prev_last.iloc[-1]) if len(prev_last) else np.nan

# Contribución ROI% por símbolo y año
rows = []
for y in ANIOS:
    pnl_sym = pnl_anual_por_symbol(df_sell, y)  # PnL anual por símbolo
    eq_ini = equity_ini_anio[y]
    if np.isnan(eq_ini) or eq_ini == 0:
        contrib = pnl_sym * 0.0
    else:
        contrib = pnl_sym / eq_ini * 100.0  # ROI% de contribución
    for sym, val in contrib.items():
        rows.append({"Año": y, "Symbol": sym, "ROI_contrib_%": float(val)})

df_contrib = pd.DataFrame(rows)

# ROI% total por año (comprobación: suma contribuciones ≈ ROI anual)
roi_total_anual = {}
for y in ANIOS:
    eq_y = eq[eq.index.year == y]
    if len(eq_y):
        start = float(equity_ini_anio[y])
        end = float(eq_y.iloc[-1])
        roi_total_anual[y] = (end - start) / start * 100.0
    else:
        roi_total_anual[y] = 0.0

# ========= Agrupar “Otros” para legibilidad (TOP_K por contribución absoluta media) =========
# Ranking global por importancia media (abs) a través de los años
mean_abs = (df_contrib.groupby("Symbol")["ROI_contrib_%"]
            .apply(lambda s: s.abs().mean())).sort_values(ascending=False)
top_syms = mean_abs.index[:TOP_K].tolist()
df_plot = df_contrib.copy()
df_plot["Symbol_group"] = df_plot["Symbol"].where(df_plot["Symbol"].isin(top_syms), "Otros")

# Pivot por año x grupo
pivot = (df_plot.groupby(["Año", "Symbol_group"])["ROI_contrib_%"]
         .sum().unstack(fill_value=0.0).reindex(ANIOS))

# Asegurar orden de columnas (top_syms + 'Otros' si existe)
cols = [c for c in top_syms if c in pivot.columns]
if "Otros" in pivot.columns:
    cols.append("Otros")
pivot = pivot[cols]

# ========= Plot: barras apiladas por año =========
fig, ax = plt.subplots(figsize=(11, 6))

# paleta simple para grupos
palette = plt.cm.tab20.colors  # muchos colores disponibles
colors = {sym: palette[i % len(palette)] for i, sym in enumerate(cols)}

bottom = np.zeros(len(pivot))
x = np.arange(len(pivot.index))  # años en orden

for sym in cols:
    vals = pivot[sym].values
    ax.bar(x, vals, bottom=bottom, label=sym, color=colors[sym], edgecolor="white", linewidth=0.6)
    bottom += vals

# Etiquetas de años
ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in ANIOS])

ax.set_ylabel("ROI anual (%)")
ax.set_title("Aportación por símbolo al ROI anual – XGB W5H5 (barras apiladas)")

# Mostrar la suma (ROI total) encima de cada barra
for i, y in enumerate(ANIOS):
    total = roi_total_anual[y]
    ax.text(i, total + np.sign(total)*1.0, f"{total:.1f}%", ha="center", va="bottom" if total>=0 else "top", fontsize=9)

# Formato de porcentaje en eje Y
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.grid(axis="y", alpha=0.25)

# Leyenda fuera, a la derecha
ax.legend(title="Símbolo", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=9)

plt.tight_layout()
os.makedirs("Resultados", exist_ok=True)
plt.savefig("Resultados/Figura_Aportacion_ROI_Anual_XGB_W5H5_Stacked.png", dpi=220, bbox_inches="tight")
plt.show()

# (Opcional) Exportar tabla de contribuciones agregadas (con 'Otros')
pivot.reset_index().rename(columns={"index":"Año"}).to_csv(
    "Resultados/Tabla_Aportacion_ROI_Anual_XGB_W5H5.csv", sep=";", index=False
)
