import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =============== CONFIG =================
CAPITAL_INICIAL = 10000.0      # por activo
F_FECHA_DESDE   = "2020-01-01"
F_FECHA_HASTA   = "2024-12-31"

# Cartera de símbolos
SYMBOLS_CARTERA = [
    'NVDA','AAPL','AMZN','LRCX','SBUX','REGN','KLAC','BKNG','AMD','VRTX',
    'MAR','CDNS','CAT','INTU','GILD','MU','EBAY','AXP','AMAT','COST','MSFT',
    'ORCL','ADI','MS','NKE'
]

PATH_LOG_UNIFICADO = "Resultados/Trading_Log_AllStocks.csv"
PATH_PRECIOS_TODOS = "DATA/AllStocksHistoricalData.csv"

# =============== LECTURA BASES ===============
trade_log = pd.read_csv(PATH_LOG_UNIFICADO, sep=";", parse_dates=["Fecha"])
trade_log = trade_log[trade_log["Symbol"].isin(SYMBOLS_CARTERA)].copy()

prices_all = pd.read_csv(PATH_PRECIOS_TODOS, sep=";", parse_dates=["Fecha"])
prices_all = prices_all[prices_all["Symbol"].isin(SYMBOLS_CARTERA)].copy()
prices_all = prices_all.sort_values(["Symbol","Fecha"])

# Timeline global
timeline = (
    prices_all[(prices_all["Fecha"] >= F_FECHA_DESDE) &
               (prices_all["Fecha"] <= F_FECHA_HASTA)]
    [["Fecha"]].drop_duplicates().sort_values("Fecha").set_index("Fecha")
)

# =============== CURVAS DE CAPITAL POR STOCK ===============
curvas = []
for sym in SYMBOLS_CARTERA:
    tl_sym = trade_log[trade_log["Symbol"] == sym].copy()
    tl_sym = tl_sym[tl_sym["Accion"].str.startswith("SELL")]

    p_sym = (prices_all[(prices_all["Symbol"] == sym) &
                        (prices_all["Fecha"] >= F_FECHA_DESDE) &
                        (prices_all["Fecha"] <= F_FECHA_HASTA)]
             [["Fecha"]].drop_duplicates().sort_values("Fecha"))

    if p_sym.empty:
        cap_curve = pd.Series(CAPITAL_INICIAL, index=timeline.index, name=sym)
        curvas.append(cap_curve)
        continue

    cap_curve = p_sym.merge(
        tl_sym[["Fecha", "Capital_Actual"]],
        on="Fecha", how="left"
    )
    cap_curve["Capital_Actual"] = cap_curve["Capital_Actual"].ffill().fillna(CAPITAL_INICIAL)
    cap_curve = cap_curve.set_index("Fecha")["Capital_Actual"]
    cap_curve = cap_curve.reindex(timeline.index).ffill().fillna(CAPITAL_INICIAL)
    cap_curve.name = sym
    curvas.append(cap_curve)

df_capital = pd.concat(curvas, axis=1)

# =============== ROI ANUAL POR STOCK ===============
df_capital_yearly = df_capital.copy()
df_capital_yearly["Year"] = df_capital_yearly.index.year

cap_ini = df_capital_yearly.groupby("Year").first()
cap_fin = df_capital_yearly.groupby("Year").last()

cap_ini_long = cap_ini.reset_index().melt(id_vars="Year", var_name="Symbol", value_name="CapIni")
cap_fin_long = cap_fin.reset_index().melt(id_vars="Year", var_name="Symbol", value_name="CapFin")

roi_year = pd.merge(cap_ini_long, cap_fin_long, on=["Year","Symbol"], how="inner")
roi_year["ROI"] = (roi_year["CapFin"] / roi_year["CapIni"] - 1.0) * 100.0

years_sorted = sorted(roi_year["Year"].unique())
pivot = roi_year.pivot(index="Symbol", columns="Year", values="ROI").reindex(index=SYMBOLS_CARTERA, columns=years_sorted)

# =============== HEATMAP CON ETIQUETAS =================
fig, ax = plt.subplots(figsize=(14, 10))

vmax = np.nanpercentile(np.abs(pivot.values), 95)
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", norm=norm)

# Ticks
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)

# Grid
ax.set_xticks(np.arange(-.5, len(pivot.columns), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(pivot.index), 1), minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth=1.5)

# Anotar valores en celdas
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color="black")

# Barra de color
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("ROI anual (%)")

ax.set_title("Mapa de calor: ROI anual por stock (2020–2024)", fontsize=14)
plt.tight_layout()
plt.show()
