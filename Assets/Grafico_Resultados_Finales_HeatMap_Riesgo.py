# -*- coding: utf-8 -*-
# Heatmap de Drawdown máximo anual (%) por modelo (etiquetas manuales).
# - equity(t) = capital_total_inicial (desde log) + cumsum(PnL_diario_agg SELL)
# - Colormap atenuado: <10% muy claro, 10–15% naranjas, >15% rojos.

import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable

# ============== 1) ETIQUETAS MANUALES ==============
# Rellena con tus pares {Etiqueta bonita : ruta al CSV}
LABELS = {
    # ==== XGB (TARGET_TREND_ANG) ====
    "XGB-W5H5"    : "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_5_5.csv",
    "XGB-W5H8"    : "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_5_8.csv",
    "XGB-W5H10"   : "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_5_10.csv",
    "XGB-W5H12"   : "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_5_12.csv",
    "XGB-W5H15"   : "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_5_15.csv",

    "XGB-W10H5"   : "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_10_5.csv",
    "XGB-W10H8"   : "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_10_8.csv",
    "XGB-W10H10"  : "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_10_10.csv",
    "XGB-W10H12"  : "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_10_12.csv",
    "XGB-W10H15"  : "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_10_15.csv",

    "XGB-W15H5"   : "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_15_5.csv",
    "XGB-W15H8"   : "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_15_8.csv",
    "XGB-W15H10"  : "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_15_10.csv",
    "XGB-W15H12"  : "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_15_12.csv",
    "XGB-W15H15"  : "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_15_15.csv",

    # ==== MLP (OHLCV) ====
    "MLP (32,)"     : "Resultados/Trading_Log_AllStocks_MLP_OHLCV_32.csv",
    "MLP (64,)"     : "Resultados/Trading_Log_AllStocks_MLP_OHLCV_64.csv",
    "MLP (64,32)"   : "Resultados/Trading_Log_AllStocks_MLP_OHLCV_64_32.csv",
    "MLP (128,64)"  : "Resultados/Trading_Log_AllStocks_MLP_OHLCV_128_64.csv",
    "MLP (128,)"    : "Resultados/Trading_Log_AllStocks_MLP_OHLCV_128.csv",
}

F_INI, F_FIN = "2020-01-01", "2024-12-31"  # recorte temporal opcional

# ============== 2) ESCALA DE COLOR (0–5% más oscuro) ==============
VMIN, VMAX = 0, 25
BOUNDS = [0, 5, 8, 10, 12, 15, 17, 20, 25]
# Primer color ~10% más oscuro que el anterior (beige muy claro, no blanco puro)
RISK_COLORS = [
    "#f2ecde",  # 0–5   (antes casi blanco)
    "#ffeab0",  # 5–8
    "#ffd98c",  # 8–10
    "#ffc36b",  # 10–12
    "#ffad57",  # 12–15
    "#f46b4e",  # 15–17
    "#e04545",  # 17–20
    "#b51f29",  # 20–25
]
ANNOT_FMT = "{:.0f}%"

# ============== 3) HELPERS ==============
def equity_cartera_desde_log(df: pd.DataFrame, f_ini=None, f_fin=None) -> pd.Series:
    """
    equity(t) = capital_total_inicial + cumsum(PnL_diario_agg SELL)
      - capital_total_inicial: suma del Capital_Actual del primer registro de cada símbolo.
      - PnL_diario_agg: suma de Profit diario de TODA la cartera (solo Accion que empieza por 'SELL').
    """
    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    if f_ini: df = df[df["Fecha"] >= pd.to_datetime(f_ini)]
    if f_fin: df = df[df["Fecha"] <= pd.to_datetime(f_fin)]

    needed = {"Symbol","Capital_Actual"}
    if not needed.issubset(df.columns):
        raise ValueError("El log debe tener columnas 'Symbol' y 'Capital_Actual'.")

    first_per_sym = (df.sort_values("Fecha")
                       .groupby("Symbol", as_index=False).first())
    capital_inicial_total = float(first_per_sym["Capital_Actual"].fillna(0).sum())

    df_sell = df[df["Accion"].astype(str).str.startswith("SELL", na=False)].copy()
    pnl_diario = (df_sell.groupby("Fecha", as_index=True)["Profit"]
                        .sum().sort_index())
    if pnl_diario.empty:
        raise ValueError("No hay operaciones SELL para reconstruir PnL diario.")

    timeline = pd.date_range(pnl_diario.index.min(), pnl_diario.index.max(), freq="D")
    pnl_diario = pnl_diario.reindex(timeline).fillna(0.0)
    equity = capital_inicial_total + pnl_diario.cumsum()
    equity.index = timeline
    equity.name = "Equity_Cartera"
    return equity

def max_drawdown_percent(eq: pd.Series) -> float:
    v = eq.values.astype(float)
    peaks = np.maximum.accumulate(v)
    dd = (peaks - v) / np.maximum(peaks, 1e-12)
    return float(dd.max() * 100.0)

def yearly_dd(eq: pd.Series) -> pd.DataFrame:
    rows = []
    for y in sorted(pd.Index(eq.index).year.unique()):
        eq_y = eq[eq.index.year == y]
        if len(eq_y) < 2: 
            continue
        rows.append({"Anio": int(y), "DD_Pct": max_drawdown_percent(eq_y)})
    return pd.DataFrame(rows)

# ============== 4) CARGA Y CÁLCULO ==============
if not LABELS:
    raise RuntimeError("LABELS está vacío. Añade {'Etiqueta':'ruta/al/log.csv'}.")

rows = []
for etiqueta, path in LABELS.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"No encuentro el fichero para '{etiqueta}': {path}")
    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        df = pd.read_csv(path)

    needed = {"Fecha","Accion","Profit","Symbol","Capital_Actual"}
    if not needed.issubset(df.columns):
        faltan = sorted(needed - set(df.columns))
        raise ValueError(f"{os.path.basename(path)} no tiene columnas requeridas: {faltan}")

    df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce").fillna(0.0)
    if "Capital_Actual" in df.columns:
        df["Capital_Actual"] = pd.to_numeric(df["Capital_Actual"], errors="coerce")

    eq = equity_cartera_desde_log(df, F_INI, F_FIN)
    dd_y = yearly_dd(eq)
    if dd_y.empty: 
        continue
    dd_y["Modelo_Config"] = etiqueta
    rows.append(dd_y)

if not rows:
    raise RuntimeError("No se pudieron derivar DD% anuales a partir de los logs de LABELS.")

df = pd.concat(rows, ignore_index=True)

# ============== 5) PIVOT + ORDEN Y ==============
order_y = (df.groupby("Modelo_Config")["DD_Pct"]
             .max().sort_values(ascending=False).index.tolist())
years = sorted(df["Anio"].unique())
mat = (df.pivot(index="Modelo_Config", columns="Anio", values="DD_Pct")
         .reindex(index=order_y, columns=years))

# ============== 6) HEATMAP ==============
cmap_risk = ListedColormap(RISK_COLORS)
norm_risk = BoundaryNorm(BOUNDS, cmap_risk.N, clip=True)

fig_h = max(6, 0.5 * len(order_y))
fig, ax = plt.subplots(figsize=(12, fig_h))

im = ax.imshow(mat.values, aspect="auto", cmap=cmap_risk, norm=norm_risk)

# Anotaciones (blanco solo cuando fondo >15%)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        val = mat.iat[i, j]
        if pd.isna(val): 
            continue
        txt_color = "white" if val >= 15 else "#1b1b1b"
        ax.text(j, i, ANNOT_FMT.format(val), ha="center", va="center",
                fontsize=9, color=txt_color)

ax.set_xticks(range(len(years))); ax.set_xticklabels([str(y) for y in years])
ax.set_yticks(range(len(order_y))); ax.set_yticklabels(order_y)
ax.set_xlabel("Año"); ax.set_ylabel("Modelo")
ax.set_title("Riesgo: Drawdown máximo anual (%) por modelo")
ax.invert_yaxis()

cbar = plt.colorbar(ScalarMappable(norm=norm_risk, cmap=cmap_risk), ax=ax, pad=0.01)
cbar.set_label("Caída de capital, drawdown (%)")
cbar.set_ticks(BOUNDS)

plt.tight_layout()
plt.show()

# ============== 7) EXPORT OPCIONAL ==============
# import os
# os.makedirs("Resultados", exist_ok=True)
# plt.savefig(os.path.join("Resultados", "Figura_Heatmap_DD_Anual_Modelos_suave.png"),
#             dpi=220, bbox_inches="tight")
# mat.reset_index().to_csv(os.path.join("Resultados", "Tabla_Heatmap_DD_Anual_Modelos.csv"),
#                          sep=";", index=True)
