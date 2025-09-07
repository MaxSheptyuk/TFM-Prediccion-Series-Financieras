# -*- coding: utf-8 -*-
# Resumen por modelo (CONSISTENTE con el script antiguo):
# - Universo fijo de 25 símbolos (10k por símbolo => capital inicial 250k)
# - Equity = 250k + cumsum(Profit diario de SELL) agregado a nivel cartera
# - ROI / DD sobre esa equity
# - CAGR con años efectivos (min(fecha) -> max(fecha) del propio log)
# - Orden por ROI desc, formateo español
# - Export: CSV formateado, CSV numérico y XLSX (si hay engine)

import os
import pandas as pd
import numpy as np
from pathlib import Path
from importlib.util import find_spec

RESULTS_DIR = "Resultados"

# ===== 1) Diccionario etiqueta -> fichero =====
LABELS = {
    # XGB (TARGET_TREND_ANG)
    "XGB-W5H5"  : f"{RESULTS_DIR}/Trading_Log_AllStocks_TARGET_TREND_ANG_5_5.csv",
    "XGB-W5H8"  : f"{RESULTS_DIR}/Trading_Log_AllStocks_TARGET_TREND_ANG_5_8.csv",
    "XGB-W5H10" : f"{RESULTS_DIR}/Trading_Log_AllStocks_TARGET_TREND_ANG_5_10.csv",
    "XGB-W5H12" : f"{RESULTS_DIR}/Trading_Log_AllStocks_TARGET_TREND_ANG_5_12.csv",
    "XGB-W5H15" : f"{RESULTS_DIR}/Trading_Log_AllStocks_TARGET_TREND_ANG_5_15.csv",
    "XGB-W10H5" : f"{RESULTS_DIR}/Trading_Log_AllStocks_TARGET_TREND_ANG_10_5.csv",
    "XGB-W10H8" : f"{RESULTS_DIR}/Trading_Log_AllStocks_TARGET_TREND_ANG_10_8.csv",
    "XGB-W10H10": f"{RESULTS_DIR}/Trading_Log_AllStocks_TARGET_TREND_ANG_10_10.csv",
    "XGB-W10H12": f"{RESULTS_DIR}/Trading_Log_AllStocks_TARGET_TREND_ANG_10_12.csv",
    "XGB-W10H15": f"{RESULTS_DIR}/Trading_Log_AllStocks_TARGET_TREND_ANG_10_15.csv",
    "XGB-W15H5" : f"{RESULTS_DIR}/Trading_Log_AllStocks_TARGET_TREND_ANG_15_5.csv",
    "XGB-W15H8" : f"{RESULTS_DIR}/Trading_Log_AllStocks_TARGET_TREND_ANG_15_8.csv",
    "XGB-W15H10": f"{RESULTS_DIR}/Trading_Log_AllStocks_TARGET_TREND_ANG_15_10.csv",
    "XGB-W15H12": f"{RESULTS_DIR}/Trading_Log_AllStocks_TARGET_TREND_ANG_15_12.csv",
    "XGB-W15H15": f"{RESULTS_DIR}/Trading_Log_AllStocks_TARGET_TREND_ANG_15_15.csv",
    # MLP (OHLCV)
    "MLP (32,)"    : f"{RESULTS_DIR}/Trading_Log_AllStocks_MLP_OHLCV_32.csv",
    "MLP (64,)"    : f"{RESULTS_DIR}/Trading_Log_AllStocks_MLP_OHLCV_64.csv",
    "MLP (64,32)"  : f"{RESULTS_DIR}/Trading_Log_AllStocks_MLP_OHLCV_64_32.csv",
    "MLP (128,64)" : f"{RESULTS_DIR}/Trading_Log_AllStocks_MLP_OHLCV_128_64.csv",
    "MLP (128,)"   : f"{RESULTS_DIR}/Trading_Log_AllStocks_MLP_OHLCV_128.csv",
}

# ===== 2) Baseline fijo de cartera =====
CAPITAL_X_STOCK = 10_000.0
SYMBOLS_CARTERA = [
    'NVDA','AAPL','AMZN','LRCX','SBUX','REGN','KLAC','BKNG','AMD','VRTX',
    'MAR','CDNS','CAT','INTU','GILD','MU','EBAY','AXP','AMAT','COST','MSFT',
    'ORCL','ADI','MS','NKE'
]
N_BASE_SYMBOLS = len(SYMBOLS_CARTERA)
CAPITAL_INICIAL = CAPITAL_X_STOCK * N_BASE_SYMBOLS  # 250.000

# ===== 3) Utilidades =====
def years_between(d0: pd.Timestamp, d1: pd.Timestamp) -> float:
    return max((d1 - d0).days, 1) / 365.25

def equity_from_profit(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Equity global: 250k + cumsum de Profit diario (SELL*), filtrando a SYMBOLS_CARTERA."""
    df = df[df["Symbol"].isin(SYMBOLS_CARTERA)].copy()
    df_trades = df[df["Accion"].astype(str).str.startswith("SELL", na=False)].copy()

    # De-dup defensivo si hay columnas
    dedup_keys = [c for c in ["Symbol","Fecha","Accion","Precio","Profit"] if c in df_trades.columns]
    if dedup_keys:
        df_trades = df_trades.drop_duplicates(subset=dedup_keys, keep="last")

    if len(df_trades):
        daily = (df_trades.groupby("Fecha", as_index=False)["Profit"].sum()
                          .sort_values("Fecha").reset_index(drop=True))
        equity = CAPITAL_INICIAL + np.cumsum(daily["Profit"].values)
        fechas = daily["Fecha"].values
    else:
        equity = np.array([CAPITAL_INICIAL])
        fechas = np.array([df["Fecha"].min()]) if len(df) else np.array([])
    return equity, fechas

def calcular_metricas_modelo(path_csv: str) -> dict:
    df = pd.read_csv(path_csv, sep=";", parse_dates=["Fecha"])
    req = {"Symbol","Fecha","Accion","Profit","Capital_Actual"}
    if not req.issubset(df.columns):
        raise ValueError(f"{os.path.basename(path_csv)} carece de columnas: {req}")

    df = df.sort_values(["Symbol","Fecha"]).reset_index(drop=True)

    equity, fechas = equity_from_profit(df)

    cap_ini = CAPITAL_INICIAL
    cap_fin = float(equity[-1])
    cap_gan = cap_fin - cap_ini
    roi_pct = 100.0 * cap_gan / cap_ini if cap_ini > 0 else np.nan

    # Drawdown sobre equity
    run_max = np.maximum.accumulate(equity)
    dd = run_max - equity
    dd_pct = np.divide(dd, run_max, out=np.zeros_like(dd, dtype=float), where=run_max>0)
    i_trough = int(np.argmax(dd)) if len(dd) else 0
    max_dd = float(dd[i_trough]) if len(dd) else 0.0
    max_dd_pct = float(dd_pct[i_trough]) * 100.0 if len(dd) else 0.0

    # Trades, winrate, payoff, expectancy (solo SELL en cartera)
    df_trades = df[
        df["Symbol"].isin(SYMBOLS_CARTERA) &
        df["Accion"].astype(str).str.startswith("SELL", na=False)
    ].copy()
    n_trades = int(df_trades.shape[0])
    winrate = (df_trades["Profit"] > 0).mean()*100.0 if n_trades else 0.0
    g = df_trades.loc[df_trades["Profit"] > 0, "Profit"]
    l = df_trades.loc[df_trades["Profit"] < 0, "Profit"]
    payoff = float(g.mean()/abs(l.mean())) if (len(g)>0 and len(l)>0 and abs(l.mean())>1e-12) else np.nan
    expectancy = float(df_trades["Profit"].mean() if n_trades else 0.0)

    # CAGR con años efectivos del LOG (min -> max fecha del propio df)
    if len(df):
        anios = years_between(df["Fecha"].min(), df["Fecha"].max())
    else:
        anios = np.nan
    cagr_pct = (((cap_fin / cap_ini) ** (1.0 / anios) - 1.0) * 100.0) if (cap_ini > 0 and anios > 0) else np.nan

    return {
        "Capital inicial $": cap_ini,
        "Capital final $": cap_fin,
        "Capital ganado $": cap_gan,
        "ROI_raw": roi_pct,          # numérico para ordenar
        "Max DD $": max_dd,
        "Max DD %": max_dd_pct,
        "% Ganadores": winrate,
        "Expectancy $": expectancy,
        "Payoff ratio": payoff,
        "CAGR %": cagr_pct,
        "Trades": n_trades,
        "Símbolos": N_BASE_SYMBOLS
    }

# ===== 4) Cálculo por modelo =====
rows = []
for label, fpath in LABELS.items():
    if not Path(fpath).exists():
        print(f"[AVISO] No encontrado: {fpath} ({label}) — se omite.")
        continue
    m = calcular_metricas_modelo(fpath)
    m["Variable Objetivo"] = label
    rows.append(m)

if not rows:
    raise SystemExit("No se pudo calcular ninguna métrica. Revisa rutas en LABELS.")

df_num = pd.DataFrame(rows)
# Orden definitivo por ROI (numérico) DESC
df_num = df_num.sort_values("ROI_raw", ascending=False).reset_index(drop=True)

# ===== 5) Formato español (no altera el orden) =====
def fmt_num(x, dec=2):
    return "-" if not np.isfinite(x) else f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")
def fmt_pct(x, dec=2):
    return "-" if not np.isfinite(x) else f"{x:.{dec}f}%".replace(".", ",")

tabla = df_num.copy()
for c in ["Capital inicial $","Capital final $","Capital ganado $","Max DD $","Expectancy $"]:
    tabla[c] = tabla[c].apply(lambda v: fmt_num(v, 0))
tabla["ROI %"] = tabla["ROI_raw"].apply(lambda v: fmt_pct(v, 2))
for c in ["Max DD %","% Ganadores","CAGR %"]:
    tabla[c] = tabla[c].apply(lambda v: fmt_pct(v, 2))
tabla["Payoff ratio"] = tabla["Payoff ratio"].apply(lambda v: fmt_num(v, 2))
tabla["Trades"] = tabla["Trades"].apply(lambda v: fmt_num(v, 0))

tabla = tabla[[
    "Variable Objetivo","Capital inicial $","Capital final $","Capital ganado $",
    "ROI %","Max DD $","Max DD %","% Ganadores","Expectancy $","Payoff ratio","CAGR %","Trades","Símbolos"
]]

print("\n===== TABLA RESUMEN (consistente) ORDENADA POR ROI =====")
print(tabla.to_string(index=False))

# ===== 6) Exports robustos =====
os.makedirs(RESULTS_DIR, exist_ok=True)

# CSV formateado (texto bonito)
out_csv_fmt = f"{RESULTS_DIR}/Resumen_Rendimiento_Cartera_Por_Modelo_FORMATADO.csv"
tabla.to_csv(out_csv_fmt, sep=";", index=False, encoding="utf-8-sig")
print(f"\n[OK] CSV FORMATEADO -> {out_csv_fmt}")

# CSV numérico seguro (para Excel; sin % ni separadores)
out_csv_num = f"{RESULTS_DIR}/Resumen_Rendimiento_Cartera_Por_Modelo_NUMERICO.csv"
df_num_export = df_num.drop(columns=["ROI_raw"]).copy()
df_num_export.rename(columns={
    "% Ganadores":"Winrate_%", "CAGR %":"CAGR_%", "Max DD %":"MaxDD_%"
}, inplace=True)
df_num_export.insert(4, "ROI_%", df_num["ROI_raw"])  # insert ROI_% como número (ej. 188.55)
df_num_export.to_csv(out_csv_num, sep=";", index=False, encoding="utf-8-sig")
print(f"[OK] CSV NUMÉRICO   -> {out_csv_num}")

# XLSX con dos hojas si hay engine disponible
xlsx_engine = next((e for e in ("openpyxl","xlsxwriter") if find_spec(e) is not None), None)
if xlsx_engine:
    out_xlsx = f"{RESULTS_DIR}/Resumen_Rendimiento_Cartera_Por_Modelo.xlsx"
    with pd.ExcelWriter(out_xlsx, engine=xlsx_engine) as w:
        tabla.to_excel(w, index=False, sheet_name="FORMATEADO")
        df_num_export.to_excel(w, index=False, sheet_name="NUMERICO")
    print(f"[OK] XLSX ({xlsx_engine}) -> {out_xlsx}")
else:
    print("Nota: instala 'openpyxl' o 'xlsxwriter' para exportar también a .xlsx.")
