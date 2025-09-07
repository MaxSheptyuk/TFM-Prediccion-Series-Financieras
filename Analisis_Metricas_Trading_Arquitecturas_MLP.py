# Resumen_Metricas_Trading_Arquitecturas.py (5 años, cartera)
# -----------------------------------------------------------
# - Capital inicial FIJO: N_BASE_SYMBOLS * CAPITAL_X_STOCK
# - Capital final desde equity global = capital_inicial + cumsum(Profit diario)
# - Drawdown calculado sobre esa equity global
# - Comparación limpia entre arquitecturas

import os, glob
import pandas as pd
import numpy as np

RESULTS_DIR = "Resultados"
CAPITAL_X_STOCK = 10_000.0
N_BASE_SYMBOLS = 25  # baseline fijo de la cartera

def _years_between(d0: pd.Timestamp, d1: pd.Timestamp) -> float:
    return max((d1 - d0).days, 1) / 365.25

def calcular_metricas(path_csv: str) -> dict:
    df = pd.read_csv(path_csv, sep=";", parse_dates=["Fecha"])
    req = {"Symbol","Fecha","Accion","Profit","Capital_Actual"}
    if not req.issubset(df.columns):
        raise ValueError(f"{os.path.basename(path_csv)} carece de columnas: {req}")

    df = df.sort_values(["Symbol","Fecha"]).reset_index(drop=True)

    # --- SELLs deduplicados (por si hay duplicidad de escritura)
    df_trades = df[df["Accion"].astype(str).str.startswith("SELL", na=False)].copy()
    dedup_keys = [c for c in ["Symbol","Fecha","Accion","Precio","Profit"] if c in df_trades.columns]
    if dedup_keys:
        df_trades = df_trades.drop_duplicates(subset=dedup_keys, keep="last")

    # --- Baseline FIJO
    symbols_cubiertos = df["Symbol"].nunique()
    capital_inicial = N_BASE_SYMBOLS * CAPITAL_X_STOCK

    # --- Profit diario agregado (cartera)
    daily = (df_trades.groupby("Fecha", as_index=False)["Profit"].sum()
                      .sort_values("Fecha").reset_index(drop=True))

    if len(daily):
        equity = capital_inicial + np.cumsum(daily["Profit"].values)
        fechas = daily["Fecha"].values
        capital_final = float(equity[-1])
    else:
        equity = np.array([capital_inicial])
        fechas = np.array([df["Fecha"].min()])
        capital_final = float(capital_inicial)

    capital_ganado = capital_final - capital_inicial
    roi_pct = 100.0 * capital_ganado / capital_inicial

    # --- Drawdown sobre equity global
    running_max = np.maximum.accumulate(equity)
    dd_series = running_max - equity
    dd_pct_series = np.divide(dd_series, running_max, out=np.zeros_like(dd_series, dtype=float), where=running_max>0)

    i_trough = int(np.argmax(dd_series))
    max_dd = float(dd_series[i_trough])
    max_dd_pct = float(dd_pct_series[i_trough]) * 100.0
    trough_date = fechas[i_trough] if len(fechas) else pd.NaT
    i_peak = int(np.argmax(equity[:i_trough+1])) if i_trough > 0 else 0
    peak_date = fechas[i_peak] if len(fechas) else pd.NaT

    # fecha de recuperación (si vuelve a máximo previo)
    recovery_date = pd.NaT
    if max_dd > 0 and len(equity) > i_trough:
        for j in range(i_trough+1, len(equity)):
            if equity[j] >= equity[i_peak]:
                recovery_date = fechas[j]
                break

    # --- Métricas de trades
    trades_total = len(df_trades)
    win_all = int((df_trades["Profit"] > 0).sum())
    winrate_all = 100.0 * win_all / trades_total if trades_total else 0.0

    valid = df_trades[df_trades["Profit"] != 0]
    winrate_excl0 = 100.0 * (valid["Profit"] > 0).mean() if len(valid) else 0.0

    expectancy = float(df_trades["Profit"].mean() if trades_total else 0.0)
    g = df_trades[df_trades["Profit"] > 0]["Profit"]
    l = df_trades[df_trades["Profit"] < 0]["Profit"]
    payoff = float(g.mean() / abs(l.mean())) if (len(g) > 0 and len(l) > 0 and abs(l.mean()) > 1e-12) else np.nan

    anios = _years_between(df["Fecha"].min(), df["Fecha"].max()) if len(df) else np.nan
    cagr = (((capital_final / capital_inicial) ** (1.0 / anios) - 1.0) * 100.0) if (capital_inicial > 0 and anios > 0) else np.nan

    return {
        "Capital_inicial": round(capital_inicial, 2),
        "Capital_final": round(capital_final, 2),
        "Capital_ganado": round(capital_ganado, 2),
        "ROI_%": round(roi_pct, 2) if not np.isnan(roi_pct) else np.nan,
        "Max_Drawdown": round(max_dd, 2),
        "Max_Drawdown_%": round(max_dd_pct, 2),
        "DD_Peak_Date": peak_date,
        "DD_Trough_Date": trough_date,
        "DD_Recovery_Date": recovery_date,
        "%Ganadores_excl0": round(winrate_excl0, 2),
        "%Ganadores_todos": round(winrate_all, 2),
        "Expectancy": round(expectancy, 2),
        "Payoff": round(payoff, 2) if not np.isnan(payoff) else np.nan,
        "CAGR_%": round(cagr, 2) if not np.isnan(cagr) else np.nan,
        "Trades": trades_total,
        "Symbols_cubiertos": symbols_cubiertos
    }

def main():
    files = glob.glob(os.path.join(RESULTS_DIR, "Trading_Log_AllStocks_MLP_OHLCV_*.csv"))
    if not files:
        print("[ERROR] No se encontraron logs.")
        return

    rows = []
    for f in files:
        arch = os.path.basename(f).replace("Trading_Log_AllStocks_MLP_OHLCV_", "").replace(".csv", "")
        print(f"Procesando {arch}...")
        m = calcular_metricas(f)
        m["Arquitectura"] = arch
        rows.append(m)

    out = pd.DataFrame(rows).sort_values("Arquitectura").reset_index(drop=True)
    out_path = os.path.join(RESULTS_DIR, "Resumen_Metricas_Trading_Arquitecturas.csv")
    out.to_csv(out_path, sep=";", index=False)
    print(f"\n[OK] Guardado: {out_path}")
    print(out)

if __name__ == "__main__":
    main()
