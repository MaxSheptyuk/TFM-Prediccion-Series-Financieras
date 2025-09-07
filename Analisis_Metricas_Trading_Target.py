# Analisis_Metricas_Trading_Target.py  — versión consistente (tabla que cuadra con la equity)
import os, glob
import pandas as pd
import numpy as np

RESULTS_DIR = "Resultados"

# === Baseline fijo de cartera ===
CAPITAL_X_STOCK = 10_000.0
SYMBOLS_CARTERA = [
    'NVDA','AAPL','AMZN','LRCX','SBUX','REGN','KLAC','BKNG','AMD','VRTX',
    'MAR','CDNS','CAT','INTU','GILD','MU','EBAY','AXP','AMAT','COST','MSFT',
    'ORCL','ADI','MS','NKE'
]
N_BASE_SYMBOLS = len(SYMBOLS_CARTERA)
CAPITAL_INICIAL = CAPITAL_X_STOCK * N_BASE_SYMBOLS

def _years_between(d0: pd.Timestamp, d1: pd.Timestamp) -> float:
    """Años (con decimales) entre dos fechas."""
    return max((d1 - d0).days, 1) / 365.25

def _equity_from_profit(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Equity global de cartera: capital_inicial + cumsum(Profit diario de SELL).
    Devuelve (equity_values, fechas_numpy).
    """
    # Solo cierres (SELL*), y filtramos símbolos de la cartera
    df = df[df["Symbol"].isin(SYMBOLS_CARTERA)].copy()
    df_trades = df[df["Accion"].astype(str).str.startswith("SELL", na=False)].copy()

    # De-dup defensivo
    dedup_keys = [c for c in ["Symbol","Fecha","Accion","Precio","Profit"] if c in df_trades.columns]
    if dedup_keys:
        df_trades = df_trades.drop_duplicates(subset=dedup_keys, keep="last")

    # Profit diario agregado a nivel cartera
    if len(df_trades):
        daily = (df_trades.groupby("Fecha", as_index=False)["Profit"].sum()
                          .sort_values("Fecha").reset_index(drop=True))
        equity = CAPITAL_INICIAL + np.cumsum(daily["Profit"].values)
        fechas = daily["Fecha"].values
    else:
        equity = np.array([CAPITAL_INICIAL])
        fechas = np.array([df["Fecha"].min()]) if len(df) else np.array([])
    return equity, fechas

def calcular_metricas(path_csv: str) -> dict:
    df = pd.read_csv(path_csv, sep=";", parse_dates=["Fecha"])
    req = {"Symbol","Fecha","Accion","Profit","Capital_Actual"}
    if not req.issubset(df.columns):
        raise ValueError(f"{os.path.basename(path_csv)} carece de columnas: {req}")

    # Orden básico
    df = df.sort_values(["Symbol","Fecha"]).reset_index(drop=True)

    # === Equity consistente (la misma que usarías para el gráfico correcto)
    equity, fechas = _equity_from_profit(df)

    # Capitales y ROI
    capital_inicial = CAPITAL_INICIAL                     # FIJO
    capital_final   = float(equity[-1])                   # desde equity
    capital_ganado  = capital_final - capital_inicial
    roi_pct         = 100.0 * capital_ganado / capital_inicial if capital_inicial > 0 else np.nan

    # Drawdown (sobre equity)
    run_max = np.maximum.accumulate(equity)
    dd = run_max - equity
    dd_pct = np.divide(dd, run_max, out=np.zeros_like(dd, dtype=float), where=run_max>0)

    i_trough   = int(np.argmax(dd)) if len(dd) else 0
    max_dd     = float(dd[i_trough]) if len(dd) else 0.0
    max_dd_pct = float(dd_pct[i_trough]) * 100.0 if len(dd) else 0.0
    trough_dt  = (pd.to_datetime(fechas[i_trough]) if len(fechas) else pd.NaT)

    i_peak = int(np.argmax(equity[:i_trough+1])) if i_trough > 0 else 0
    peak_dt = (pd.to_datetime(fechas[i_peak]) if len(fechas) else pd.NaT)

    recovery_dt = pd.NaT
    if max_dd > 0 and len(equity) > i_trough:
        peak_val = equity[i_peak]
        for j in range(i_trough+1, len(equity)):
            if equity[j] >= peak_val:
                recovery_dt = pd.to_datetime(fechas[j]); break

    # Trades y winrates (sobre SELL)
    df_trades = df[df["Symbol"].isin(SYMBOLS_CARTERA) &
                   df["Accion"].astype(str).str.startswith("SELL", na=False)].copy()
    trades_total = len(df_trades)
    win_all = int((df_trades["Profit"] > 0).sum())
    winrate_all = 100.0 * win_all / trades_total if trades_total else 0.0

    valid = df_trades[df_trades["Profit"] != 0]
    winrate_excl0 = 100.0 * (valid["Profit"] > 0).mean() if len(valid) else 0.0

    expectancy = float(df_trades["Profit"].mean() if trades_total else 0.0)
    g = df_trades[df_trades["Profit"] > 0]["Profit"]
    l = df_trades[df_trades["Profit"] < 0]["Profit"]
    payoff = float(g.mean() / abs(l.mean())) if (len(g) > 0 and len(l) > 0 and abs(l.mean()) > 1e-12) else np.nan

    # CAGR (mismo periodo de la equity)
    if len(df):
        anios = _years_between(df["Fecha"].min(), df["Fecha"].max())
    else:
        anios = np.nan
    cagr = (((capital_final / capital_inicial) ** (1.0 / anios) - 1.0) * 100.0) if (capital_inicial > 0 and anios > 0) else np.nan

    return {
        "Capital_inicial": round(capital_inicial, 2),
        "Capital_final": round(capital_final, 2),
        "Capital_ganado": round(capital_ganado, 2),
        "ROI_%": round(roi_pct, 2) if not np.isnan(roi_pct) else np.nan,
        "Max_Drawdown": round(max_dd, 2),
        "Max_Drawdown_%": round(max_dd_pct, 2),
        "DD_Peak_Date": peak_dt if pd.notna(peak_dt) else "",
        "DD_Trough_Date": trough_dt if pd.notna(trough_dt) else "",
        "DD_Recovery_Date": recovery_dt if pd.notna(recovery_dt) else "",
        "%Ganadores_excl0": round(winrate_excl0, 2),
        "%Ganadores_todos": round(winrate_all, 2),
        "Expectancy": round(expectancy, 2),
        "Payoff": round(payoff, 2) if not np.isnan(payoff) else np.nan,
        "CAGR_%": round(cagr, 2) if not np.isnan(cagr) else np.nan,
        "Trades": trades_total,
        # Info de cobertura (meramente informativa)
        "Symbols_cubiertos": int(df[df["Symbol"].isin(SYMBOLS_CARTERA)]["Symbol"].nunique())
    }

def main():
    # ⚠️ Ajusta el patrón si solo quieres ciertos targets (p.ej., ANG_* o W5_*):
    pattern = os.path.join(RESULTS_DIR, "Trading_Log_AllStocks_TARGET_*.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"[ERROR] No se encontraron logs con patrón: {pattern}")
        return

    rows = []
    for f in sorted(files):
        target = os.path.basename(f).replace("Trading_Log_AllStocks_", "").replace(".csv", "")
        print(f"Procesando {target}...")
        m = calcular_metricas(f)
        m["Target"] = target
        rows.append(m)

    out = pd.DataFrame(rows).sort_values("Target").reset_index(drop=True)

    # Sanity check opcional: si detectas columnas antiguas, puedes compararlas aquí.
    # Ej.: imprimir capital_final por método 'último Capital_Actual' (NO usar en tabla final).
    # df_aux = pd.read_csv(files[0], sep=";", parse_dates=["Fecha"])
    # ... (omitir para no liar)

    out_path = os.path.join(RESULTS_DIR, "Resumen_Metricas_Trading.csv")
    out.to_csv(out_path, sep=";", index=False)
    print(f"\n[OK] Guardado: {out_path}")
    print(out)

if __name__ == "__main__":
    main()
