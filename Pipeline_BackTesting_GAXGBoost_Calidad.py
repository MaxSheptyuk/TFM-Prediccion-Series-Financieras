# Pipeline_BackTesting_GAXGBoost_Calidad_All.py
# ------------------------------------------------
# Evalúa calidad en 25 activos y guarda métricas extendidas
# ------------------------------------------------

import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from BackTesting.BacktesterCalidadML import BacktesterCalidadML, CalidadConfig

DATASET_PATH = "DATA/Dataset_All_Features_Transformado.csv"
FEATS_CSV    = "Resultados/Features_Seleccionadas_GA.csv"
HPARAMS_CSV  = "Resultados/Hiperparametros_por_stock.csv"

OUT_DIR      = "Resultados"
CALIDAD_SYM  = os.path.join(OUT_DIR, "Calidad_Summary_BySymbol.csv")
CALIDAD_GLB  = os.path.join(OUT_DIR, "Calidad_Summary_Global.csv")

SYMBOLS_TO_TEST = [
    'NVDA','AAPL','AMZN','LRCX','SBUX','REGN','KLAC','BKNG','AMD','VRTX',
    'MAR','CDNS','CAT','INTU','GILD','MU','EBAY','AXP','AMAT','COST','MSFT',
    'ORCL','ADI','MS','NKE'
]

TARGET_COL     = "TARGET_TREND_ANG_15_5"
TRAIN_YEARS    = list(range(2010, 2020))
OOS_TEST_YEARS = [2020, 2021, 2022, 2023, 2024]

PACIENCIA_MAX_DIAS = 5
THRESHOLD_BUY = 0.51

N_JOBS = 8
RANDOM_STATE = 42

def leer_features_por_stock(path_csv: str) -> dict:
    df = pd.read_csv(path_csv, sep=';')
    out = {}
    for _, r in df.iterrows():
        out[str(r["Stock"]).strip()] = [f.strip() for f in str(r["Features"]).split(",") if f.strip()]
    return out

def leer_hparams_por_stock(path_csv: str) -> dict:
    df = pd.read_csv(path_csv, sep=';')
    out = {}
    for _, r in df.iterrows():
        out[str(r["Stock"]).strip()] = dict(
            n_estimators=int(round(r["n_estimators"])),
            max_depth=int(round(r["max_depth"])),
            learning_rate=float(r["learning_rate"]),
            subsample=float(r["subsample"]),
            colsample_bytree=float(r["colsample_bytree"]),
        )
    return out

def split_por_anios(df: pd.DataFrame, years: list, symbol: str | None = None) -> pd.DataFrame:
    out = df[df["Fecha"].dt.year.isin(years)]
    if symbol is not None:
        out = out[out["Symbol"] == symbol]
    return out

def configurar_xgb(hp: dict) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        tree_method="hist",
        **hp
    )

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = (pd.read_csv(DATASET_PATH, sep=';', parse_dates=["Fecha"])
            .sort_values(["Symbol","Fecha"])
            .reset_index(drop=True))

    feats_map = leer_features_por_stock(FEATS_CSV)
    hps_map   = leer_hparams_por_stock(HPARAMS_CSV)

    calidad = BacktesterCalidadML(CalidadConfig(
        horizon=PACIENCIA_MAX_DIAS,
        threshold_buy=THRESHOLD_BUY,
        signal_colname="y_hat",
        price_colname="Close",
        date_colname="Fecha",
        symbol_colname="Symbol",
    ))

    sym_summaries = []

    for stock in SYMBOLS_TO_TEST:
        if stock not in feats_map or stock not in hps_map:
            print(f"[WARN] {stock}: faltan features o hparams. Se omite.")
            continue

        feats = [f for f in feats_map[stock] if f in df.columns]
        if not feats:
            print(f"[WARN] {stock}: ninguna feature válida en dataset. Se omite.")
            continue

        df_train = split_por_anios(df, TRAIN_YEARS)
        df_test  = split_por_anios(df, OOS_TEST_YEARS, symbol=stock)
        if df_train.empty or df_test.empty:
            print(f"[WARN] {stock}: train/test vacío. Se omite.")
            continue

        X_train = df_train[feats]
        y_train = df_train[TARGET_COL].astype(float)
        X_test  = df_test[feats]

        model = configurar_xgb(hps_map[stock])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        df_test = df_test.copy()
        df_test["y_hat"] = y_pred
        res = calidad.run(df_test[["Fecha","Symbol","Close","y_hat"]], y_pred, symbol=stock)

        m = res["metrics"]
        if not m:
            continue

        sym_summaries.append(pd.DataFrame([{
            "Symbol": stock,
            "n_trades": m.get("n_trades", 0),
            "mean_final_ret": m.get("mean_final_ret", np.nan),
            "median_final_ret": m.get("median_final_ret", np.nan),
            "mean_avg_pos_path_ret": m.get("mean_avg_pos_path_ret", np.nan),
            "total_sum_pos_path_ret": m.get("total_sum_pos_path_ret", np.nan),
            "total_pos_days": m.get("total_pos_days", np.nan),
            "hit_ratio": m.get("hit_ratio", np.nan),
            "IC_final": m.get("IC_final", np.nan),
            "Rank_IC_final": m.get("Rank_IC_final", np.nan),
        }]))

        print(f"[{stock}] Trades={m.get('n_trades',0)} "
              f"| IC_final={m.get('IC_final',np.nan):.3f} "
              f"| Rank-IC_final={m.get('Rank_IC_final',np.nan):.3f} "
              f"| Hit%={100*m.get('hit_ratio',0):.1f}%")

    # ------- Guardar resúmenes -------
    if sym_summaries:
        bysym = pd.concat(sym_summaries, ignore_index=True)
        bysym.to_csv(CALIDAD_SYM, sep=';', index=False)

        # Promedio global de métricas numéricas
        glb = bysym.drop(columns=["Symbol"]).mean(numeric_only=True).to_frame("mean").reset_index()
        glb.rename(columns={"index":"metric"}, inplace=True)
        glb.to_csv(CALIDAD_GLB, sep=';', index=False)

        print(f"[OK] Resúmenes guardados en:\n - {os.path.abspath(CALIDAD_SYM)}\n - {os.path.abspath(CALIDAD_GLB)}")
    else:
        print("[WARN] No hubo resúmenes por símbolo.")

if __name__ == "__main__":
    main()
