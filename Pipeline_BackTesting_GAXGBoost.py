# Pipeline_BackTesting_GAXGBoost_MultiTarget.py
# ----------------------------------------------
# Para cada TARGET en TARGETS:
#   - Usa Features_Seleccionadas_GA_{TARGET}.csv y Hiperparametros_{TARGET}.csv
#   - Entrena XGB por stock (TRAIN 2010–2019, pooling multi-stock)
#   - Predice OOS 2020–2024 por stock
#   - Ejecuta Backtester y guarda Trading_Log_AllStocks_{TARGET}.csv

import os
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from BackTesting.Backtester import Backtester

# ============ CONFIG ============ #
DATASET_PATH = "DATA/Dataset_All_Features_Transformado.csv"
RESULTS_DIR  = "Resultados"

SYMBOLS_TO_TEST = [
    'NVDA','AAPL','AMZN','LRCX','SBUX','REGN','KLAC','BKNG','AMD','VRTX',
    'MAR','CDNS','CAT','INTU','GILD','MU','EBAY','AXP','AMAT','COST','MSFT',
    'ORCL','ADI','MS','NKE'
]

# Lista de TARGETS a procesar
TARGETS = [
    # W = 15
    "TARGET_TREND_ANG_15_5",
    "TARGET_TREND_ANG_15_8",
    "TARGET_TREND_ANG_15_10",
    "TARGET_TREND_ANG_15_12",
    "TARGET_TREND_ANG_15_15",

    # W = 10
    "TARGET_TREND_ANG_10_5",
    "TARGET_TREND_ANG_10_8",
    "TARGET_TREND_ANG_10_10",
    "TARGET_TREND_ANG_10_12",
    "TARGET_TREND_ANG_10_15",

    # W = 5
    "TARGET_TREND_ANG_5_5",
    "TARGET_TREND_ANG_5_8",
    "TARGET_TREND_ANG_5_10",
    "TARGET_TREND_ANG_5_12",
    "TARGET_TREND_ANG_5_15",
]


TRAIN_YEARS     = list(range(2010, 2020))        # 2010–2019 (pool multi-stock)
OOS_TEST_YEARS  = [2020, 2021, 2022, 2023, 2024] # 2020–2024 (mono-stock)

# Backtester (igual que en todos los experimentos) 
THRESHOLD_BUY      = 0.51
PACIENCIA_MAX_DIAS = 5
CAPITAL_INICIAL    = 10000

TP_PCT             = 0.015
SL_PCT             = 0.03

# XGBoost siempre CPU
N_JOBS = 32
RANDOM_STATE = 42
# ================================ #

def leer_features_por_stock(path_csv: str) -> dict:
    if not os.path.isfile(path_csv):
        raise FileNotFoundError(f"No se encontró {path_csv}")
    df = pd.read_csv(path_csv, sep=';')
    if not {"Stock", "Features"}.issubset(df.columns):
        raise ValueError("CSV de features debe tener columnas 'Stock' y 'Features'.")
    m = {}
    for _, row in df.iterrows():
        stock = str(row["Stock"]).strip()
        feats = [f.strip() for f in str(row["Features"]).split(",") if f.strip()]
        m[stock] = feats
    return m

def leer_hparams_por_stock(path_csv: str) -> dict:
    if not os.path.isfile(path_csv):
        raise FileNotFoundError(f"No se encontró {path_csv}")
    df = pd.read_csv(path_csv, sep=';')
    req = ["Stock","n_estimators","max_depth","learning_rate","subsample","colsample_bytree"]
    if not set(req).issubset(df.columns):
        raise ValueError(f"CSV de hiperparámetros debe tener columnas: {req}")
    m = {}
    for _, row in df.iterrows():
        stock = str(row["Stock"]).strip()
        m[stock] = {
            "n_estimators":     int(round(row["n_estimators"])),
            "max_depth":        int(round(row["max_depth"])),
            "learning_rate":    float(row["learning_rate"]),
            "subsample":        float(row["subsample"]),
            "colsample_bytree": float(row["colsample_bytree"]),
        }
    return m

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
        n_estimators=hp["n_estimators"],
        max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"],
        subsample=hp["subsample"],
        colsample_bytree=hp["colsample_bytree"],
    )

def backtest_signal(df_test_symbol: pd.DataFrame, y_pred: np.ndarray, symbol: str) -> pd.DataFrame:
    bt = Backtester(
        threshold_buy=THRESHOLD_BUY,
        paciencia_max_dias=PACIENCIA_MAX_DIAS,
        capital_inicial=CAPITAL_INICIAL,
        tp_pct=TP_PCT,
        sl_pct=SL_PCT,
        save_trades=False
    )
    res = bt.run(df_test_symbol, y_pred, symbol=symbol)
    trades_df = res["trade_log"].copy()
    trades_df.insert(0, "Symbol", symbol)
    trades_df.insert(1, "OrdenN", range(1, len(trades_df) + 1))
    print(f"[{symbol}] Trades={res.get('num_trades', 0)} | ROI={res.get('roi', np.nan):.2f}%")
    return trades_df

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Carga dataset base 1 sola vez
    df = pd.read_csv(DATASET_PATH, sep=';', parse_dates=["Fecha"])
    df = df.sort_values(["Symbol", "Fecha"]).reset_index(drop=True)

    # Validación: que el dataset tenga todos los TARGETS solicitados
    missing_targets = [t for t in TARGETS if t not in df.columns]
    if missing_targets:
        raise ValueError(f"[ERROR] Faltan targets en el dataset: {missing_targets}")

    # ======= Bucle por TARGET ======= #
    for TARGET_COL in TARGETS:
        print("\n" + "="*100)
        print(f"=== Backtesting para TARGET: {TARGET_COL} ===")
        print("="*100)

        feats_csv   = os.path.join(RESULTS_DIR, f"Features_Seleccionadas_GA_{TARGET_COL}.csv")
        hparams_csv = os.path.join(RESULTS_DIR, f"Hiperparametros_{TARGET_COL}.csv")
        out_log     = os.path.join(RESULTS_DIR, f"Trading_Log_AllStocks_{TARGET_COL}.csv")

        # Cargar mapas de features e hiperparámetros
        try:
            features_map = leer_features_por_stock(feats_csv)
        except Exception as e:
            print(f"[ERROR] No se pueden leer features GA para {TARGET_COL}: {e}")
            print("[INFO] Se omite este target.\n")
            continue

        try:
            hparams_map = leer_hparams_por_stock(hparams_csv)
        except Exception as e:
            print(f"[ERROR] No se pueden leer hiperparámetros para {TARGET_COL}: {e}")
            print("[INFO] Se omite este target.\n")
            continue

        logs = []

        # ------ Bucle por stock ------
        for stock in SYMBOLS_TO_TEST:

            if stock not in features_map:
                warnings.warn(f"[{stock}] No hay features en {os.path.basename(feats_csv)}. Se omite.")
                continue
            if stock not in hparams_map:
                warnings.warn(f"[{stock}] No hay hiperparámetros en {os.path.basename(hparams_csv)}. Se omite.")
                continue

            feats = [f for f in features_map[stock] if f in df.columns]
            if not feats:
                warnings.warn(f"[{stock}] Ninguna feature listada existe en dataset. Se omite.")
                continue

            df_train = split_por_anios(df, TRAIN_YEARS)                  # pool multi-stock
            df_test  = split_por_anios(df, OOS_TEST_YEARS, symbol=stock) # mono-stock

            if df_train.empty or df_test.empty:
                warnings.warn(f"[{stock}] Train o Test vacío. Se omite.")
                continue

            X_train = df_train[feats]
            y_train = df_train[TARGET_COL]
            X_test  = df_test[feats]

            model = configurar_xgb(hparams_map[stock])
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            except Exception as e:
                warnings.warn(f"[{stock}] Fallo entrenando/prediciendo: {e}. Se omite.")
                continue

            # Backtest y log
            try:
                trades_df = backtest_signal(df_test, y_pred, stock)
                logs.append(trades_df)
            except Exception as e:
                warnings.warn(f"[{stock}] Fallo en backtesting: {e}. Se omite.")
                continue

        # ------ Guardar log unificado por TARGET ------
        if logs:
            log_global = pd.concat(logs, ignore_index=True)
            # Orden lógico por símbolo y fecha de entrada de la operación si viene como 'Fecha'
            ord_cols = [c for c in ["Symbol","Fecha"] if c in log_global.columns]
            if ord_cols:
                log_global = log_global.sort_values(ord_cols).reset_index(drop=True)
            log_global.to_csv(out_log, sep=';', index=False, date_format="%Y-%m-%d")
            print(f"[OK] Guardado: {os.path.abspath(out_log)}")
        else:
            print(f"[WARN] {TARGET_COL}: no se generaron operaciones. No se guarda log.")

    print("\nProceso de backtesting multi-target COMPLETADO ✔")

if __name__ == "__main__":
    main()
