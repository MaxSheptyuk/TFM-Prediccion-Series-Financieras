# Pipeline_BackTesting_XGB_AllFeatures_H5.py
# --------------------------------------
# XGBoost por ACTIVO sin GA ni tuning, H=5 y threshold=0.60
# Usa TODAS las features excepto:
#   - columnas de control (Symbol, Fecha), OHLCV/Volumen,
#   - cualquier TARGET* (incluida la objetivo) y EMA_*.
# TRAIN: 2010–2019 (pooling multi-stock); TEST: 2020–2024 (mono-stock).
# Backtester: H=5, TP=1.5%, SL=3%, threshold=0.60
# --------------------------------------

import os
import warnings
import argparse
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from BackTesting.Backtester import Backtester

# ============== CONFIGURACIÓN POR DEFECTO ==============
DATASET_PATH = "DATA/Dataset_All_Features_Transformado.csv"

OUT_DIR    = "Resultados"
TRADES_LOG = os.path.join(OUT_DIR, "Trading_Log_AllStocks_XGB_ALLFEATS.csv")
PREDS_DIR  = os.path.join(OUT_DIR, "preds_xgb_allfeats_h5")

SYMBOLS_TO_TEST = [
    'NVDA','AAPL','AMZN','LRCX','SBUX','REGN','KLAC','BKNG','AMD','VRTX',
    'MAR','CDNS','CAT','INTU','GILD','MU','EBAY','AXP','AMAT','COST','MSFT',
    'ORCL','ADI','MS','NKE'
]

# Columna objetivo por defecto (puedes sobreescribir con --target)
TARGET_COL_DEFAULT = "TARGET_TREND_ANG_10_5"

TRAIN_YEARS    = list(range(2010, 2020))
OOS_TEST_YEARS = [2020, 2021, 2022, 2023, 2024]

# Backtester (H=5)
THRESHOLD_BUY      = 0.51  # Umbral de señal de compra
PACIENCIA_MAX_DIAS = 5
CAPITAL_INICIAL    = 10000
TP_PCT             = 0.015  # 1.5%
SL_PCT             = 0.03   # 3%

# XGBoost (sobrio, sin tuning)
XGB_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=200,
    max_depth=4,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)


# Columnas a excluir de las features
NON_FEATURE_COLS = {
    "Symbol", "Fecha",
    "Open", "High", "Low", "Close", "Adj Close", "AdjClose",
    "Volume", "Volumen"
}
EXCLUDE_PREFIXES = ("TARGET_", "EMA_")  # cualquier columna que empiece así


# ============== UTILIDADES ==============

def construir_lista_features(df: pd.DataFrame, target_col: str) -> list[str]:
    """Construye la lista de features: todas menos control/OHLCV, cualquier TARGET*, EMA_*, y la propia target."""
    feats = []
    cols = list(df.columns)
    for c in cols:
        if c == target_col:
            continue
        if c in NON_FEATURE_COLS:
            continue
        if any(c.startswith(pfx) for pfx in EXCLUDE_PREFIXES):
            continue
        feats.append(c)
    if len(feats) == 0:
        raise ValueError("La lista de features quedó vacía tras aplicar exclusiones.")
    return feats

def split_por_anios(df: pd.DataFrame, years: list, symbol: str | None = None) -> pd.DataFrame:
    out = df[df["Fecha"].dt.year.isin(years)]
    if symbol is not None:
        out = out[out["Symbol"] == symbol]
    return out


# ============== PIPELINE ==============

def main(target_col: str):

    # Sanidad: paciencia=H
    assert PACIENCIA_MAX_DIAS == 5, "PACIENCIA_MAX_DIAS debe ser 5 para H=5."

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PREDS_DIR, exist_ok=True)

    # Cargar dataset
    df = pd.read_csv(DATASET_PATH, sep=';', parse_dates=["Fecha"])
    df = df.sort_values(["Symbol", "Fecha"]).reset_index(drop=True)

    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en el dataset.")

    # Construir features (excluye TARGET*, EMA_*, OHLCV/control y la propia target)
    feature_cols = construir_lista_features(df, target_col)
    print(f"[INFO] Target (H=5): {target_col}")
    print(f"[INFO] Nº de features seleccionadas: {len(feature_cols)}")

    logs_todos = []

    for stock in SYMBOLS_TO_TEST:
        print(f"\n=== Procesando {stock} (XGB ALL-FEATS, H=5, thr={THRESHOLD_BUY:.2f}) ===")

        df_train = split_por_anios(df, TRAIN_YEARS)                  # pooling multi-stock
        df_test  = split_por_anios(df, OOS_TEST_YEARS, symbol=stock) # test mono-stock

        if df_train.empty or df_test.empty:
            warnings.warn(f"[{stock}] Train o Test vacío. Se omite.")
            continue

        # Preparar matrices
        X_train = df_train[feature_cols]
        y_train = df_train[target_col].astype(float)
        X_test  = df_test[feature_cols]

        # Entrenar
        print(f"[{stock}] Entrenando XGB... (X_train: {X_train.shape})")
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train)

        # Predecir
        print(f"[{stock}] Prediciendo en TEST... (X_test: {X_test.shape})")
        y_pred = model.predict(X_test)

        # Guardar preds
        preds_path = os.path.join(PREDS_DIR, f"XGB_ALLFEATS_H5_{stock}.csv")
        pd.DataFrame({"Date": df_test["Fecha"].values, "y_hat": y_pred}) \
          .to_csv(preds_path, sep=';', index=False, date_format="%Y-%m-%d")

        # Backtest
        bt = Backtester(
            threshold_buy=THRESHOLD_BUY,           
            paciencia_max_dias=PACIENCIA_MAX_DIAS, 
            capital_inicial=CAPITAL_INICIAL,
            tp_pct=TP_PCT,
            sl_pct=SL_PCT,
            save_trades=False
        )
        bt_result = bt.run(df_test, y_pred, symbol=stock)

        trades_df = bt_result.get("trade_log", pd.DataFrame()).copy()
        if not trades_df.empty:
            trades_df.insert(0, "Symbol", stock)
            trades_df.insert(1, "OrdenN", range(1, len(trades_df) + 1))
            logs_todos.append(trades_df)

        print(f"[{stock}] Trades={bt_result.get('num_trades', 0)} | ROI={bt_result.get('roi', np.nan):.2f}%")

    # Log unificado
    if logs_todos:
        log_global = pd.concat(logs_todos, ignore_index=True)
        log_global = log_global.sort_values(["Symbol", "Fecha"]).reset_index(drop=True)
        log_global.to_csv(TRADES_LOG, sep=';', index=False, date_format="%Y-%m-%d")
        print(f"\n[OK] Log de trading unificado guardado en: {os.path.abspath(TRADES_LOG)}")
    else:
        print("\n[WARN] No se generaron operaciones. Log no guardado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGB All-Features H=5 Backtesting")
    parser.add_argument(
        "--target",
        type=str,
        default=TARGET_COL_DEFAULT,
        help=f"Nombre de la columna objetivo (por defecto: {TARGET_COL_DEFAULT})"
    )
    args = parser.parse_args()
    main(target_col=args.target)
