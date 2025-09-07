# -*- coding: utf-8 -*-
"""
Analisis_DA_AllStocks_SIMPLE_CloseOnly.py
-----------------------------------------
- TRAIN  = pooling multi-stock (2010–2019)
- TEST   = OOS mono-stock (2020–2024)

Criterio de acierto (Close-only, con umbral u):
  UP:   max(Close[t+1..t+H]) >= Close[t] * (1 + u)
  DOWN: min(Close[t+1..t+H]) <= Close[t] * (1 - u)

CSV FINAL (solo filas válidas, sin zona muerta, sin ventanas incompletas):
    Symbol;Fecha;y_pred_bin;correct
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# ===================== Configuración ===================== #
DATASET_PATH = "DATA/Dataset_All_Features_Transformado.csv"
FEATS_CSV    = "Resultados/Features_Seleccionadas_GA.csv"
HPARAMS_CSV  = "Resultados/Hiperparametros_por_stock.csv"

OUT_DIR          = "Resultados"
EVAL_SIMPLE_CSV  = os.path.join(OUT_DIR, "DA_Binary_Daily_AllStocks_SIMPLE.csv")

SYMBOLS_TO_TEST = [
    'NVDA','AAPL','AMZN','LRCX','SBUX','REGN','KLAC','BKNG','AMD','VRTX',
    'MAR','CDNS','CAT','INTU','GILD','MU','EBAY','AXP','AMAT','COST','MSFT',
    'ORCL','ADI','MS','NKE'
]

TARGET_COL   = "TARGET_TREND_ANG_15_5"
TRAIN_YEARS  = list(range(2010, 2020))
TEST_YEARS   = [2020, 2021, 2022, 2023, 2024]

# XGBoost (CPU)
N_JOBS = 32
RANDOM_STATE = 42

# Ventana y umbrales de decisión del modelo
H = 5
MODEL_THRESHOLD_UP   = 0.6
MODEL_THRESHOLD_DOWN = 0.4

# Umbral porcentual u (Close-only): 0.01 = 1%, 0.005 = 0.5%, 0.02 = 2%
UMBRAL_MOV_PCT = 0.01  #  1%
# ======================================================== #


# ---------------------- Utilidades I/O --------------------- #
def leer_features_por_stock(path_csv: str) -> Dict[str, List[str]]:
    df = pd.read_csv(path_csv, sep=';')
    m: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        stock = str(row["Stock"]).strip()
        feats = [f.strip() for f in str(row["Features"]).split(",") if f.strip()]
        m[stock] = feats
    return m

def leer_hparams_por_stock(path_csv: str) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(path_csv, sep=';')
    m: Dict[str, Dict[str, float]] = {}
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
# ----------------------------------------------------------- #


# ---------------------- Modelo ----------------------------- #
def configurar_xgb(hp: Dict[str, float]) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        tree_method="hist",
        **hp
    )
# ----------------------------------------------------------- #


# --------- Oportunidad en ventana (CLOSE-ONLY con u) ------- #
def oportunidades_en_ventana_close(C: np.ndarray, t: int, h: int, u: float) -> Tuple[bool, bool]:
    """
    Ventana [t+1..t+h] respecto a P_t = Close(t):
      up_hit   = max(C[t+1..t+h]) >= P_t * (1 + u)
      down_hit = min(C[t+1..t+h]) <= P_t * (1 - u)
    Si la ventana no cabe, devuelve (False, False) para descartar la fila.
    """
    if t + h >= len(C):
        return False, False
    pt = float(C[t])
    win = C[t+1: t+h+1].astype(float)
    up_hit = np.max(win) >= pt * (1.0 + u)
    down_hit = np.min(win) <= pt * (1.0 - u)
    return bool(up_hit), bool(down_hit)

def clasificar_score(score: float) -> Optional[int]:
    """1 (UP) si score>=UP, 0 (DOWN) si score<=DOWN, None si zona muerta."""
    if score >= MODEL_THRESHOLD_UP:
        return 1
    if score <= MODEL_THRESHOLD_DOWN:
        return 0
    return None
# ----------------------------------------------------------- #


# ----------- Evaluación simple (solo lo esencial) ---------- #
def evaluar_symbol_simple(df_test: pd.DataFrame,
                          y_pred_score: np.ndarray,
                          date_col: str = "Fecha",
                          close_col: str = "Close",
                          h: int = H,
                          u: float = UMBRAL_MOV_PCT) -> pd.DataFrame:
    """
    Genera SOLO:
      - Fecha
      - y_pred_bin (1=UP si score>=UP; 0=DOWN si score<=DOWN)
      - correct (1=acierto si algún Close en ventana supera Pt*(1±u); 0=fallo)
    Descarta:
      - Filas en zona muerta (entre umbrales)
      - Filas sin ventana completa (t+h fuera de rango)
    """
    df = df_test.sort_values(date_col).reset_index(drop=True).copy()
    dates = df[date_col].to_numpy()
    C = df[close_col].to_numpy(dtype=float)
    n = len(df)

    rows = []
    for t in range(n):
        score_t = float(y_pred_score[t])
        pred_class = clasificar_score(score_t)
        if pred_class is None or t + h >= n:
            continue  # zona muerta o ventana incompleta → no guardamos

        up_hit, down_hit = oportunidades_en_ventana_close(C, t, h, u)

        if pred_class == 1:          # UP
            correct = 1 if up_hit else 0
        else:                         # DOWN
            correct = 1 if down_hit else 0

        rows.append([dates[t], int(pred_class), int(correct)])

    return pd.DataFrame(rows, columns=["Fecha", "y_pred_bin", "correct"])
# ----------------------------------------------------------- #


# --------------------------- Main -------------------------- #
def main():
    # Chequeo básico de umbrales
    assert 0.0 <= MODEL_THRESHOLD_DOWN < MODEL_THRESHOLD_UP <= 1.0, \
        "Umbrales inválidos: usar 0 <= DOWN < UP <= 1."
    assert 0.0 <= UMBRAL_MOV_PCT < 0.20, "UMBRAL_MOV_PCT fuera de rango razonable (0, 20%)."

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATASET_PATH, sep=';', parse_dates=["Fecha"])
    df = df.sort_values(["Symbol", "Fecha"]).reset_index(drop=True)

    features_map = leer_features_por_stock(FEATS_CSV)
    hparams_map  = leer_hparams_por_stock(HPARAMS_CSV)

    df_train = df[df["Fecha"].dt.year.isin(TRAIN_YEARS)].copy()

    print(f"[SETUP] H={H}, u={UMBRAL_MOV_PCT:.2%}")

    daily_eval_list: List[pd.DataFrame] = []

    for stock in SYMBOLS_TO_TEST:
        if stock not in features_map or stock not in hparams_map:
            warnings.warn(f"[{stock}] sin config de features/hparams; omitido.")
            continue

        feats = [f for f in features_map[stock] if f in df.columns]
        if not feats:
            warnings.warn(f"[{stock}] ninguna feature existe en dataset; omitido.")
            continue

        df_test = df[(df["Symbol"] == stock) & (df["Fecha"].dt.year.isin(TEST_YEARS))].copy()
        if df_test.empty:
            warnings.warn(f"[{stock}] TEST vacío; omitido.")
            continue

        X_train, y_train = df_train[feats], df_train[TARGET_COL]
        X_test = df_test[feats]

        model = configurar_xgb(hparams_map[stock])
        model.fit(X_train, y_train)
        y_pred_score = model.predict(X_test)

        # Evaluación Close-only con umbral porcentual u
        df_eval = evaluar_symbol_simple(df_test, y_pred_score, "Fecha", "Close", H, UMBRAL_MOV_PCT)
        if df_eval.empty:
            print(f"[{stock}] sin filas válidas (zona muerta o ventana incompleta).")
            continue

        df_eval.insert(0, "Symbol", stock)
        daily_eval_list.append(df_eval)

        # Métrica simple: Accuracy y cobertura sobre filas guardadas
        acc = df_eval["correct"].mean()
        cobertura = len(df_eval) / max(1, (len(df_test) - H))  # referencia
        print(f"[{stock}] filas={len(df_eval)} | ACC={acc:.3f} | u={UMBRAL_MOV_PCT:.2%} | H={H} | cobertura≈{cobertura:.1%}")

    if not daily_eval_list:
        print("Nada que guardar.")
        return

    eval_global = pd.concat(daily_eval_list, ignore_index=True)
    eval_global.sort_values(["Symbol", "Fecha"], inplace=True)

    eval_global.to_csv(EVAL_SIMPLE_CSV, sep=';', index=False, date_format="%Y-%m-%d")
    print(f"\n[OK] Guardado: {os.path.abspath(EVAL_SIMPLE_CSV)}")
    print("Columnas: Symbol; Fecha; y_pred_bin; correct")
    print(f"Regla (Close-only): UP si max Close_win >= Pt*(1+u) | DOWN si min Close_win <= Pt*(1-u), u={UMBRAL_MOV_PCT:.2%}, H={H} días.")

if __name__ == "__main__":
    main()
