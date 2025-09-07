# Pipeline_Ajuste_Hiperparametros_MultiTarget.py
# ----------------------------------------------
# RandomizedSearchCV con pooling multi-stock en TRAIN y TEST por stock.
# Genera un archivo por target con los mejores hiperparámetros por stock:
# Resultados/Hiperparametros_{TARGET}.csv

import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

# --------------- CONFIG --------------- #
DATASET_PATH  = "DATA/Dataset_All_Features_Transformado.csv"
RESULTS_DIR   = "Resultados"

SYMBOLS_TO_TEST = [
    'NVDA','AAPL','AMZN','LRCX','SBUX','REGN','KLAC','BKNG','AMD','VRTX',
    'MAR','CDNS','CAT','INTU','GILD','MU','EBAY','AXP','AMAT','COST','MSFT',
    'ORCL','ADI','MS','NKE'
]

# Lista de TARGETS
# TARGETS = [
#     'TARGET_TREND_ANG_15_15',
#     'TARGET_TREND_ANG_15_10',
#     'TARGET_TREND_ANG_15_5',
#     'TARGET_TREND_ANG_10_15',
#     'TARGET_TREND_ANG_10_10',
#     'TARGET_TREND_ANG_10_5',
#     'TARGET_TREND_ANG_5_15',
#     'TARGET_TREND_ANG_5_10',
#     'TARGET_TREND_ANG_5_5'
# ]

TARGETS = [
    'TARGET_TREND_ANG_15_12',
    'TARGET_TREND_ANG_15_8',
    
    'TARGET_TREND_ANG_10_12',
    'TARGET_TREND_ANG_10_8',
    
    'TARGET_TREND_ANG_5_12',
    'TARGET_TREND_ANG_5_8',

]


SCORING      = "neg_root_mean_squared_error"
N_ITER       = 100
N_JOBS       = -1
VERBOSE      = 1
RANDOM_STATE = 42

KFOLDS = [
    {"TRAIN": [2010, 2011, 2012, 2013, 2014, 2015],             "TEST": [2016, 2017]},
    {"TRAIN": [2010, 2011, 2012, 2013, 2014, 2015, 2016],       "TEST": [2017, 2018]},
    {"TRAIN": [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017], "TEST": [2018, 2019]},
]

PARAM_DIST = {
    "n_estimators":      np.arange(50, 251, 50),
    "max_depth":         [3, 4, 5, 6],
    "learning_rate":     [0.01, 0.03, 0.05, 0.1],
    "subsample":         [0.7, 0.85, 1.0],
    "colsample_bytree":  [0.7, 0.85, 1.0],
}

# --------------- UTILS --------------- #
def leer_features_por_stock(path_csv: str) -> dict:
    """Lee CSV con columnas Stock; Features (separadas por comas)."""
    df = pd.read_csv(path_csv, sep=';')
    m = {}
    for _, row in df.iterrows():
        stock = str(row["Stock"]).strip()
        feats = [f.strip() for f in str(row["Features"]).split(",") if f.strip()]
        m[stock] = feats
    return m

def construir_cv_indices_pooling(df_all: pd.DataFrame, stock_objetivo: str):
    """Devuelve lista de (train_idx, test_idx) para folds temporales."""
    splits = []
    for f in KFOLDS:
        train_mask = df_all["Fecha"].dt.year.isin(f["TRAIN"])
        test_mask  = (df_all["Fecha"].dt.year.isin(f["TEST"])) & (df_all["Symbol"] == stock_objetivo)
        tr = df_all.index[train_mask].to_numpy()
        te = df_all.index[test_mask].to_numpy()
        if len(tr) == 0 or len(te) == 0:
            continue
        splits.append((tr, te))
    return splits

# --------------- MAIN --------------- #
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Dataset global
    df = pd.read_csv(DATASET_PATH, sep=';', parse_dates=["Fecha"])
    df = df.sort_values(["Symbol", "Fecha"]).reset_index(drop=True)

    # Validación rápida
    if not set(TARGETS).issubset(df.columns):
        faltan = set(TARGETS) - set(df.columns)
        raise ValueError(f"[ERROR] Faltan targets en el dataset: {faltan}")

    for TARGET_COL in TARGETS:
        print("\n" + "="*90)
        print(f"=== HPO para TARGET: {TARGET_COL} ===")
        print("="*90)

        # Cargar features GA de este target
        feats_csv = os.path.join(RESULTS_DIR, f"Features_Seleccionadas_GA_{TARGET_COL}.csv")
        if not os.path.isfile(feats_csv):
            print(f"[ERROR] No se encuentran features GA para {TARGET_COL} ({feats_csv}).")
            continue
        features_map = leer_features_por_stock(feats_csv)

        filas_resultados = []

        for stock in SYMBOLS_TO_TEST:
            if stock not in features_map:
                print(f"[WARN] No hay features GA para {stock} en {TARGET_COL}. Se omite.")
                continue

            feats = features_map[stock]
            if not feats:
                continue

            X = df[feats]
            y = df[TARGET_COL]

            cv_indices = construir_cv_indices_pooling(df, stock)
            if not cv_indices:
                print(f"[WARN] {stock}: sin splits válidos. Se omite.")
                continue

            base_est = XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
                tree_method="hist",
            )

            rcv = RandomizedSearchCV(
                estimator=base_est,
                param_distributions=PARAM_DIST,
                n_iter=N_ITER,
                scoring=SCORING,
                cv=cv_indices,
                n_jobs=N_JOBS,
                verbose=VERBOSE,
                random_state=RANDOM_STATE,
                refit=True,
                return_train_score=False
            )

            print(f">>> Buscando hiperparámetros para {stock} ({len(feats)} feats)...")
            try:
                rcv.fit(X, y)
            except Exception as e:
                print(f"[ERROR] HPO falló en {stock} ({TARGET_COL}): {e}")
                continue

            best = rcv.best_params_
            best_score = rcv.best_score_

            fila = {
                "Target": TARGET_COL,
                "Stock": stock,
                "best_score_neg_rmse": best_score,
                "best_rmse_pos": -best_score,
            }
            fila.update(best)
            filas_resultados.append(fila)

        if not filas_resultados:
            print(f"[AVISO] No hubo resultados para {TARGET_COL}.")
            continue

        df_params = pd.DataFrame(filas_resultados)\
                      .sort_values(["best_rmse_pos","Stock"], ascending=[True, True])\
                      .reset_index(drop=True)

        out_file = os.path.join(RESULTS_DIR, f"Hiperparametros_{TARGET_COL}.csv")
        df_params.to_csv(out_file, index=False, sep=';')
        print(f"Guardado: {out_file}")

    print("\nProceso HPO multi-target COMPLETADO ✔")

if __name__ == "__main__":
    main()
