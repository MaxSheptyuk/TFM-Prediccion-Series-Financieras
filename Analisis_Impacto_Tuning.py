"""
Analisis_Impacto_Tuning.py
Compara el impacto del ajuste de hiperparámetros por stock (RandomizedCV)
frente a un baseline sin tuning, usando el mismo split:
  - Train: hasta 2019 inclusive
  - Test : 2020–2024

Salida:
- Resultados/Comparativa_Tuning_2020_2024.csv
- Resultados/Grafico_Mejora_Tuning.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# -------------------- CONFIG -------------------- #
DATASET_PATH  = "DATA/Dataset_All_Features_Transformado.csv"
FEATS_PATH    = "Resultados/Features_Seleccionadas_GA.csv"      # columnas: Stock;Features (coma-sep)
PARAMS_STOCK  = "Resultados/Hiperparametros_por_stock.csv"      # generado por tu RandomizedCV
PARAMS_GLOBAL = "Resultados/Hiperparametros_globales_media.csv" # opcional
OUT_COMP_CSV  = "Resultados/Comparativa_Tuning_2020_2024.csv"
OUT_FIG_PATH  = "Resultados/Grafico_Mejora_Tuning.png"

TARGET_COL    = "TARGET_TREND_ANG_15_5"
SYMBOLS       = [
    'NVDA','AAPL','AMZN','LRCX','SBUX','REGN','KLAC','BKNG','AMD','VRTX',
    'MAR','CDNS','CAT','INTU','GILD','MU','EBAY','AXP','AMAT','COST','MSFT',
    'ORCL','ADI','MS','NKE'
]

# Baseline a comparar: "defaults" (XGB por defecto) o "global_media" (si existe CSV)
BASELINE_MODE = "defaults"  # "defaults" | "global_media"

RANDOM_STATE  = 42
TREE_METHOD   = "hist"      # cambia a "gpu_hist" si vas con GPU
N_JOBS        = -1

TRAIN_YEARS   = list(range(2010, 2020))  # 2010–2019
TEST_YEARS    = [2020, 2021, 2022, 2023, 2024]

# -------------------- HELPERS -------------------- #
def leer_features_por_stock(path_csv: str) -> dict:
    df = pd.read_csv(path_csv, sep=';')
    m = {}
    for _, row in df.iterrows():
        stock = str(row["Stock"]).strip()
        feats = [f.strip() for f in str(row["Features"]).split(",") if f.strip()]
        m[stock] = feats
    return m

def cargar_parametros_stock(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv, sep=';')
    # nombres esperados: Stock, n_estimators, max_depth, learning_rate, subsample, colsample_bytree
    return df

def modelo_xgb(params: dict) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method=TREE_METHOD,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        **params
    )

def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
def split_train_test(df_sym: pd.DataFrame):
    tr = df_sym[df_sym["Fecha"].dt.year.isin(TRAIN_YEARS)].copy()
    te = df_sym[df_sym["Fecha"].dt.year.isin(TEST_YEARS)].copy()
    return tr, te

def params_por_defecto() -> dict:
    # Defaults razonables de XGBoost (equivalentes a los de la lib)
    return dict(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.3,
        subsample=1.0,
        colsample_bytree=1.0
    )

def params_media_global(path_csv: str) -> dict:
    df = pd.read_csv(path_csv, sep=';')
    p = {k: (int(v) if k in {"n_estimators","max_depth"} else float(v)) for k, v in df.iloc[0].to_dict().items()}
    return p

# -------------------- MAIN -------------------- #
def main():
    # Carga dataset
    df = pd.read_csv(DATASET_PATH, sep=';', parse_dates=["Fecha"])
    df = df.sort_values(["Symbol","Fecha"]).reset_index(drop=True)

    features_map = leer_features_por_stock(FEATS_PATH)
    df_params    = cargar_parametros_stock(PARAMS_STOCK)

    # Baseline params
    if BASELINE_MODE == "global_media" and os.path.isfile(PARAMS_GLOBAL):
        base_params = params_media_global(PARAMS_GLOBAL)
    else:
        base_params = params_por_defecto()

    registros = []

    for stock in SYMBOLS:
        if stock not in features_map:
            print(f"[WARN] {stock} sin features en {FEATS_PATH}. Se omite.")
            continue

        feats = features_map[stock]
        df_sym = df[df["Symbol"] == stock].copy()
        if df_sym.empty:
            print(f"[WARN] Sin datos para {stock}.")
            continue

        tr, te = split_train_test(df_sym)
        if tr.empty or te.empty:
            print(f"[WARN] Split vacío para {stock}. Revisa años.")
            continue

        Xtr, ytr = tr[feats], tr[TARGET_COL]
        Xte, yte = te[feats], te[TARGET_COL]

        # ------- Baseline -------
        m_base = modelo_xgb(base_params)
        m_base.fit(Xtr, ytr)
        rmse_base = rmse(yte, m_base.predict(Xte))

        # ------- Tuned por stock -------
        rowp = df_params[df_params["Stock"] == stock]
        if rowp.empty:
            print(f"[WARN] Sin params tuned para {stock}. Uso baseline como tuned (no hay comparativa).")
            tuned_params = base_params.copy()
        else:
            tuned_params = dict(
                n_estimators=int(rowp["n_estimators"].iloc[0]),
                max_depth=int(rowp["max_depth"].iloc[0]),
                learning_rate=float(rowp["learning_rate"].iloc[0]),
                subsample=float(rowp["subsample"].iloc[0]),
                colsample_bytree=float(rowp["colsample_bytree"].iloc[0]),
            )

        m_tuned = modelo_xgb(tuned_params)
        m_tuned.fit(Xtr, ytr)
        rmse_tuned = rmse(yte, m_tuned.predict(Xte))

        mejora_pct = 100.0 * (rmse_base - rmse_tuned) / rmse_base if rmse_base > 0 else 0.0

        registros.append({
            "Stock": stock,
            "RMSE_Base": rmse_base,
            "RMSE_Tuned": rmse_tuned,
            "Mejora_%": mejora_pct,
            "BaselineMode": BASELINE_MODE
        })

        print(f"{stock}: RMSE base={rmse_base:.6f} | tuned={rmse_tuned:.6f} | mejora={mejora_pct:+.2f}%")

    if not registros:
        raise RuntimeError("No se generaron resultados. Revisa warnings arriba.")

    df_out = pd.DataFrame(registros).sort_values("Mejora_%", ascending=False)
    df_out.to_csv(OUT_COMP_CSV, index=False, sep=';')
    print(f"\nGuardado comparativa: {OUT_COMP_CSV}")

    # ----- Gráfico -----
    plt.figure(figsize=(12, 6))
    orden = df_out.sort_values("Mejora_%", ascending=False)
    plt.bar(orden["Stock"], orden["Mejora_%"])
    plt.axhline(0, linestyle="--")
    plt.title(f"Mejora porcentual (RMSE) con tuning vs. baseline [{BASELINE_MODE}] · Test 2020–2024")
    plt.ylabel("Mejora % (↑ mejor)")
    plt.xlabel("Stock")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_FIG_PATH, dpi=160)
    print(f"Guardado gráfico: {OUT_FIG_PATH}")

if __name__ == "__main__":
    main()
