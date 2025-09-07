# Pipeline_BackTesting_MLP_OHLCV_CloseTarget_ByDate_Refactor_ArchLoop.py
# -----------------------------------------------------------
# OBJETIVO
#   - Entrenar un MLP por ACTIVO para predecir Close_{t+H} usando SOLO OHLCV.
#   - Evaluar varias arquitecturas de MLP en bucle.
#   - Señal de trading simple: 0.6 si ŷ_Close_{t+H} > Close_t; si no 0.4.
#
# DISEÑO
#   - Mono-stock, por FECHA (t y t+H). Nada de índices cruzados.
#   - SIN dropna global: validamos cada ventana localmente.
#   - StandardScaler SOLO con TRAIN (X e y). Sin fuga de información.
#   - Split y alineación SIN masks: listas de índices + merge por Fecha.
#
# REQUISITOS
#   - CSV con: Symbol, Fecha, Open, High, Low, Close, Volume.
#   - Clase BackTesting.Backtester con: run(df_test, y_pred, symbol).
# -----------------------------------------------------------

import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from BackTesting.Backtester import Backtester

# ================== CONFIGURACIÓN ==================

DATASET_PATH = "DATA/Dataset_All_Features_Transformado.csv"

OUT_DIR    = "Resultados"
PREDS_DIR  = os.path.join(OUT_DIR, "preds_mlp_ohlcv")

SYMBOLS_TO_TEST = [
    "NVDA","AAPL","AMZN","LRCX","SBUX","REGN","KLAC","BKNG","AMD","VRTX",
    "MAR","CDNS","CAT","INTU","GILD","MU","EBAY","AXP","AMAT","COST",
    "MSFT","ORCL","ADI","MS","NKE"
]

# Periodos
TRAIN_YEARS    = list(range(2010, 2020))   # 2010..2019
OOS_TEST_YEARS = [2020, 2021, 2022, 2023, 2024]

# Ventana e horizonte
L_WINDOW  = 30
H_HORIZON = 5

# Scalers
X_SCALER_TYPE = "standard"      # "standard" | "minmax" | None
Y_SCALER_TYPE = "standard"      # "standard" | "minmax" | None

# Arquitecturas a probar
MLP_ARCHS = [
    (128,)
]

# Parámetros base del MLP
MLP_PARAMS = dict(
    activation="relu",
    solver="adam",
    learning_rate_init=1e-3,
    alpha=1e-4,
    max_iter=250,
    early_stopping=False,
    shuffle=False,              # no mezclamos para mantener orden temporal
    random_state=42,
    verbose=False
)

# Backtester (la señal que recibe es 0.4/0.6)
THRESHOLD_BUY      = 0.51
PACIENCIA_MAX_DIAS = 5
CAPITAL_INICIAL    = 10000
TP_PCT, SL_PCT     = 0.015, 0.03

# ================== UTILIDADES ==================

def get_scaler(kind: str):
    if kind is None:
        return None
    k = kind.lower()
    if k == "standard":
        return StandardScaler()
    if k == "minmax":
        return MinMaxScaler()
    raise ValueError("Scaler desconocido. Usa 'standard', 'minmax' o None.")

def build_samples_by_date(df_sym: pd.DataFrame, L: int, H: int):
    assert df_sym["Symbol"].nunique() == 1
    df = df_sym.sort_values("Fecha").reset_index(drop=True).copy()
    cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n = len(df)
    if n - H <= L:
        return None, None, None, None, None

    values    = df[cols].to_numpy(dtype=float)
    close_col = values[:, 3]
    row_ok = np.isfinite(values).all(axis=1)

    X_list, y_list, cnow_list, ft_list, fth_list = [], [], [], [], []
    for t in range(L, n - H):
        ventana_ok = True
        for k in range(t - L, t):
            if not row_ok[k]:
                ventana_ok = False
                break
        if not ventana_ok:
            continue

        c_t  = close_col[t]
        c_tH = close_col[t + H]
        if not (np.isfinite(c_t) and np.isfinite(c_tH) and c_t > 0):
            continue

        win = values[t - L: t, :]
        X_list.append(win.reshape(-1))
        y_list.append(c_tH)
        cnow_list.append(c_t)
        ft_list.append(df.at[t, "Fecha"])
        fth_list.append(df.at[t + H, "Fecha"])

    if len(X_list) == 0:
        return None, None, None, None, None

    X          = np.asarray(X_list, dtype=np.float64)
    y_close_tH = np.asarray(y_list,    dtype=np.float64)
    close_t    = np.asarray(cnow_list, dtype=np.float64)
    fechas_t   = pd.Series(ft_list)
    fechas_tH  = pd.Series(fth_list)
    return X, y_close_tH, close_t, fechas_t, fechas_tH

def split_train_test_indices(fechas_t, fechas_tH, train_years, test_years):
    idx_train, idx_test = [], []
    for i in range(len(fechas_t)):
        y_t  = int(fechas_t.iloc[i].year)
        y_tH = int(fechas_tH.iloc[i].year)
        if (y_t in train_years) and (y_tH in train_years):
            idx_train.append(i)
        elif (y_t in test_years) and (y_tH in test_years):
            idx_test.append(i)
    return idx_train, idx_test

def fit_and_transform_scalers(X_tr, y_tr, x_scaler_type, y_scaler_type):
    x_scaler = get_scaler(x_scaler_type)
    y_scaler = get_scaler(y_scaler_type)

    if x_scaler is not None:
        x_scaler.fit(X_tr)
        X_tr_scaled = x_scaler.transform(X_tr)
    else:
        X_tr_scaled = X_tr

    if y_scaler is not None:
        y_scaler.fit(y_tr.reshape(-1, 1))
        y_tr_scaled = y_scaler.transform(y_tr.reshape(-1, 1)).ravel()
    else:
        y_tr_scaled = y_tr

    return x_scaler, y_scaler, X_tr_scaled, y_tr_scaled

def transform_with_x_scaler(X, x_scaler):
    if x_scaler is None:
        return X
    return x_scaler.transform(X)

def inverse_y(y_arr, y_scaler):
    if y_scaler is None:
        return y_arr
    return y_scaler.inverse_transform(y_arr.reshape(-1, 1)).ravel()

# ================== PIPELINE ==================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PREDS_DIR, exist_ok=True)

    df_all = pd.read_csv(DATASET_PATH, sep=";", parse_dates=["Fecha"])
    df_all = df_all.sort_values(["Symbol", "Fecha"]).reset_index(drop=True)

    required = {"Symbol","Fecha","Open","High","Low","Close","Volume"}
    if not required.issubset(df_all.columns):
        missing = required - set(df_all.columns)
        raise ValueError(f"Faltan columnas: {sorted(missing)}")

    # ====== LOOP POR ARQUITECTURA ======
    for arch in MLP_ARCHS:
        print(f"\n===== Probando arquitectura {arch} =====")
        logs_todos = []

        # ----- Loop por símbolo -----
        for stock in SYMBOLS_TO_TEST:
            print(f"\n--- {stock} | MLP OHLCV (ByDate) -> TARGET: Close_{H_HORIZON} | Arch={arch} ---")

            df_sym = df_all[df_all["Symbol"] == stock].copy()
            if df_sym.empty:
                print(f"[{stock}] sin datos; se omite.")
                continue

            X_all, y_all_close, close_t_all, fechas_t_all, fechas_tH_all = build_samples_by_date(
                df_sym, L=L_WINDOW, H=H_HORIZON
            )
            if X_all is None:
                print(f"[{stock}] sin ventanas válidas; se omite.")
                continue

            idx_train, idx_test = split_train_test_indices(
                fechas_t_all, fechas_tH_all, TRAIN_YEARS, OOS_TEST_YEARS
            )
            if len(idx_train) == 0 or len(idx_test) == 0:
                print(f"[{stock}] train/test insuficientes; se omite.")
                continue

            X_tr        = X_all[idx_train]
            y_tr_close  = y_all_close[idx_train]
            X_te        = X_all[idx_test]
            y_te_close  = y_all_close[idx_test]
            close_t_te  = close_t_all[idx_test]
            fechas_te   = fechas_t_all.iloc[idx_test].reset_index(drop=True)

            x_scaler, y_scaler, X_tr_s, y_tr_s = fit_and_transform_scalers(
                X_tr, y_tr_close, X_SCALER_TYPE, Y_SCALER_TYPE
            )
            X_te_s = transform_with_x_scaler(X_te, x_scaler)

            mlp_params = dict(MLP_PARAMS)
            mlp_params["hidden_layer_sizes"] = arch
            mlp = MLPRegressor(**mlp_params)
            mlp.fit(X_tr_s, y_tr_s)

            y_hat_te_s = mlp.predict(X_te_s)
            y_hat_close = inverse_y(y_hat_te_s, y_scaler)

            y_hat_ret   = (y_hat_close - close_t_te) / close_t_te
            y_true_ret  = (y_te_close - close_t_te) / close_t_te
            y_hat_score = np.where(y_hat_close > close_t_te, 0.6, 0.4)

            preds_df = pd.DataFrame({
                "Fecha":       fechas_te.values,
                "y_hat_score": y_hat_score,
                "y_hat_close": y_hat_close,
                "y_true_close": y_te_close,
                "close_t":     close_t_te,
                "y_hat_ret":   y_hat_ret,
                "y_true_ret":  y_true_ret
            })

            df_test_sym = df_sym[df_sym["Fecha"].dt.year.isin(OOS_TEST_YEARS)].copy()
            df_test_sym = df_test_sym.sort_values("Fecha").reset_index(drop=True)

            df_bt = pd.merge(
                df_test_sym,
                preds_df[["Fecha", "y_hat_score", "y_hat_close", "y_true_close",
                          "close_t", "y_hat_ret", "y_true_ret"]],
                on="Fecha",
                how="inner"
            )

            if df_bt.empty:
                print(f"[{stock}] sin fechas comunes entre TEST y preds; se omite.")
                continue

            y_pred_bt = df_bt["y_hat_score"].to_numpy()

            out_csv = os.path.join(PREDS_DIR, f"MLP_OHLCV_CLOSE_BYDATE_{stock}_{'_'.join(map(str, arch))}.csv")
            df_bt_out = df_bt[["Fecha","y_hat_score","y_hat_close","y_true_close",
                               "close_t","y_hat_ret","y_true_ret"]].copy()
            df_bt_out.to_csv(out_csv, sep=";", index=False, date_format="%Y-%m-%d")

            bt = Backtester(
                threshold_buy=THRESHOLD_BUY,
                paciencia_max_dias=PACIENCIA_MAX_DIAS,
                capital_inicial=CAPITAL_INICIAL,
                tp_pct=TP_PCT,
                sl_pct=SL_PCT,
                save_trades=False
            )
            bt_result = bt.run(df_bt, y_pred_bt, symbol=stock)

            trades_df = bt_result.get("trade_log", pd.DataFrame()).copy()
            if not trades_df.empty:
                trades_df.insert(0, "Symbol", stock)
                trades_df.insert(1, "OrdenN", range(1, len(trades_df) + 1))
                logs_todos.append(trades_df)

            print(f"[{stock}] Trades={bt_result.get('num_trades', 0)} | ROI={bt_result.get('roi', np.nan):.2f}%")

        # ===== Log global por arquitectura =====
        if logs_todos:
            log_global = pd.concat(logs_todos, ignore_index=True)
            log_global = log_global.sort_values(["Symbol", "Fecha"]).reset_index(drop=True)
            arq_name = "_".join(map(str, arch))
            out_file = os.path.join(OUT_DIR, f"Trading_Log_AllStocks_MLP_OHLCV_{arq_name}.csv")
            log_global.to_csv(out_file, sep=";", index=False, date_format="%Y-%m-%d")
            print(f"\n[OK] Log unificado en: {os.path.abspath(out_file)}")
        else:
            print(f"\n[WARN] No se generaron operaciones para arquitectura {arch}.")


if __name__ == "__main__":
    main()
