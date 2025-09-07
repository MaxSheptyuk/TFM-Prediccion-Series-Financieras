"""
Pipeline_BackTesting_ARIMA_SingleTrain.py
-----------------------------------------
ARIMA entrenado una sola vez (2010–2019).
Predicciones continuas sobre 2020–2024.
Comparación justa con ML (misma lógica de entrenamiento/test).
"""

# ========== CONFIGURACIÓN ==========

DATA_DIR        = "DATA"
DATASET_FILE    = "AllStocksHistoricalData.csv"

SYMBOLS  = [
    'NVDA','AAPL','AMZN','LRCX','SBUX','REGN','KLAC','BKNG','AMD','VRTX',
    'MAR','CDNS','CAT','INTU','GILD','MU','EBAY','AXP','AMAT','COST','MSFT',
    'ORCL','ADI','MS','NKE'
]

OUT_DIR         = "Resultados"

TRAIN_START_Y   = 2010
TRAIN_END_Y     = 2019
TEST_START_Y    = 2020
TEST_END_Y      = 2024

ORDER           = (1, 1, 0)
USE_LOG         = True

THRESHOLD       = 0.51
USE_CONTINUOUS  = True

CAPITAL_INI     = 10_000
TP_PCT          = 0.015
SL_PCT          = 0.03
PACIENCIA_DIAS  = 5

# ===================================

import os
import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from BackTesting.Backtester import Backtester

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- Carga ----------
def cargar_dataset_unico(path_csv: str) -> pd.DataFrame:
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"No se encuentra el dataset: {path_csv}")
    df = pd.read_csv(path_csv, sep=';', parse_dates=['Fecha'])
    return df.sort_values(['Symbol', 'Fecha']).reset_index(drop=True)

# ---------- Señales ARIMA ----------
def arima_single_train(df_symbol: pd.DataFrame,
                       order=(1, 1, 0),
                       use_log: bool = True) -> pd.DataFrame:
    """
    Entrena ARIMA una sola vez con el periodo 2010–2019
    y genera predicciones continuas para 2020–2024.
    """
    # Ordenamos por fecha
    df = df_symbol[['Fecha','Close']].sort_values('Fecha').reset_index(drop=True).copy()
    precios = df['Close'].astype(float).values

    # Transformación logarítmica si procede
    serie_y = np.log(precios) if use_log else precios

    # Máscaras de train/test
    mask_train = (df['Fecha'].dt.year >= TRAIN_START_Y) & (df['Fecha'].dt.year <= TRAIN_END_Y)
    mask_test  = (df['Fecha'].dt.year >= TEST_START_Y)  & (df['Fecha'].dt.year <= TEST_END_Y)

    serie_train = serie_y[mask_train.values]

    if len(serie_train) < 250:
        # No hay suficientes datos
        df['ARIMA_DELTA_5D'] = np.nan
        df['ARIMA_SIGNAL_5D'] = 0.0
        df['ARIMA_PROBA_5D'] = 0.5
        return df

    # Entrenamiento único del modelo ARIMA
    modelo = SARIMAX(serie_train, order=order, trend='n',
                     enforce_stationarity=False, enforce_invertibility=False)
    res = modelo.fit(disp=False, maxiter=50, method='lbfgs', concentrate_scale=True)

    # Índices absolutos del test en el DataFrame completo
    idx_test = np.where(mask_test.values)[0]
    start_idx = idx_test[0]
    end_idx   = idx_test[-1]

    # Predicciones para todo el rango test
    forecast_obj = res.get_prediction(start=start_idx, end=end_idx)
    pred_values = np.asarray(forecast_obj.predicted_mean)   # ✅ CORREGIDO

    # Ajustar escala log si procede
    if use_log:
        pred_values = np.exp(pred_values)

    # Valores reales de test
    precios_test = precios[mask_test.values]

    # Delta: predicho – real
    delta = pred_values - precios_test

    # Construcción de columnas nuevas
    df['ARIMA_DELTA_5D'] = np.nan
    df['ARIMA_SIGNAL_5D'] = 0.0
    df['ARIMA_PROBA_5D']  = 0.5

    df.loc[mask_test, 'ARIMA_DELTA_5D'] = delta
    df.loc[mask_test, 'ARIMA_SIGNAL_5D'] = (delta > 0).astype(float)
    df.loc[mask_test, 'ARIMA_PROBA_5D']  = df.loc[mask_test, 'ARIMA_SIGNAL_5D'].map({1.0: 0.6, 0.0: 0.4})

    return df

# ---------- Backtesting ----------
def backtest_arima_un_symbol(df_all: pd.DataFrame, symbol: str) -> dict:
    df_sym = df_all[df_all['Symbol'] == symbol].copy()
    df_pred_all = arima_single_train(
        df_symbol=df_sym,
        order=ORDER,
        use_log=USE_LOG
    )

    # Filtrar solo periodo test
    mask_test = (df_pred_all['Fecha'].dt.year >= TEST_START_Y) & (df_pred_all['Fecha'].dt.year <= TEST_END_Y)
    df_test = df_pred_all.loc[mask_test].copy()

    if USE_CONTINUOUS:
        signal = df_test['ARIMA_PROBA_5D'].fillna(0.5).values
    else:
        signal = df_test['ARIMA_SIGNAL_5D'].fillna(0.0).values

    bt = Backtester(
        threshold_buy=THRESHOLD,
        paciencia_max_dias=PACIENCIA_DIAS,
        capital_inicial=CAPITAL_INI,
        tp_pct=TP_PCT,
        sl_pct=SL_PCT,
        save_trades=False
    )

    results = bt.run(
        df_test=df_test[['Fecha','Close']].copy(),
        y_pred=signal,
        symbol=symbol
    )

    trade_log = results.get("trade_log", pd.DataFrame())
    if not trade_log.empty:
        trade_log.insert(0, "Symbol", symbol)
    return {"symbol": symbol, "trade_log": trade_log}

# ---------- MAIN ----------
def main():
    dataset_path = os.path.join(DATA_DIR, DATASET_FILE)
    df_all = cargar_dataset_unico(dataset_path)
    os.makedirs(OUT_DIR, exist_ok=True)

    trade_logs = []
    for sym in SYMBOLS:
        print(f">> Ejecutando ARIMA (single-train) para {sym}")
        r = backtest_arima_un_symbol(df_all, sym)
        if r and not r["trade_log"].empty:
            trade_logs.append(r["trade_log"])

    if trade_logs:
        df_alltrades = pd.concat(trade_logs, ignore_index=True)
        out_csv = os.path.join(OUT_DIR, "Trading_Log_ARIMA_AllStocks.csv")
        df_alltrades.to_csv(out_csv, index=False, sep=";")
        print(f"\n Log maestro guardado en {out_csv} con {len(df_alltrades)} operaciones")
    else:
        print("No se generó ningún trade log.")

if __name__ == "__main__":
    main()
