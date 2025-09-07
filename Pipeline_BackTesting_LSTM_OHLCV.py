# Pipeline_BackTesting_LSTM_OHLCV_SoloPorStock.py
# ------------------------------------------------
# Objetivo (versión sencilla y clara):
#   - Entrenar un LSTM por ACTIVO usando SOLO OHLCV (Open, High, Low, Close, Volume).
#   - Usar como objetivo el MISMO target que ya tenemos (ej.: TARGET_TREND_ANG_15_5).
#   - TRAIN = 2010–2019 (solo ese stock). TEST OOS = 2020–2024 (solo ese stock).
#   - Señal: BUY si y_hat > 0.51 (sin calibraciones ni barridos).
#   - Usar el MISMO Backtester (sin tocar su código) y guardar un CSV único con TODAS las operaciones.
#
# Decisiones fijas (sin tuning para mantenerlo simple):
#   - Ventana temporal L=60 (historial de 60 días).
#   - Horizonte H=5 (coincide con la paciencia del backtester).
#   - Modelo: LSTM (1 capa, 64 unidades, dropout 0.2), MSELoss, Adam lr=1e-3, 20 épocas, batch=256.
#   - Estandarización z-score por canal usando SOLO el TRAIN del propio stock (evita “contaminar” con otros).
#   - CPU.
#
# Requisitos:
#   - pandas, numpy, torch (PyTorch).
#   - Tu clase Backtester disponible: from BackTesting.Backtester import Backtester
#   - Dataset con columnas: ['Symbol','Fecha','Open','High','Low','Close','Volume', TARGET_COL]
#
# Nota:
#   - Este script NO modifica el Backtester. Le pasamos:
#       * df_test_aligned -> dataframe de test alineado con las predicciones
#       * y_pred          -> señales continuas (y_hat)
#       * symbol=stock    -> ticker actual
#   - El Backtester aplica internamente threshold_buy=0.51 para decidir comprar o no.
# ------------------------------------------------

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from BackTesting.Backtester import Backtester  # Asegúrate de que la importación funciona en tu entorno

# ============== CONFIGURACIÓN (EDITA AQUÍ SI LO NECESITAS) ==============

# Ruta al dataset (debe incluir OHLCV + TARGET_COL)
DATASET_PATH = "DATA/Dataset_All_Features_Transformado.csv"

# Carpeta de salida para logs y predicciones
OUT_DIR    = "Resultados"
TRADES_LOG = os.path.join(OUT_DIR, "Trading_Log_AllStocks_LSTM.csv")   # CSV final con TODAS las operaciones
PREDS_DIR  = os.path.join(OUT_DIR, "preds_lstm")                       # y_hat por activo (trazabilidad)

# Lista de símbolos a evaluar
SYMBOLS_TO_TEST = [
    'NVDA','AAPL','AMZN','LRCX','SBUX','REGN','KLAC','BKNG','AMD','VRTX',
    'MAR','CDNS','CAT','INTU','GILD','MU','EBAY','AXP','AMAT','COST','MSFT',
    'ORCL','ADI','MS','NKE'
]

# Columna objetivo (tu target ya calculado en el dataset)
TARGET_COL = "TARGET_TREND_ANG_15_5"

# Años de entrenamiento y test (solo por stock)
TRAIN_YEARS    = list(range(2010, 2020))      # 2010–2019
OOS_TEST_YEARS = [2020, 2021, 2022, 2023, 2024]

# Parámetros del Backtester (los mismos que usas en XGB)
THRESHOLD_BUY      = 0.51
PACIENCIA_MAX_DIAS = 5       # H
CAPITAL_INICIAL    = 10000
TP_PCT             = 0.015   # 1.5%
SL_PCT             = 0.03    # 3%

# Parámetros del LSTM (fijos)
L_WINDOW     = 60     # tamaño de ventana (histórico)
H_HORIZON    = 5      # horizonte de 5 días (coincidir con backtester)
HIDDEN_UNITS = 64
NUM_LAYERS   = 1
DROPOUT      = 0.2
LR           = 1e-3
EPOCHS       = 20
BATCH_SIZE   = 256

# Semilla para reproducibilidad
RANDOM_STATE = 42

# ========================================================================


# -------------------- FUNCIONES AUXILIARES (SIMPLLES) --------------------

def comprobar_columnas_minimas(df: pd.DataFrame):
    """
    Asegura que el dataset tiene todas las columnas necesarias.
    """
    requeridas = {"Symbol","Fecha","Open","High","Low","Close","Volume", TARGET_COL}
    faltan = requeridas - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en el dataset: {faltan}")


def filtrar_por_anios_y_simbolo(df: pd.DataFrame, anios: list, symbol: str) -> pd.DataFrame:
    """
    Devuelve un DataFrame con filas del símbolo indicado y años en la lista.
    """
    df2 = df[(df["Symbol"] == symbol) & (df["Fecha"].dt.year.isin(anios))].copy()
    return df2


def safe_log_serie(serie: pd.Series) -> pd.Series:
    """
    Aplica log de forma segura:
      - Si valor <= 0 => NaN (evita log(0) o log negativos).
    Lo usamos para precios (Open, High, Low, Close).
    """
    arr = serie.to_numpy(dtype=np.float64, copy=False)
    out = np.full_like(arr, np.nan, dtype=np.float64)
    mask = arr > 0
    out[mask] = np.log(arr[mask])
    return pd.Series(out, index=serie.index)


def construir_ventanas_ohlcv_simple(df_sym: pd.DataFrame, L: int, H: int):
    """
    Construye ventanas para UN SÓLO SÍMBOLO (ya filtrado):
      - X: (N, L, 5) con columnas [logO, logH, logL, logC, dlogV]
      - y: (N,) = TARGET_COL alineado al tiempo t
      - fechas: Serie (N,) con df['Fecha'] en t
      - df_base_limpio: df tras limpiar NaN que sirve para alinear con backtester

    Nota:
      - dlogV = diff(log1p(Volume)) para aceptar ceros en volumen.
      - Quitamos filas con NaN (por log seguro y la primera diff de volumen).
      - Las últimas H filas no se usan como "t" porque no existe t+H para backtest.
    """
    # Copia para no tocar el original
    df = df_sym.copy()

    # Asegurar tipos numéricos
    for col in ['Open','High','Low','Close','Volume']:
        df[col] = df[col].astype(float)

    # Logs seguros para precios
    df['logO'] = safe_log_serie(df['Open'])
    df['logH'] = safe_log_serie(df['High'])
    df['logL'] = safe_log_serie(df['Low'])
    df['logC'] = safe_log_serie(df['Close'])

    # Volumen: log1p y diferencia para estabilizar
    df['log1pV'] = np.log1p(df['Volume'])
    df['dlogV'] = df['log1pV'].diff()

    # Quitamos filas con NaN en columnas necesarias
    df = df.dropna(subset=['logO','logH','logL','logC','dlogV', TARGET_COL]).reset_index(drop=True)

    # Comprobación de longitud mínima (debe permitir ventana L y horizonte H)
    last_idx_util = len(df) - H
    if last_idx_util <= L:
        # No hay datos suficientes para hacer al menos 1 muestra
        return None, None, None, None

    # Construimos las ventanas
    X_list, y_list, fechas = [], [], []
    for t in range(L, last_idx_util):
        ventana = df.loc[t-L:t-1, ['logO','logH','logL','logC','dlogV']].values  # (L,5)
        X_list.append(ventana)
        y_list.append(df.loc[t, TARGET_COL])
        fechas.append(df.loc[t, 'Fecha'])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    fechas = pd.Series(fechas)

    # También devolvemos el df limpio para poder alinear test con backtester
    return X, y, fechas, df


def zscore_por_canal_usando_train(X_train: np.ndarray, X_test: np.ndarray):
    """
    Estandariza (z-score) por canal usando SOLO el TRAIN del propio stock.
    Así evitamos "mirar" el test y mantenemos comparabilidad.
    """
    C = X_train.shape[-1]          # número de canales (5)
    flat = X_train.reshape(-1, C)  # (N*L, C) para calcular medias y std
    media = flat.mean(axis=0)
    std   = flat.std(axis=0)
    std[std == 0] = 1.0            # por si acaso

    def aplicar(x):
        xf = x.reshape(-1, C)
        xf = (xf - media) / std
        return xf.reshape(x.shape)

    return aplicar(X_train), aplicar(X_test), media, std


# ------------------------- MODELO LSTM SENCILLO -------------------------

class LSTMRegresor(nn.Module):
    """
    LSTM muy simple que recibe una secuencia (L, C=5) y devuelve un escalar (y_hat).
    """
    def __init__(self, input_dim=5, hidden_units=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Linear(hidden_units, 1)

    def forward(self, x):
        # x: (batch, L, C)
        out, _ = self.lstm(x)   # (batch, L, hidden)
        out = out[:, -1, :]     # nos quedamos con el último paso temporal
        out = self.head(out)    # (batch, 1)
        return out.squeeze(1)   # (batch,)


def entrenar_lstm_simple(X_train: np.ndarray, y_train: np.ndarray) -> LSTMRegresor:
    """
    Entrena el LSTM con parámetros fijos. Devuelve el modelo entrenado.
    """
    # Semillas
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    # Modelo, optimizador y pérdida
    model = LSTMRegresor(input_dim=X_train.shape[-1],
                         hidden_units=HIDDEN_UNITS,
                         num_layers=NUM_LAYERS,
                         dropout=DROPOUT)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # Tensores en CPU
    Xtr = torch.from_numpy(X_train).to(torch.float32)
    ytr = torch.from_numpy(y_train).to(torch.float32)

    # Bucle de entrenamiento sencillo por minibatches
    n = len(Xtr)
    idx = np.arange(n)

    for ep in range(1, EPOCHS + 1):
        np.random.shuffle(idx)
        ep_loss = 0.0
        pasos = 0

        for i in range(0, n, BATCH_SIZE):
            batch_idx = idx[i:i+BATCH_SIZE]
            xb = Xtr[batch_idx]
            yb = ytr[batch_idx]

            model.train()
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()

            ep_loss += float(loss.item())
            pasos += 1

        if pasos == 0: pasos = 1
        if ep == 1 or ep % 5 == 0 or ep == EPOCHS:
            print(f"  Época {ep:02d}/{EPOCHS} | Pérdida media: {ep_loss/pasos:.6f}")

    return model


@torch.no_grad()
def predecir_en_lotes(model: LSTMRegresor, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """
    Genera predicciones en bloques grandes para ahorrar memoria y tiempo.
    """
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i+batch_size]).to(torch.float32)
        pred = model(xb).cpu().numpy()
        preds.append(pred)
    return np.concatenate(preds, axis=0)


# ------------------------------ PIPELINE ------------------------------

def main():
    # Crear carpetas de salida
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PREDS_DIR, exist_ok=True)

    # Cargar dataset y ordenar
    df = pd.read_csv(DATASET_PATH, sep=';', parse_dates=["Fecha"])
    df = df.sort_values(["Symbol", "Fecha"]).reset_index(drop=True)

    # Comprobar columnas necesarias
    comprobar_columnas_minimas(df)

    # Aquí guardaremos el log de TODAS las operaciones
    logs_todos = []

    # ======= Bucle por cada símbolo (entrenamiento y test por stock) =======
    for stock in SYMBOLS_TO_TEST:
        print(f"\n=== Procesando {stock} ===")

        # Filtrar TRAIN y TEST SOLO para este stock
        df_train_sym = filtrar_por_anios_y_simbolo(df, TRAIN_YEARS, stock)
        df_test_sym  = filtrar_por_anios_y_simbolo(df, OOS_TEST_YEARS, stock)

        if df_train_sym.empty:
            warnings.warn(f"[{stock}] TRAIN vacío (2010–2019). Se omite.")
            continue
        if df_test_sym.empty:
            warnings.warn(f"[{stock}] TEST vacío (2020–2024). Se omite.")
            continue

        # Construir ventanas para TRAIN (solo este stock)
        X_train, y_train, _, df_train_limpio = construir_ventanas_ohlcv_simple(
            df_train_sym, L=L_WINDOW, H=H_HORIZON
        )
        if X_train is None:
            warnings.warn(f"[{stock}] No hay suficientes datos para ventanas de TRAIN. Se omite.")
            continue

        # Construir ventanas para TEST (solo este stock)
        X_test, y_test_dummy, fechas_test, df_test_limpio = construir_ventanas_ohlcv_simple(
            df_test_sym, L=L_WINDOW, H=H_HORIZON
        )
        if X_test is None or len(X_test) == 0:
            warnings.warn(f"[{stock}] No hay suficientes datos para ventanas de TEST. Se omite.")
            continue

        # Estandarizar usando SOLO las estadísticas de TRAIN (por stock)
        X_train_z, X_test_z, media_canal, std_canal = zscore_por_canal_usando_train(X_train, X_test)

        # Entrenar LSTM para este stock
        print(f"[{stock}] Entrenando LSTM (ventanas TRAIN = {len(X_train_z):,}) ...")
        model = entrenar_lstm_simple(X_train_z, y_train)

        # Predecir en TEST
        print(f"[{stock}] Prediciendo en TEST (ventanas = {len(X_test_z):,}) ...")
        y_pred = predecir_en_lotes(model, X_test_z, batch_size=4096)

        # Alinear el DataFrame de TEST con las predicciones:
        # - df_test_limpio es el df tras logs seguros y dropna.
        # - build_windows usa índices t en [L, len(df_limpio)-H).
        # - Por tanto, las filas que corresponden a y_pred son:
        #       df_test_aligned = df_test_limpio.iloc[L : len(df_test_limpio) - H]
        df_test_aligned = df_test_limpio.iloc[L_WINDOW : len(df_test_limpio) - H_HORIZON].copy()

        # Seguridad: comprobar longitudes iguales
        if len(df_test_aligned) != len(y_pred):
            raise RuntimeError(
                f"[{stock}] Desalineación: df_test_aligned={len(df_test_aligned)} vs y_pred={len(y_pred)}"
            )

        # Guardar y_hat por fecha (trazabilidad y gráficos)
        preds_path = os.path.join(PREDS_DIR, f"LSTM_{stock}.csv")
        pd.DataFrame({
            "Date": df_test_aligned["Fecha"].values,
            "y_hat": y_pred
        }).to_csv(preds_path, sep=';', index=False, date_format="%Y-%m-%d")

        # Ejecutar Backtester con la señal continua (umbral 0.51 lo gestiona el backtester)
        bt = Backtester(
            threshold_buy=THRESHOLD_BUY,
            paciencia_max_dias=PACIENCIA_MAX_DIAS,
            capital_inicial=CAPITAL_INICIAL,
            tp_pct=TP_PCT,
            sl_pct=SL_PCT,
            save_trades=False
        )
        bt_result = bt.run(df_test_aligned, y_pred, symbol=stock)

        # Recuperar log de operaciones del backtester y añadir metadatos
        trades_df = bt_result.get("trade_log", pd.DataFrame()).copy()
        if not trades_df.empty:
            trades_df.insert(0, "Symbol", stock)
            trades_df.insert(1, "OrdenN", range(1, len(trades_df) + 1))
            logs_todos.append(trades_df)

        # Mensaje resumen por consola
        print(f"[{stock}] Trades={bt_result.get('num_trades', 0)} | ROI={bt_result.get('roi', np.nan):.2f}%")

    # ======= Guardar el CSV unificado con TODAS las operaciones =======
    if logs_todos:
        log_global = pd.concat(logs_todos, ignore_index=True)
        log_global = log_global.sort_values(["Symbol", "Fecha"]).reset_index(drop=True)
        os.makedirs(OUT_DIR, exist_ok=True)
        log_global.to_csv(TRADES_LOG, sep=';', index=False, date_format="%Y-%m-%d")
        print(f"\n[OK] Log de trading unificado guardado en: {os.path.abspath(TRADES_LOG)}")
    else:
        print("\n[WARN] No se generaron operaciones. Log no guardado.")


# Punto de entrada del script
if __name__ == "__main__":
    main()
