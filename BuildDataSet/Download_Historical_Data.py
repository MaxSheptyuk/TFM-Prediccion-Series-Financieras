import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.impute import KNNImputer

def apply_etl(df, ohlcv_strategy='median', max_null_pct=0.10, knn_neighbors=5):
    """
    Imputador parametrizable para valores nulos en las columnas OHLCV.

    Estrategias disponibles:
        - 'median': Rellena valores faltantes con la mediana de cada columna.
        - 'average': Rellena valores faltantes con la media aritmética.
        - 'last_known': Propaga el último valor conocido (forward-fill + backward-fill).
        - 'moda' / 'most_frequent': Utiliza el valor más frecuente (moda).
        - 'knn_imputation': Imputación avanzada con KNNImputer (requiere sklearn).

    El parámetro max_null_pct permite fijar un umbral máximo de filas con nulos por símbolo.
    Si lo supera, el proceso para ese símbolo y lo ignora (control de calidad).

    Al final, los precios se redondean a 5 decimales y el volumen se convierte a entero.
    """
    ohlcv = ["Open", "High", "Low", "Close", "Volume"]

    # --- Control de calidad: verificar % de filas con valores nulos ---
    total = len(df)
    filas_con_nulos = df[ohlcv].isnull().any(axis=1).sum()
    null_pct = filas_con_nulos / total

    # Si no hay filas con nulos, pasamos al paso de Redondeo y Formatos de ETL
    if (filas_con_nulos > 0):

        # Si el porcentaje de filas con nulos supera el umbral, detenemos el proceso para este símbolo
        # para evitar problemas de calidad en el dataset final.
        if null_pct > max_null_pct:
            raise ValueError(
                f"Demasiados valores faltantes en {df['Symbol'].iloc[0]}: "
                f"{null_pct:.1%} de las filas tienen algún nulo en OHLCV. "
                f"Proceso detenido para este símbolo por calidad insuficiente."
            )

        # --- Imputación robusta según estrategia seleccionada ---
        if ohlcv_strategy == 'median':
            # Rellenar NaN con la mediana de cada columna
            for col in ohlcv:
                med = df[col].median()
                df[col] = df[col].fillna(med)

        elif ohlcv_strategy == 'average':
            # Rellenar NaN con la media aritmética de cada columna
            for col in ohlcv:
                avg = df[col].mean()
                df[col] = df[col].fillna(avg)

        elif ohlcv_strategy == 'last_known':
            # Forward-fill y backward-fill: el valor válido más próximo
            df[ohlcv] = df[ohlcv].fillna(method='ffill').fillna(method='bfill')

        elif ohlcv_strategy == 'moda' or ohlcv_strategy == 'most_frequent':
            # Moda (valor más frecuente), útil para volumen o casos discretos
            for col in ohlcv:
                moda = df[col].mode()
                if not moda.empty:
                    df[col] = df[col].fillna(moda[0])
                else:
                    df[col] = df[col].fillna(0)  # fallback para columnas vacías

        elif ohlcv_strategy == 'knn_imputation':
            # Imputación KNN: usa el patrón de los datos vecinos
            imputer = KNNImputer(n_neighbors=knn_neighbors)
            df[ohlcv] = imputer.fit_transform(df[ohlcv])

        else:
            raise ValueError(f"Estrategia '{ohlcv_strategy}' no reconocida.")


    # --- Redondeo y formatos ---

    # Redondear precios a 5 decimales (precisión suficiente para análisis financiero)
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].round(5)

    # Convertimos Volumen a entero (por integridad de datos y almacenamiento)
    df["Volume"] = df["Volume"].astype(int)

    return df

# ----------- CARGA DE DATOS PRINCIPAL ----------- #

symbols = [ "AAPL", "ADI", "ADP", "AMAT", "AMD", "AMGN", "AMZN", "AXP", "BKNG",
           "CAT", "CDNS", "CMCSA", "COST", "CSCO", "CVX", "DIS", "EBAY", "GE",
           "GILD", "HON", "IBM", "INTC", "INTU", "JNJ", "KLAC", "KO", "LRCX",
           "MAR", "MCD", "MCHP", "MMM", "MRK", "MS", "MSFT", "MU", "NKE", "NVDA",
           "ORCL", "ORLY", "PEP", "PFE", "PG", "QCOM", "REGN", "SBUX", "T", "TXN",
           "VRTX", "WMT", "XOM", "^GSPC"]

start_date = "1999-07-01"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

all_data = []

for symbol in symbols:
    
    print(f"Descargando {symbol} ...")
    df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)

    # Si columnas son multinivel (MultiIndex), eliminamos el nivel extra
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # --- Control crítico de ausencia de datos ---
    if df.empty:
        
        # Detenemos el proceso porque no se admite un dataset incompleto.
        raise ValueError(
            f"ERROR: No se encontraron datos para el símbolo '{symbol}'. "
            f"Proceso detenido para evitar inconsistencias y problemas posteriores en el análisis multi-stock."
        )

    # --- Renombramos la columna de fecha  eliminando el índice actual ---
    df = df.reset_index().rename(columns={"Date": "Fecha"})
    df["Symbol"] = symbol
    df = df[["Fecha", "Symbol", "Open", "Close", "High", "Low", "Volume"]]

    # --- ETL: Imputación parametrizada de valores nulos en OHLCV ---
    # La función apply_etl rellena todos los valores faltantes según la estrategia especificada
    # y nunca elimina filas para no romper la alineación temporal del dataset.
        
    # Aplicamos el ETL con la estrategia de imputación de NaN seleccionada
    df = apply_etl(
        df,
        ohlcv_strategy='median',  # La estrategia para experimentos: 'median', 'average', 'moda', 'knn_imputation'
        max_null_pct=0.10,  # Máximo 10% de filas con nulos permitidas
        knn_neighbors=5  # Número de vecinos para KNNImputer
    )
    
    # --- Añadimos el DataFrame procesado a la lista de todos los datos ---
    all_data.append(df)

# --- Concatenación final ---
final_df = pd.concat(all_data, axis=0)

# --- Ordenar por símbolo y fecha para consistencia ---
final_df = final_df.sort_values(["Symbol", "Fecha"]).reset_index(drop=True)

# --- Guardar a CSV (formato estándar para ETL financieros) ---
final_df.to_csv("Data/AllStocksHistoricalData.csv", sep=";", index=False)
