import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def cross_val_scores_symbol_years(df, symbol, features, target_col, years_test):
    """
    Calcula métricas de validación cruzada temporal para un símbolo concreto, usando años predefinidos como folds.
    Se entrena con todos los símbolos hasta cada año de test, y se evalúa solo el símbolo y año correspondientes.
    """
    df_symbol = df[df['Symbol'] == symbol].sort_values('Fecha')

    scores = []

    for year in years_test:
        
        # Definimos la fecha de corte del fold: todo lo anterior a ese año es train, el año es test
        df_train = df[df['Fecha'].dt.year < year]
        df_test = df_symbol[df_symbol['Fecha'].dt.year == year]

        # Si no hay datos para train o test, saltamos el fold (puede pasar para históricos cortos)
        if df_test.empty or df_train.empty:
            continue

        # Definimos variables de entrada (X) y salida (y) para ambos conjuntos
        X_train = df_train[features]
        y_train = df_train[target_col]
        X_test = df_test[features]
        y_test = df_test[target_col]

        
        # Entrenamos el modelo con todos los datos previos al fold
        model = XGBRegressor(n_estimators=60, max_depth=4, random_state=42, n_jobs=30)
        model.fit(X_train, y_train)
        
        # Predecimos sobre el año objetivo del símbolo
        y_pred = model.predict(X_test)
        
        # Calculamos las métricas para evaluar el rendimiento fuera de muestra
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Guardamos todo en una lista para luego promediar o analizar fold a fold
        scores.append({
            'year': year,
            'train_hasta_año': year - 1,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'n_train': len(X_train),
            'n_test': len(X_test)
        })
    return scores

def main():
    # Cargamos el dataset ya preprocesado, asegurando que la columna Fecha es datetime
    df = pd.read_csv("DATA/Dataset_All_Features_Transformado.csv", sep=";", parse_dates=["Fecha"])
    
    # Seleccionamos solo las columnas relevantes como features (quitamos fecha, símbolo, precios, etc.)
    cols_a_excluir = ['Fecha', 'Symbol', 'Open', 'Close', 'High', 'Low', 'AdjClose', 'Volume']
    cols_a_excluir += [c for c in df.columns if c.startswith('EMA_')]
    features = [c for c in df.columns if c not in cols_a_excluir and not c.startswith('TARGET_')]
    
        
    
    # Con GA (N Features)
    #features =         ['ROC_1', 'ATR_12', 'BB_WIDTH_10_2', 'BB_LOWER_20_2', 'BB_WIDTH_20_2.5', 'BB_WIDTH_20_3', 'CONNORS_RSI_3_2_25', 'ADL_EMA_17', 'TEMA_10', 'TEMA_14', 'BINARY_RSI_10_OVERBOUGHT', 'BINARY_RSI_10_OVERSOLD', 'BINARY_RSI_14_OVERSOLD', 'BINARY_MACD_12_26_9_UP', 'BINARY_MACD_SIGNAL_7_18_9_UP', 'BINARY_MACD_10_20_7_UP', 'BINARY_MACD_SIGNAL_10_20_7_UP', 'BINARY_ROC_2_UPTREND', 'BINARY_BB_20_2.5_BELOW', 'BINARY_BB_30_2_BELOW', 'BINARY_CONNORS_RSI_3_2_25_OVERBOUGHT', 'BINARY_ADL_EMA_15_DOWN', 'BINARY_ADL_EMA_20_DOWN', 'BINARY_TEMA_7_UPTREND', 'BINARY_ULCER_INDEX_14_UP']

    # Indicamos el target (variable objetivo)
    target_col = "TARGET_TREND_ANG_15_5"
    
    # Definimos años de validación cruzada temporal
    years_test = [2020, 2021, 2022, 2023, 2024]
    
    # Indicamos el símbolo a evaluar (puedes cambiarlo por el que desees analizar)
    symbol = "NVDA"
    print(f"\nResultados para {symbol} utilizando modelo XGBRegressor con {len(features)} features:")

    # Ejecutamos validación cruzada temporal solo para ese símbolo
    scores = cross_val_scores_symbol_years(df, symbol, features, target_col, years_test)
    if not scores:
        print(f"  Sin datos de test suficientes para {symbol}")
        return

    # Imprimimos métricas fold a fold
    for s in scores:
        print(f"Fold año {s['year']}: train_hasta_año={s['train_hasta_año']} | "
              f"n_train={s['n_train']}, n_test={s['n_test']}, "
              f"RMSE={s['rmse']:.4f}, R2={s['r2']:.3f}")

    # Calculamos y mostramos el promedio de las métricas
    resumen = {k: np.mean([s[k] for s in scores]) for k in ['mse','rmse','mae','r2']}
    resumen['Symbol'] = symbol
    print("\nResumen promedio para", symbol)
    for k in ['mse','rmse','mae','r2']:
        print(f"{k.upper()}: {resumen[k]:.6f}")

if __name__ == "__main__":
    main()
