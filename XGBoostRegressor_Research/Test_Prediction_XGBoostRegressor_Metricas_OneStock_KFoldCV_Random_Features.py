import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def cross_val_scores_symbol_years(df, symbol, features, target_col, years_test):
    """
    Realiza validación cruzada temporal fold-a-fold para un símbolo (stock), 
    usando años como splits cronológicos. Entrena con todo el pasado y evalúa 
    sobre el año/fold correspondiente.
    """
    # Filtrar los datos solo para el símbolo que queremos analizar
    df_symbol = df[df['Symbol'] == symbol].sort_values('Fecha')
    scores = []

    # Recorremos cada año de test (un fold)
    for year in years_test:
        # Entrenamiento: todo el pasado
        df_train = df[df['Fecha'].dt.year < year]
        # Test: solo el año objetivo, para ese símbolo
        df_test = df_symbol[df_symbol['Fecha'].dt.year == year]

        # Extraer variables independientes (X) y dependientes (y)
        X_train = df_train[features]
        y_train = df_train[target_col]
        X_test = df_test[features]
        y_test = df_test[target_col]

        # Modelo XGBoost (se puede cambiar aquí por otro modelo si se desea)
        model = XGBRegressor(
            n_estimators=60,  # número de árboles (ajustable)
            max_depth=4,      # profundidad máxima del árbol (ajustable)
            random_state=42,  # semilla para reproducibilidad
            n_jobs=30         # número de threads paralelos (ajusta según tu CPU)
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calcular métricas de rendimiento en el fold
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Guardar las métricas y tamaños de train/test
        scores.append({
            'year': year,
            'train_hasta_año': year-1,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'n_train': len(X_train),
            'n_test': len(X_test)
        })
    return scores

def main():
    """
    Script principal para comparar el rendimiento de modelos con selección
    aleatoria de features, excluyendo las más correlacionadas con el target.
    """
    # -----------------------------------------------
    # 1. CARGA Y PREPARACIÓN DE DATOS
    # -----------------------------------------------
    df = pd.read_csv("DATA/Dataset_All_Features_Transformado.csv", sep=";", parse_dates=["Fecha"])
    
    # Definimos columnas a excluir: identificadores, precios brutos y columnas target
    cols_a_excluir = [
        'Fecha', 'Symbol', 'Open', 'Close', 'High', 'Low', 'AdjClose', 'Volume'
    ]
    cols_a_excluir += [c for c in df.columns if c.startswith('EMA_')]  # EMAs fuera

    # Filtramos todas las columnas que pueden ser features (no targets, no precios)
    all_features = [c for c in df.columns if c not in cols_a_excluir and not c.startswith('TARGET_')]
    
    # Variable objetivo
    target_col = "TARGET_TREND_ANG_15_5"

    # Años que usaremos como folds de validación cruzada temporal
    years_test = [2020, 2021, 2022, 2023, 2024]
    
    # Símbolo (stock) a analizar
    symbol = "NVDA"

    # Número de repeticiones (tandas de selección aleatoria)
    N_REPS = 10

    # Número de features a seleccionar aleatoriamente en cada tanda
    nfeatures_to_select = 15

    # -----------------------------------------------
    # 2. FILTRADO POR CORRELACIÓN CON EL TARGET
    # -----------------------------------------------
    # Definimos el umbral de correlación máxima permitida (absoluta)
    corr_thresh = 0.4  # Puedes probar 0.4, 0.6, etc.

    # Calculamos la correlación de cada feature con el target (usando todo el dataset)
    # Esto es un filtro para evitar features "trampa" (demasiado relacionadas con la variable objetivo)
    corrs = df[all_features + [target_col]].corr()[target_col].abs().drop(target_col)

    # Nos quedamos solo con features cuya correlación absoluta es inferior al umbral
    features_no_corr = [f for f in all_features if corrs[f] < corr_thresh]

    print(f"\nTras filtrar por correlación (umbral={corr_thresh}), quedan {len(features_no_corr)} features candidatas.")

    # -----------------------------------------------
    # 3. BUCLE PRINCIPAL DE EXPERIMENTOS
    # -----------------------------------------------
    all_rmse = []  # Para guardar el RMSE de cada repetición
    all_r2 = []    # Para guardar el R2 de cada repetición

    print(f"\nResultados para {symbol}: {N_REPS} tandas de selección aleatoria de features ({nfeatures_to_select} cada vez):")

    for rep in range(N_REPS):
        # Cambiamos la seed en cada repetición para que la selección de features aleatorias sea distinta
        np.random.seed(rep * 23)

        # Seleccionamos nfeatures_to_select features aleatorias SIN reposición (replace=False)
        features = [str(f) for f in np.random.choice(features_no_corr, nfeatures_to_select, replace=False)]

        # Ejecutamos la validación cruzada temporal y recogemos métricas
        scores = cross_val_scores_symbol_years(df, symbol, features, target_col, years_test)
        # Calculamos la media de métricas de los folds para cada tanda
        resumen = {k: np.mean([s[k] for s in scores]) for k in ['mse', 'rmse', 'mae', 'r2']}
        all_rmse.append(resumen['rmse'])
        all_r2.append(resumen['r2'])
        print(f"  Tanda {rep+1}: RMSE={resumen['rmse']:.5f}, R2={resumen['r2']:.3f}")

    # -----------------------------------------------
    # 4. RESUMEN FINAL
    # -----------------------------------------------
    print("\n--- Promedio final sobre 10 selecciones aleatorias ---")
    print(f"RMSE medio: {np.mean(all_rmse):.5f} ± {np.std(all_rmse):.5f}")
    print(f"R2 medio: {np.mean(all_r2):.3f} ± {np.std(all_r2):.3f}")

if __name__ == "__main__":
    main()
