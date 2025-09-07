import pandas as pd
import numpy as np
import random
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from GA.GA_Feature_Selection import GA_Feature_Selection

# =============================================================================
# Parámetros globales y configuraciones generales
# =============================================================================
DATASET_PATH = "DATA/Dataset_All_Features_Transformado.csv"
RESULTS_CSV = "Resultados/Comparativa_GA_vs_Aleatorias.csv"

# Lista de símbolos a analizar
STOCKS_TO_ANALYSE  = ["AAPL", "NVDA", "CAT", "BKNG", "LRCX", "REGN", "NKE"]

TARGET_COL = "TARGET_TREND_ANG_15_5"    # Columna objetivo a predecir
N_FEATURES = 15                         # Número de features a seleccionar
N_RANDOM_REP = 10                       # Número de repeticiones aleatorias para comparar (en caso selection aleatoria)
CORR_THRESHOLD = 0.99                   # Umbral de correlación para filtrar features
RANDOM_STATE = 42                       # Semilla para reproducibilidad  


# DEFINICIÓN MANUAL DE LOS FOLDS DE TEST OUT-OF-SAMPLE:
TEST_YEARS_OUT_OF_SAMPLE = [2022, 2023, 2024]
TEST_GA_YEARS = 2  # Ventana de años para el test interno del GA (ejemplo: 2018-2019 si test es 2020)

# =============================================================================
# Función: filtrar_features_correlacion
# =============================================================================
def filtrar_features_correlacion(df, features, target_col, corr_threshold):
    """
    Filtra las features dejando solo aquellas con correlación <= umbral respecto al target.
    """
    corrs = df[features].corrwith(df[target_col]).abs()
    return list(corrs[corrs <= corr_threshold].index)


# =============================================================================
# Función: seleccionar_features_aleatorias
# =============================================================================
def seleccionar_features_aleatorias(features_pool, n_features):
    """
    Selecciona n_features aleatorias SIN repetición desde el conjunto de features disponibles (features_pool).
    """
    if n_features > len(features_pool):
        raise ValueError(f"Intentas seleccionar {n_features} features pero solo hay {len(features_pool)} disponibles.")
    return random.sample(features_pool, n_features)

# =============================================================================
# Función: run_genetic_algorithm_from_splits
# =============================================================================
def run_genetic_algorithm_from_splits(df_train, df_test, features_pool, target_col, n_features, random_state):
    """
    Ejecuta el Algoritmo Genético (GA) para selección de features.
    ATENCIÓN: df_train y df_test deben estar cronológicamente ANTES del año de validación final.
    El GA nunca ve ni una fila de datos futura respecto al año de test final.
    """
    print(f"Procesando GA para símbolo: {df_train['Symbol'].iloc[0]}")
    print(f"  Train GA: {df_train['Fecha'].min().strftime('%d/%m/%Y')} - {df_train['Fecha'].max().strftime('%d/%m/%Y')}")
    print(f"  Test  GA: {df_test['Fecha'].min().strftime('%d/%m/%Y')} - {df_test['Fecha'].max().strftime('%d/%m/%Y')}")
    print(f"  Total train GA: {len(df_train)}, Total test GA: {len(df_test)}")

    # Dividimos features y variable objetivo
    X_train = df_train[features_pool]
    y_train = df_train[target_col]
    X_test = df_test[features_pool]
    y_test = df_test[target_col]

    # Ejecutamos el GA: seleccionará subconjuntos de n_features óptimas 
    ga = GA_Feature_Selection(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=features_pool,
        fitness_model='XGBRegressor',
        fitness_metric='rmse',
        n_pop=25,
        n_gen=20,
        elite=10,
        mut_prob=0.5,
        random_state=random_state,
        max_active=n_features,
        min_active=n_features,
        tournament_size=3
    )
    ga.fit()
    return ga.get_best_features()

# =============================================================================
# BLOQUE PRINCIPAL: Pipeline para cada stock y cada fold definido manualmente
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # 1. Carga del dataset (aseguramos parseo de fechas desde el principio)
    # -------------------------------------------------------------------------
    df = pd.read_csv(DATASET_PATH, sep=";", parse_dates=["Fecha"])
    random.seed(RANDOM_STATE)  # Reproducibilidad en selección aleatoria

    # Preparamos lista de features (excluimos campos técnicos y targets derivados)
    exclude_cols = ['Fecha', 'Symbol', 'Open', 'Close', 'High', 'Low', 'AdjClose', 'Volume']
    exclude_cols += [c for c in df.columns if c.startswith('EMA_')]
    exclude_cols += [c for c in df.columns if c.startswith('TARGET_') and c != TARGET_COL]
    all_features = [c for c in df.columns if c not in exclude_cols]

    resultados = []

    # -------------------------------------------------------------------------
    # 2. Iteramos por cada stock seleccionado
    # -------------------------------------------------------------------------
    for stock in STOCKS_TO_ANALYSE:
        
        print(f"\n=== Procesando stock: {stock} ===")
        
        # Filtramos el DataFrame para el stock actual y ordenamos por fecha
        df_stock = df[df["Symbol"] == stock].copy().sort_values('Fecha')
        
        # Filtro de correlación para evitar features triviales o "semi-leakage"
        features_pool = filtrar_features_correlacion(df_stock, all_features, TARGET_COL, CORR_THRESHOLD)
        print(f"  Features tras filtro de correlación: {len(features_pool)}")

        # ---------------------------------------------------------------------
        # 3. Para cada año de test OUT-OF-SAMPLE definido manualmente (fold)
        # ---------------------------------------------------------------------
        for year_test_final in TEST_YEARS_OUT_OF_SAMPLE:
            
            # Definimos los años previos (test GA = 2 años antes del test final)
            year_test_ga_start = year_test_final - TEST_GA_YEARS

            # -------------------------------------------
            # SPLITS CRONOLÓGICOS:
            # -------------------------------------------
            
            # Train GA: todo hasta justo antes de test GA (pasado puro)
            df_train_ga = df_stock[df_stock['Fecha'].dt.year < year_test_ga_start]
            
            # Test GA: Ventana de años justo antes del test final (ej: 2018-2019 si test=2020)
            df_test_ga = df_stock[
                (df_stock['Fecha'].dt.year >= year_test_ga_start) &
                (df_stock['Fecha'].dt.year < year_test_final)
            ]
            
            # Test final: SOLO el año del fold (out-of-sample absoluto)
            df_test_final = df_stock[df_stock['Fecha'].dt.year == year_test_final]


            print(f"\nFold {year_test_final} para {stock} Train GA: desde inicio hasta < {year_test_ga_start}  Test GA: {year_test_ga_start}-{year_test_final-1}  Test Final (out-of-sample): {year_test_final}")

            # -------------- SELECCIÓN DE FEATURES CON GA --------------
            features_ga = run_genetic_algorithm_from_splits(
                df_train_ga, df_test_ga, features_pool, TARGET_COL, N_FEATURES, RANDOM_STATE
            )
            print(f"    Features GA seleccionadas: {features_ga}")

            # -------------- EVALUACIÓN FINAL (OUT-OF-SAMPLE) --------------
            # El modelo se entrena con todo el pasado y test interno, se evalúa solo en el año de test final
            X_train_model = pd.concat([df_train_ga, df_test_ga])[features_ga]
            y_train_model = pd.concat([df_train_ga, df_test_ga])[TARGET_COL]
            X_test_model = df_test_final[features_ga]
            y_test_model = df_test_final[TARGET_COL]

            model = XGBRegressor(
                n_estimators=60, max_depth=4, n_jobs=-1, random_state=RANDOM_STATE, verbosity=0
            )
            model.fit(X_train_model, y_train_model)
            y_pred = model.predict(X_test_model)
            metrics_ga = {
                "rmse": np.sqrt(mean_squared_error(y_test_model, y_pred)),
                "mae": mean_absolute_error(y_test_model, y_pred),
                "r2": r2_score(y_test_model, y_pred)
            }
            print(f"    Métricas GA (test {year_test_final}): {metrics_ga}")

            # -------------- COMPARATIVA ALEATORIA (BENCHMARK) --------------
            # Entrenamos modelos con features aleatorias para comparar de forma robusta

            metrics_random_list = []
            for i in range(N_RANDOM_REP):
                
                # Seleccionamos N_FEATURES aleatorias del pool
                print(f"    Repetición {i+1}/{N_RANDOM_REP} con {N_FEATURES} features aleatorias ...")
                features_random = seleccionar_features_aleatorias(features_pool, N_FEATURES)
                X_train_model_rnd = pd.concat([df_train_ga, df_test_ga])[features_random]
                y_train_model_rnd = pd.concat([df_train_ga, df_test_ga])[TARGET_COL]
                X_test_model_rnd = df_test_final[features_random]
                y_test_model_rnd = df_test_final[TARGET_COL]

                model = XGBRegressor(
                    n_estimators=60, max_depth=4, n_jobs=-1, random_state=RANDOM_STATE+i, verbosity=0
                )

                model.fit(X_train_model_rnd, y_train_model_rnd)
                y_pred_rnd = model.predict(X_test_model_rnd)
                metrics_random = {
                    "rmse": np.sqrt(mean_squared_error(y_test_model_rnd, y_pred_rnd)),
                    "mae": mean_absolute_error(y_test_model_rnd, y_pred_rnd),
                    "r2": r2_score(y_test_model_rnd, y_pred_rnd)
                }
                metrics_random_list.append(metrics_random)

            # Calculamos la media de métricas de los modelos aleatorios
            metrics_random_mean = {k: np.mean([m[k] for m in metrics_random_list]) for k in metrics_random_list[0]}
            print(f"    Métricas Aleatorias (media de {N_RANDOM_REP} repeticiones): {metrics_random_mean}")

            # -------------- GUARDAR RESULTADOS DEL FOLD (GA y Aleatorias) --------------
            resultados.append({
                "Stock": stock,
                "Test_Year": year_test_final,
                "Metodo": "GA",
                "RMSE": metrics_ga["rmse"],
                "MAE": metrics_ga["mae"],
                "R2": metrics_ga["r2"]
            })

            resultados.append({
                "Stock": stock,
                "Test_Year": year_test_final,
                "Metodo": "Aleatorias",
                "RMSE": metrics_random_mean["rmse"],
                "MAE": metrics_random_mean["mae"],
                "R2": metrics_random_mean["r2"]
            })

    # -------------------------------------------------------------------------
    # 4. Exportamos resultados finales a CSV
    # -------------------------------------------------------------------------
    df_resultados = pd.DataFrame(resultados)
    os.makedirs("Resultados", exist_ok=True)
    df_resultados.to_csv(RESULTS_CSV, index=False)
    print(f"\nResultados guardados en {RESULTS_CSV}")
    print("\n¡Pipeline completo! Puedes ahora visualizar los resultados comparativos.")

# =============================================================================
if __name__ == "__main__":
    main()
