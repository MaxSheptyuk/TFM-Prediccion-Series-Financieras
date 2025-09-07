import pandas as pd
import numpy as np
import os
from GA.GA_Feature_Selection import GA_Feature_Selection
from Tests_Prediccion.Test_Prediccion_Metricas_KFoldCV import cross_val_scores_symbol_years

# =======================
# Configuración general del experimento
# =======================
DATASET_PATH = "DATA/Dataset_All_Features_Transformado.csv"
RESULTS_CSV = "Resultados/Comparativa_GA_Multimodelo_KFold.csv"

# Lista de stocks y modelos a analizar
STOCKS_TO_ANALYSE = ["NVDA"]  
MODELS_TO_ANALYSE = ["XGBRegressor", "ElasticNet", "Ridge", "SVR", "Lasso", "LinearRegression", "MLP-Torch"]

TARGET_COL = "TARGET_TREND_ANG_15_5"
N_FEATURES = 25
CORR_THRESHOLD = 0.99
RANDOM_STATE = 42

# Definimos los folds de K-Fold CV (3 folds)
KFOLDS = [
    {"TRAIN_GA_YEARS": list(range(2010, 2020)), "TEST_GA_YEARS": [2020, 2021], "CROSSVAL_YEARS": [2022]},
    {"TRAIN_GA_YEARS": list(range(2010, 2021)), "TEST_GA_YEARS": [2021, 2022], "CROSSVAL_YEARS": [2023]},
    {"TRAIN_GA_YEARS": list(range(2010, 2022)), "TEST_GA_YEARS": [2022, 2023], "CROSSVAL_YEARS": [2024]}
]

def filtrar_features_correlacion(df, features, target_col, corr_threshold):
    corrs = df[features].corrwith(df[target_col]).abs()
    return list(corrs[corrs <= corr_threshold].index)

# Se ejecuta el algoritmo genético para la selección de features
# y se devuelve la lista de las mejores features seleccionadas.
def run_genetic_algorithm(df_train, df_test, features_pool, target_col, n_features, random_state, model_name):
    ga = GA_Feature_Selection(
        X_train=df_train[features_pool],
        y_train=df_train[target_col],
        X_test=df_test[features_pool],
        y_test=df_test[target_col],
        feature_names=features_pool,
        fitness_model=model_name,
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


#Punto de entrada para el experimento
def main():
    
    df = pd.read_csv(DATASET_PATH, sep=";", parse_dates=["Fecha"])
    exclude_cols = ['Fecha', 'Symbol', 'Open', 'Close', 'High', 'Low', 'AdjClose', 'Volume']
    exclude_cols += [c for c in df.columns if c.startswith('EMA_')]
    exclude_cols += [c for c in df.columns if c.startswith('TARGET_') and c != TARGET_COL]
    all_features = [c for c in df.columns if c not in exclude_cols]

    resultados = []

    # Iteramos sobre cada stock
    for stock in STOCKS_TO_ANALYSE:
        
        print(f"\n=== Procesando stock: {stock} ===")
        df_stock = df[df["Symbol"] == stock].copy().sort_values('Fecha')
        features_pool = filtrar_features_correlacion(df_stock, all_features, TARGET_COL, CORR_THRESHOLD)
        print(f"  Features tras filtro de correlación: {len(features_pool)}")
        
        # Iteramos sobre cada fold
        for fold_idx, fold in enumerate(KFOLDS):
            
            print(f"\nFold {fold_idx + 1}:")
            df_train_ga = df_stock[df_stock['Fecha'].dt.year.isin(fold["TRAIN_GA_YEARS"])]
            df_test_ga = df_stock[df_stock['Fecha'].dt.year.isin(fold["TEST_GA_YEARS"])]
            # El test final es la validación cruzada out-of-sample
            years_test = fold["CROSSVAL_YEARS"]

            # Iteramos sobre cada modelo
            for model_name in MODELS_TO_ANALYSE:
                
                print(f"Modelo: {model_name}")
                
                # 1. Selección de features con GA
                best_features = run_genetic_algorithm(
                    df_train_ga, df_test_ga, features_pool, TARGET_COL, N_FEATURES, RANDOM_STATE, model_name
                )
                print(f"  Features seleccionadas ({model_name}): {best_features}")

                # 2. Validación cruzada temporal usando función modular
                scores, _ = cross_val_scores_symbol_years(
                    df, stock, best_features, TARGET_COL, years_test, model_name
                )

                # Guardar métricas por fold (puedes promediar aquí si tienes varios años en years_test)
                for s in scores:
                    resultados.append({
                        "Stock": stock,
                        "Modelo": model_name,
                        "Fold": fold_idx + 1,
                        "Año_Test": s['year'],
                        "N_features": len(best_features),
                        "RMSE": s["rmse"],
                        "MAE": s["mae"],
                        "R2": s["r2"]
                    })

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(RESULTS_CSV, index=False)
    print(f"\nResultados guardados en {RESULTS_CSV}")

if __name__ == "__main__":
    main()
