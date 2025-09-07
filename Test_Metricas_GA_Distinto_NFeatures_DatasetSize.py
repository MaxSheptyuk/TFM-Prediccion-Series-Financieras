import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from GA.GA_Feature_Selection import GA_Feature_Selection

def filtrar_features_correlacion(df, features, target_col, corr_threshold):
    corrs = df[features].corrwith(df[target_col]).abs()
    return list(corrs[corrs <= corr_threshold].index)



SYMBOL_TEST = 'BKNG'
TARGET_COL = 'TARGET_TREND_ANG_15_5'
CORR_THRESHOLD = 0.6

GA_TRAIN_YEARS = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
GA_TEST_YEARS = [2020, 2021, 2022, 2023]

# Out-of-sample test y train years
OUT_OF_SAMPLE_TEST_YEARS = [2023, 2024]
OUT_OF_SAMPLE_TRAIN_YEARS = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]

N_FEATURES_LIST = [5,  10, 20, 30, 40]
DATASET_SIZES = [10000, 20000, 30000, 50000, 60000]

df = pd.read_csv("DATA/Dataset_All_Features_Transformado.csv", parse_dates=['Fecha'], sep=';')
df = df.sort_values(['Symbol', 'Fecha']).reset_index(drop=True)
cols_a_excluir = ['Fecha', 'Symbol', 'Open', 'Close', 'High', 'Low', 'AdjClose', 'Volume']
cols_a_excluir += [c for c in df.columns if c.startswith('EMA_')]   
all_features = [c for c in df.columns if c not in cols_a_excluir and not c.startswith('TARGET_')]

# Filtro de correlación para evitar features triviales o "semi-leakage"
features_pool = filtrar_features_correlacion(df, all_features, TARGET_COL, CORR_THRESHOLD)
print(f"  Features tras filtro de correlación: {len(features_pool)}")




resultados = []

for n_features in N_FEATURES_LIST:
    for dataset_size in DATASET_SIZES:
        
        # --- Split ---
        df_train = df[df['Fecha'].dt.year.isin(GA_TRAIN_YEARS)].copy().head(dataset_size)
        df_test = df[(df['Symbol'] == SYMBOL_TEST) & (df['Fecha'].dt.year.isin(GA_TEST_YEARS))]

        X_train = df_train[features_pool]
        y_train = df_train[TARGET_COL]
        X_test = df_test[features_pool]
        y_test = df_test[TARGET_COL]

        ga = GA_Feature_Selection(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=features_pool,
            fitness_model='XGBRegressor',
            fitness_metric='r2',
            n_pop=25,
            n_gen=20,
            elite=10,
            mut_prob=0.5,
            random_state=100,
            max_active=n_features,
            min_active=n_features, 
            tournament_size=3
        )
        ga.fit(verbose=True)
        

        # 1. GA obtiene best_features = [lista de nombres de features]
        best_features = ga.get_best_features()

        # 2. Split de datos para el test real fuera de muestra
        df_outsample_train = df[df['Fecha'].dt.year.isin(OUT_OF_SAMPLE_TRAIN_YEARS)]  
        df_outsample_test =  df[(df['Symbol'] == SYMBOL_TEST) & (df['Fecha'].dt.year.isin(OUT_OF_SAMPLE_TEST_YEARS))]

        X_train = df_outsample_train[best_features]
        y_train = df_outsample_train[TARGET_COL]

        X_test = df_outsample_test[best_features]
        y_test = df_outsample_test[TARGET_COL]

        # 3. Entrenamiento y test final con XGBoost
        model = XGBRegressor(n_estimators=60, max_depth=4, n_jobs=32,
                                     random_state=42, verbosity=0)
        
        # 4. Entrenamos el modelo con las mejores features seleccionadas        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 5. Evaluamos el modelo con las métricas RMSE, MAE y R²
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"RMSE OUT-OF-SAMPLE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}\n")
        print(f"features seleccionadas: {best_features}\n")

        # 6. Guardamos TODAS las métricas necesarias, incluyendo la del GA para análisis futuros
        rmse_best = ga.best_score
        resultados.append({
            'n_features': n_features,
            'dataset_size': dataset_size,
            'rmse_out_of_sample': rmse,
            'mae_out_of_sample': mae,
            'r2_out_of_sample': r2,
            'mse_out_of_sample': mse,
            'rmse_best_ga': rmse_best,       # Métrica del mejor individuo en el GA
            'features_selected': best_features
        })        

df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv(f"Resultados/Benchmark_GA_Automatico_{SYMBOL_TEST}.csv", index=False)
print(df_resultados)
