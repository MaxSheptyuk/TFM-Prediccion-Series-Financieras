import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from GA.GA_Feature_Selection import GA_Feature_Selection

# =========================
# CONFIGURACIÓN
# =========================
SYMBOLS_TO_TEST = ['NVDA', 'AAPL', 'AMZN', 'LRCX', 'SBUX', 'REGN', 'KLAC', 'BKNG', 'AMD', 'VRTX',
                   'MAR', 'CDNS', 'CAT', 'INTU', 'GILD', 'MU', 'EBAY', 'AXP', 'AMAT', 'COST',
                   'MSFT', 'ORCL', 'ADI', 'MS', 'NKE']


TARGETS =  [
 'TARGET_TREND_ANG_5_5',
 'TARGET_TREND_ANG_5_8',
 'TARGET_TREND_ANG_5_10',
 'TARGET_TREND_ANG_5_12',
 'TARGET_TREND_ANG_5_15',
 'TARGET_TREND_ANG_10_5',
 'TARGET_TREND_ANG_10_8',
 'TARGET_TREND_ANG_10_10',
 'TARGET_TREND_ANG_10_12',
 'TARGET_TREND_ANG_10_15',
 'TARGET_TREND_ANG_15_5',
 'TARGET_TREND_ANG_15_8',
 'TARGET_TREND_ANG_15_10',
 'TARGET_TREND_ANG_15_12',
 'TARGET_TREND_ANG_15_15'
]


N_FEATURES = 25

GA_TRAIN_YEARS = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
GA_TEST_YEARS  = [2018, 2019]

DATA_PATH = "DATA/Dataset_All_Features_Transformado.csv"
RESULTS_DIR = "Resultados"  # guardamos 1 CSV por target

# =========================
# FUNCIONES
# =========================
def split_por_anios(df, years, symbol=None):
    df_split = df[df['Fecha'].dt.year.isin(years)]
    if symbol:
        df_split = df_split[df_split['Symbol'] == symbol]
    return df_split

def get_all_features(df):
    cols_a_excluir = ['Fecha', 'Symbol', 'Open', 'Close', 'High', 'Low', 'AdjClose', 'Volume']
    cols_a_excluir += [c for c in df.columns if c.startswith('EMA_')]
    # Excluir cualquier TARGET_ del set de features
    all_features = [c for c in df.columns if c not in cols_a_excluir and not c.startswith('TARGET_')]
    return all_features

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # 0) Preparar salida
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) Cargar dataset (única vez)
    df = pd.read_csv(DATA_PATH, parse_dates=["Fecha"], sep=';')
    df = df.sort_values(['Symbol', 'Fecha']).reset_index(drop=True)

    # 2) Determinar lista de features disponibles (única vez)
    all_features = get_all_features(df)

    print(f"Dataset cargado: {DATA_PATH}")
    print(f"Total filas: {len(df):,} | Total features candidatas: {len(all_features)}")
    print(f"Símbolos a procesar ({len(SYMBOLS_TO_TEST)}): {SYMBOLS_TO_TEST}")
    
    # Validación rápida de targets antes del loop
    if not set(TARGETS).issubset(df.columns):
        faltan = set(TARGETS) - set(df.columns)
        raise ValueError(f"[ERROR] Faltan targets en el dataset: {faltan}")

    # 3) Loop por TARGET
    for TARGET_COL in TARGETS:
        print("\n" + "="*80)
        print(f"=== TARGET: {TARGET_COL} ===")
        print("="*80)

        if TARGET_COL not in df.columns:
            print(f"[ADVERTENCIA] El target {TARGET_COL} no existe en el dataset. Se omite.")
            continue

        filas_salida = []  # reiniciar para este target

        # 3.1) Proceso por stock
        for SYMBOL_TEST in SYMBOLS_TO_TEST:
            print(f"\n--- {SYMBOL_TEST} ---")
            print(f"Años GA train: {GA_TRAIN_YEARS}, GA test: {GA_TEST_YEARS}")

            # Splits (train con todos los símbolos; test sólo símbolo actual)
            df_ga_train = split_por_anios(df, GA_TRAIN_YEARS)                  # pool multi-símbolo
            df_ga_test  = split_por_anios(df, GA_TEST_YEARS, symbol=SYMBOL_TEST)

            # Si el test está vacío para el símbolo/años, avisar y continuar
            if df_ga_test.empty:
                print(f"[AVISO] No hay datos de TEST para {SYMBOL_TEST} en años {GA_TEST_YEARS}. Se omite.")
                continue

            # Matrices
            X_train_ga = df_ga_train[all_features]
            y_train_ga = df_ga_train[TARGET_COL]
            X_test_ga  = df_ga_test[all_features]
            y_test_ga  = df_ga_test[TARGET_COL]

            # GA Feature Selection (idéntico a tu configuración actual)
            ga = GA_Feature_Selection(
                X_train=X_train_ga,
                y_train=y_train_ga,
                X_test=X_test_ga,
                y_test=y_test_ga,
                feature_names=all_features,
                fitness_model='XGBRegressor',
                fitness_metric='rmse',
                n_pop=25,
                n_gen=20,
                elite=10,
                mut_prob=0.5,
                random_state=42,
                max_active=N_FEATURES,
                min_active=N_FEATURES,
                tournament_size=3
            )
            ga.fit(verbose=True)
            best_features = ga.get_best_features()

            print(f"Features seleccionadas ({len(best_features)}): {best_features}")

            # Guardar fila
            filas_salida.append([SYMBOL_TEST, ", ".join(best_features)])

        # 3.2) Guardar resultado GLOBAL por TARGET
        fname = f"Features_Seleccionadas_GA_{TARGET_COL}.csv"
        results_path = os.path.join(RESULTS_DIR, fname)

        df_out = pd.DataFrame(filas_salida, columns=['Stock', 'Features'])
        df_out.to_csv(results_path, sep=';', index=False, encoding='utf-8')

        print(f"\nOK -> guardado: {results_path}")

    print("\nProceso multi-target COMPLETADO ✔")
