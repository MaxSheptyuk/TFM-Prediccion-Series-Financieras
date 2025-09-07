from random import random, randrange 
import pandas as pd
import numpy as np


# Importamos nuestras clases GA 
from GA.GA_Feature_Selection import GA_Feature_Selection


# ---------------------------
# CONFIGURACIÓN
# ---------------------------

# Umbral de correlación absoluta máxima permitida entre feature y target
# Sin umbral para la versión final
CORR_UMBRAL = 0.99    # Probamos 0.4, 0.6, etc. según nuestro experimento

# Símbolo/stock que vamos a utilizar para el GA
# ----------- CONFIGURACIÓN -----------
SYMBOLS_TO_TEST = ['NVDA', 'AAPL', 'AMZN', 'LRCX', 'SBUX', 'REGN', 'KLAC', 'BKNG', 'AMD', 'VRTX',
                'MAR', 'CDNS',  'CAT', 'INTU', 'GILD',  'MU', 'EBAY', 'AXP', 'AMAT', 'COST', 'MSFT',
                'ORCL', 'ADI', 'MS', 'NKE']

TARGET_COL = 'TARGET_TREND_ANG_15_5'

GA_TRAIN_YEARS = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
GA_TEST_YEARS =  [[2018, 2019],
                 [2018, 2019, 2020], 
                 [2018, 2019, 2020, 2021]]



# Target principal del experimento
target_col = 'TARGET_TREND_ANG_15_5'

# Definimos los años manualmente para Train y Test del GA
train_years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]


 
# ---------------------------
# 1. CARGA Y PREPARACIÓN DEL DATASET
# ---------------------------

# Cargamos el dataset ya con las features generadas y targets calculados
df = pd.read_csv("DATA/Dataset_All_Features_Transformado.csv", parse_dates=['Fecha'], sep=';')


# Ordenamos por símbolo y fecha (importante para series temporales)
df = df.sort_values(['Symbol', 'Fecha']).reset_index(drop=True)

# Excluimos columnas que nunca deben ser usadas como features
cols_a_excluir = [
    'Fecha', 'Symbol', 'Open', 'Close', 'High', 'Low', 'AdjClose', 'Volume'
]
cols_a_excluir += [c for c in df.columns if c.startswith('EMA_')]  # Si quieres quitar EMAs

# Obtenemos la lista completa de features candidatas (sin targets ni columnas excluidas)
all_features = [c for c in df.columns if c not in cols_a_excluir and not c.startswith('TARGET_')]


# ---------------------------
# 3. APLICAMOS DE TRAIN/TEST SPLITS TEMPORALES
# ---------------------------

# Train Para modelo de GA: Todos los simbolos y años de entrenamiento
df_train_alltrain_years = df[df['Fecha'].dt.year.isin(train_years)]



# resultados de test para combinación de simbolo , N_Featuires y años de test 
df_resultados = pd.DataFrame(columns=['Symbol', 'Year_Test', 'N_Features', 'Rows_Train',  'Rnd_State', 'Selected_Features'])
resultados = []

iteracion = 0

for symbol in SYMBOLS_TO_TEST:
    
    for years in GA_TEST_YEARS:
        
        
        
        # Filtramos el DataFrame train dejando sus filas entre 30000 y 50000
        df_train = df_train_alltrain_years.head(randrange(30000, 50000)) 
      
        # Filtramos el DataFrame para el símbolo y año de test
        df_test = df[(df['Symbol'] == symbol) & (df['Fecha'].dt.year.isin(years))]

        # Número aleatorio de features a seleccionar POR GA entre 18 y 35
        N_FEATURES = randrange(18, 36)  
        RND_STATE = randrange(1, 9999)

        # ---------------------------
        # 4. EXTRACCIÓN DE FEATURES X y TARGET Y
        # ---------------------------
        X_train = df_train[all_features]
        y_train = df_train[target_col]
        X_test = df_test[all_features]
        y_test = df_test[target_col]

        print(f"Procesando {symbol} con {N_FEATURES} features, años de test {years} y {df_train.shape[0]} filas de entrenamiento, RND_STATE {RND_STATE}\n")

        # Seleccionamos las N_FEATURES más relevantes según el GA
        ga = GA_Feature_Selection(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=all_features,
            fitness_model='XGBRegressor',
            fitness_metric='rmse',
            n_pop=25,
            n_gen=20,
            elite=10,
            mut_prob=0.5,
            random_state=randrange(1, 9999), 
            max_active=N_FEATURES,
            min_active=N_FEATURES,
            tournament_size=3
        )
        ga.fit(verbose=True)
       
        # Obtenemos las features seleccionadas
        features = ga.get_best_features()
        
        print(f"Features seleccionadas: {features}\n")

        resultados.append({
                    'Symbol': symbol,
                    'Years_Test': years,
                    'N_Features': N_FEATURES,
                    'Rows_Train': df_train.shape[0],
                    'Rnd_State': RND_STATE,
                    'Selected_Features': features
                })
        
        iteracion += 1
        print(f"Iteración {iteracion} de {len(SYMBOLS_TO_TEST) * len(GA_TEST_YEARS)} completada.\n")

# Al final, lo pasamos a un DataFrame:
df_resultados = pd.DataFrame(resultados)

# Convertimos la columna de features a un string separado por comas
df_resultados['Selected_Features'] = df_resultados['Selected_Features'].apply(lambda feats: ','.join(map(str, feats)))
df_resultados['Years_Test'] = df_resultados['Years_Test'].apply(lambda yrs: ','.join(map(str, yrs)))

# Guardamos
df_resultados.to_csv("Resultados/GA_Frecuencia_Features.csv", index=False, sep=';')