import pandas as pd
import numpy as np


# Importa tus clases GA (asegúrate de que los paths sean correctos)
from GA.GA_Feature_Selection_ROI import GA_Feature_Selection_ROI
from GA.GA_Feature_Selection import GA_Feature_Selection


# ---------------------------
# CONFIGURACIÓN
# ---------------------------

# Umbral de correlación absoluta máxima permitida entre feature y target
# Sin umbral para la versión final
CORR_UMBRAL = 0.99    # Probamos 0.4, 0.6, etc. según nuestro experimento

# Símbolo/stock que vamos a analizar
symbol_test = 'MU'

# Target principal del experimento
target_col = 'TARGET_TREND_ANG_15_5'

# Definimos los años manualmente para Train y Test del GA
train_years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]

# Para fitness del GA (Ojo! no es test de rendimiento sino el test de selección de features dentro de GA)
test_years =  [2020, 2021]  

 
# ---------------------------
# 1. CARGA Y PREPARACIÓN DEL DATASET
# ---------------------------

# Cargamos el dataset ya con las features generadas y targets calculados
df = pd.read_csv("DATA/Dataset_All_Features_Transformado.csv", parse_dates=['Fecha'], sep=';')

print(df.shape) 

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
# 2. FILTRADO POR CORRELACIÓN 
# ---------------------------
corrs = df[all_features + [target_col]].corr()[target_col].abs().drop(target_col)

# Seleccionamos solo features con correlación < umbral definido
features = [f for f in all_features if corrs[f] < CORR_UMBRAL]

print(f"Filtradas por correlación | Umbral: {CORR_UMBRAL}")
print(f"Features disponibles para selección: {len(features)} de {len(all_features)} posibles. \n")

# ---------------------------
# 3. APLICAMOS DE TRAIN/TEST SPLITS TEMPORALES
# ---------------------------


# Train Para modelo de GA: Todos los simbolos y años de entrenamiento
df_train = df[df['Fecha'].dt.year.isin(train_years)]

# Test para modelo de GA: datos del símbolo objetivo 
df_test = df[(df['Symbol'] == symbol_test) &
             (df['Fecha'].dt.year.isin(test_years))]


# ---------------------------
# 4. EXTRACCIÓN DE FEATURES X y TARGET Y
# ---------------------------
X_train = df_train[features]
y_train = df_train[target_col]
X_test = df_test[features]
y_test = df_test[target_col]

# ---------------------------
# 5. CONFIGURACIÓN Y LANZAMIENTO DEL GA
# ---------------------------

# GA para selección de features con XGBRegressor
# ga = GA_Feature_Selection(
#     X_train=X_train,
#     y_train=y_train,
#     X_test=X_test,
#     y_test=y_test,
#     feature_names=features,      # Las features disponibles después del filtro de correlación
#     fitness_model='XGBRegressor',   # El modelo a utilizar 'XGBRegressor', 'ElasticNet', 'Ridge', 'MLP', 'MLP-Torch', 'SVR', 'Lasso', 'LinearRegression' 
#     fitness_metric='rmse',
#     n_pop=25,          # Población inicial del GA
#     n_gen=20,         # Nº de generaciones
#     elite=10,          # Número de mejores individuos que pasan directamente
#     mut_prob=0.5,      # Probabilidad de mutación
#     random_state=100,  # Semilla para reproducibilidad
#     max_active=25,      # Nº máximo de features seleccionadas
#     min_active=25,      # Nº mínimo de features seleccionadas
#     tournament_size=3,   # Tamaño de torneo para selección
# )



# --- Creamos el GA que selecciona features maximizando el ROI obtenido en backtesting ---
ga = GA_Feature_Selection_ROI(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=features,
    df_test=df_test,            # OJO: aquí el dataframe de test (del símbolo a realizar el backtesting, periodo test GA)
    symbol=symbol_test,
    threshold_buy=0.51,
    paciencia_max_dias=5,
    capital_inicial=10000,
    tp_pct=0.015,
    sl_pct=0.03,
    n_pop=25,
    n_gen=20,
    elite=10,
    mut_prob=0.5,
    random_state=100,
    max_active=25,
    min_active=25,
    tournament_size=3
)


# Versión de MLP-Torch, 
# ga = GA_Feature_Selection(
#     X_train=X_train,
#     y_train=y_train,
#     X_test=X_test,
#     y_test=y_test,
#     feature_names=features,      # Las features disponibles después del filtro de correlación
#     fitness_model='MLP-Torch',   # El modelo a utilizar 'XGBRegressor', 'ElasticNet', 'Ridge', 'MLP', 'MLP-Torch'
#     fitness_metric='rmse',
#     n_pop=25,          # Población inicial del GA
#     n_gen=20,          # Nº de generaciones
#     elite=10,          # Número de mejores individuos que pasan directamente
#     mut_prob=0.5,      # Probabilidad de mutación
#     random_state=100,  # Semilla para reproducibilidad
#     max_active=25,      # Nº máximo de features seleccionadas
#     min_active=25,      # Nº mínimo de features seleccionadas
#     tournament_size=3   # Tamaño de torneo para selección
# )




# ---------------------------
# 6. EJECUCIÓN DEL GA
# ---------------------------

print(f"Lanzando Algoritmo Genético (GA) para selección de features de stock {symbol_test} utilizando modelo {ga.fitness_model} ...")
ga.fit()

# ---------------------------
# 7. RESUMEN DE FEATURES SELECCIONADAS
# ---------------------------
ga.summary()


