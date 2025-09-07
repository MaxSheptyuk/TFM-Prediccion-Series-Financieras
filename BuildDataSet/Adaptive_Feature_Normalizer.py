import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler, FunctionTransformer
from scipy.stats import kurtosis, skew

class AdaptiveFeatureNormalizer:
    """
    Transformador adaptativo de variables numéricas basado en skewness, kurtosis y outliers.
    Selecciona automáticamente la mejor transformación para cada feature:
        - Aplica log-transform (log1p) si el sesgo es muy fuerte y la variable es estrictamente positiva.
        - Aplica Yeo-Johnson si el sesgo es muy fuerte y existen valores negativos.
        - Aplica Box-Cox si el sesgo es moderado y todos los valores son positivos.
        - Aplica Yeo-Johnson si el sesgo es moderado y hay valores negativos.
        - Si la variable es aproximadamente simétrica y sin colas largas, aplica RobustScaler.
    Finalmente, aplica StandardScaler global (media 0, varianza 1).

    El proceso se realiza *stock por stock* para evitar que se mezclen distribuciones de distintos activos.
    La selección es totalmente automática: sólo se usan propiedades estadísticas de cada serie.
    """
    def __init__(self, skew_threshold=0.8, log_skew_threshold=2.0, kurtosis_threshold=4.0):
        
        # Umbral de skewness (sesgos) para considerar transformación logarítmica
        self.skew_threshold = skew_threshold
        
        # Umbral de skewness para considerar transformación logarítmica fuerte
        # (log_skew_threshold > 2 indica un sesgo muy fuerte)
        self.log_skew_threshold = log_skew_threshold

        # Achacamiento o apuntamiento de la distribución.
        # Umbral de kurtosis para considerar colas largas
        # (kurtosis > 3 indica colas más largas que una distribución normal)
        self.kurtosis_threshold = kurtosis_threshold
        
        # Diccionario para almacenar transformadores (objetos como) por columna
        self.transformers = {}
        self.final_scaler = None

    def fit(self, X):

        #reseteamos el diccionario de transformadores
        self.transformers = {}
        X_tr = X.copy()

        for col in X.columns:
            
            # Se quitan las filas con valores NaN para el cálculo
            series = X[col].dropna()
            
            # Calculamos skewness, kurtosis y outliers
            skew_val = series.skew()
            kurt_val = kurtosis(series, fisher=True)
            q1, q3 = np.percentile(series, [25, 75])
            iqr = q3 - q1
            
            # Contamos outliers usando el criterio de 3*IQR
            outliers = ((series < q1 - 3*iqr) | (series > q3 + 3*iqr)).sum()
            
            # Selección automática de transformación
            # El sesgo es muy fuerte?
            if abs(skew_val) > self.log_skew_threshold:
                if (series > 0).all():
                    # Positivo (todos los valores son positivos)
                    transformer = FunctionTransformer(np.log1p, validate=True)
                else:
                    # Existen valores negativos
                    transformer = PowerTransformer(method='yeo-johnson', standardize=False)
            
            # El sesgo es moderado
            elif abs(skew_val) > self.skew_threshold or abs(kurt_val) > self.kurtosis_threshold or outliers > 0:
                if (series > 0).all():
                    # Positivo (todos los valores son positivos)
                    transformer = PowerTransformer(method='box-cox', standardize=False)
                else:
                    transformer = PowerTransformer(method='yeo-johnson', standardize=False)
            else:
                # Aproximadamente simétrico y sin colas largas
                transformer = RobustScaler()

            transformer.fit(series.values.reshape(-1, 1))
            self.transformers[col] = transformer
            X_tr[col] = transformer.transform(X[[col]].values)
        
        # Estandarización global final (media 0, varianza 1)
        self.final_scaler = StandardScaler()
        self.final_scaler.fit(X_tr.values)
        return self

    # Transformación de los datos
    # Aplica las transformaciones definidas a partir de los datos de entrada
    def transform(self, X):
        X_tr = X.copy()

        # Aplicamos las transformaciones creadas en fit
        for col, transformer in self.transformers.items():
            X_tr[col] = transformer.transform(X[[col]].values)
        
        # Estandarización final y devolución del DataFrame transformado
        X_scaled = pd.DataFrame(
            self.final_scaler.transform(X_tr.values),
            columns=X_tr.columns, index=X_tr.index
        )
        return X_scaled
    
    # Método fit_transform que combina fit y transform en una sola llamada
    def fit_transform(self, X):
        return self.fit(X).transform(X)

# =============== Punto de entrada vamos stock por stock ===============

#Leemos el dataset completo (con todos los sombolos y sus indicadores)
df = pd.read_csv("DATA/Dataset_All_Features.csv", parse_dates=['Fecha'], sep=';')

# --- Selección de columnas de indicadores técnicos ---
# Excluimos las columnas de EMA, TARGET, BIN, Fecha, Symbol  y OCHLV
# Las columnas de indicadores son aquellas que no empiezan por 'EMA_', 'TARGET_', 'BIN_' y no son OCHLV
# y no son de tipo objeto (dtype != 'O')
indicadores = [
    col for col in df.columns
    if (
        not col.startswith('EMA_') and
        not col.startswith('TARGET_') and
        not col.startswith('BINARY_') and
        col not in ['Fecha', 'Symbol', 'Open', 'Close', 'High', 'Low', 'Volume']
        and df[col].dtype != 'O'
    )
]

# --- Procesamos stock por stock para evitar el solapamiento de transformación y escalado de indicadores entre distintos stocks ---
dfs_final = []
for symbol, df_sym in df.groupby('Symbol'):
    
    # Seleccionar solo las columnas de indicadores para el símbolo actual
    df_indicadores = df_sym[indicadores].copy()

    auto_tf = AdaptiveFeatureNormalizer()
    df_indicadores_trans = auto_tf.fit_transform(df_indicadores)

    # Volvemos a montar el dataframe completo
    df_resultado = df_sym.copy()
    
    # Pasamos las columnas de indicadores transformadas al dataframe original
    df_resultado[indicadores] = df_indicadores_trans
    
    # Añadimos el DataFrame transformado a la lista de resultados (un dataframe por símbolo)
    dfs_final.append(df_resultado)

df_total = pd.concat(dfs_final, ignore_index=True)

# --- Volcamos el dataset completo transformado ---
df_total.to_csv("DATA/Dataset_All_Features_Transformado.csv", sep=';', index=False)

print(" Dataset completo guardado en DATA/Dataset_All_Features_Transformado.csv\n"
      "Incluye: fechas, Symbol, OCHLV, binarias, targets y features de indicadores transformados.")
