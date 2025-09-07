import pandas as pd
import numpy as np

# Cargar dataset
df = pd.read_csv("DATA/Dataset_All_Features_Transformado.csv", sep=";", parse_dates=["Fecha"])

# Definir columnas a excluir y seleccionar features válidas
cols_a_excluir = ['Fecha', 'Symbol', 'Open', 'Close', 'High', 'Low', 'AdjClose', 'Volume']
cols_a_excluir += [c for c in df.columns if c.startswith('EMA_')]
features = [c for c in df.columns if c not in cols_a_excluir and not c.startswith('TARGET_')]

target_col = "TARGET_TREND_ANG_15_5"

# Calcula correlación de cada feature con el target
corrs = df[features + [target_col]].corr()[target_col].drop(target_col).sort_values(key=np.abs, ascending=False)

# Mostrar top 30
print("\nTop 30 features más correlacionadas con el target:")
print(corrs.head(30))

# Guardar todas las correlaciones (opcional)
corrs.to_csv("resultados/correlaciones_features_target.csv")
