import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Cargar el dataset
df = pd.read_csv("DATA/Dataset_All_Features_Transformado.csv", sep=';', parse_dates=['Fecha'])

# 2. Definir columnas a excluir: fecha, símbolo, variables target y variables binarias
excluir = ['Fecha', 'Symbol'] + \
          [col for col in df.columns if col.startswith('TARGET_')] + \
          [col for col in df.columns if col.startswith('BINARY_')]

# 3. Obtener lista de variables no binarias (continuas)
features_continuas = [col for col in df.columns if col not in excluir]
print(f"Variables no binarias encontradas ({len(features_continuas)}):")
print(features_continuas)

# 4. Estadísticas descriptivas para variables continuas
desc_stats = df[features_continuas].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
desc_stats['null_pct'] = df[features_continuas].isna().mean()
print("\nEstadísticas descriptivas:")
print(desc_stats)

# 5. Seleccionar variable target a analizar
target_col = 'TARGET_TREND_ANG_15_5'

# 6. Calcular correlaciones (Pearson y Spearman) con la variable target
corr_pearson = df[features_continuas + [target_col]].corr(method='pearson')[target_col].sort_values(ascending=False)
corr_spearman = df[features_continuas + [target_col]].corr(method='spearman')[target_col].sort_values(ascending=False)

print("\nCorrelación Pearson con target:")
print(corr_pearson)

print("\nCorrelación Spearman con target:")
print(corr_spearman)

# 7. Visualización scatter plot para las top 5 variables más correlacionadas (por valor absoluto)
top_vars = corr_pearson.abs().sort_values(ascending=False).index[1:6]  # excluir el target mismo

for var in top_vars:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[var], df[target_col], alpha=0.3, s=10)
    plt.title(f'{var} vs {target_col} (corr Pearson={corr_pearson[var]:.2f})')
    plt.xlabel(var)
    plt.ylabel(target_col)
    plt.grid(True)
    plt.show()
