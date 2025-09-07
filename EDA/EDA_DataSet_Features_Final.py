import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Carga dataset ---
df = pd.read_csv("DATA/Dataset_All_Features.csv", parse_dates=['Fecha'],   sep=';')


# --- Ajuste visual general ---
pd.set_option('display.max_columns', 100)
print("Shape:", df.shape)
print(df.head())
print("\n--- Info ---")
print(df.info())
print("\n--- Nulls por columna ---")
print(df.isnull().sum())

# --- Descriptivo general EXCLUYENDO BINARY_ ---
cols_nobin = [c for c in df.columns if not c.startswith('BINARY_')]
print("\n--- Estadísticas Descriptivas (sin BINARY_) ---")
print(df[cols_nobin].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T)

# --- Histograma de variables target (excluyendo BINARY_) ---
for col in [c for c in cols_nobin if c.startswith('TARGET_SLOPE_')]:
    plt.figure(figsize=(10,5))
    plt.hist(df[col].dropna(), bins=40, color='skyblue', alpha=0.7, edgecolor='k')
    plt.title(f"Histograma de {col}")
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
    plt.grid(True, alpha=0.2)
    plt.show()

# --- Matriz de correlación rápida (numéricas, sin BINARY_) ---
numeric_cols = [c for c in cols_nobin if pd.api.types.is_numeric_dtype(df[c])]
plt.figure(figsize=(18,8))
sns.heatmap(df[numeric_cols].corr(), cmap='coolwarm', center=0, annot=False)
plt.title("Matriz de correlación numérica (sin BINARY_)")
plt.tight_layout()
plt.show()

# --- Distribución por símbolo y fechas ---
if 'Symbol' in df.columns and 'Fecha' in df.columns:
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df_group = df.groupby('Symbol')['Fecha'].agg(['min', 'max', 'count'])
    print("\n--- Rango de fechas por símbolo ---")
    print(df_group)

# --- Outliers en variables objetivo (sin BINARY_) ---
for col in [c for c in cols_nobin if c.startswith('TARGET_TREND')]:
    q01 = df[col].quantile(0.01)
    q99 = df[col].quantile(0.99)
    print(f"\n[{col}] 1%: {q01:.3f}   99%: {q99:.3f}")

    plt.figure(figsize=(8, 2))
    sns.boxplot(data=df, x=col)
    plt.title(f"Boxplot de {col}")
    plt.show()

# --- Conteo de extremos en targets (sin BINARY_) ---
for col in [c for c in cols_nobin if c.startswith('TARGET_TREND')]:
    extremos_0 = (df[col] <= 0.01).sum()
    extremos_1 = (df[col] >= 0.99).sum()
    print(f"{col} -- Valores cercanos a 0: {extremos_0}, a 1: {extremos_1}")

print("\n--- Fin del EDA ---")
