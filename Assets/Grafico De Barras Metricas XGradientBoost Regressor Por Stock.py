import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Leer el CSV y eliminar duplicados
df = pd.read_csv("DATA/Metricas_XGBoostRegressor_Promedio_Por_Symbol_CV.csv")
df = df.drop_duplicates(subset=["Symbol"])
df = df.sort_values('rmse', ascending=True)

# Cambia el estilo global
plt.style.use('seaborn-v0_8-whitegrid')

fig, ax = plt.subplots(figsize=(15, 7))  # Más grande

bars = ax.bar(df['Symbol'], df['rmse'], color='#51b7f7', edgecolor='#174d74', linewidth=1.5)

# Efecto 3D suave (opcional)
for bar in bars:
    bar.set_zorder(3)

# Agregar valores redondeados sobre las barras (fuente más grande)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 8),  # 8 puntos vertical para más separación
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=13, fontweight='bold', color='#174d74')

# Mejoras visuales y de estilo (fuentes grandes)
ax.set_title('RMSE promedio por stock\n(XGBoost Regressor con validación cruzada cronológica)', 
             fontsize=20, fontweight='bold', pad=24)
ax.set_xlabel('Stock', fontsize=16, fontweight='bold', labelpad=10)
ax.set_ylabel('RMSE promedio', fontsize=16, fontweight='bold', labelpad=10)
ax.set_ylim(0, df['rmse'].max() + 0.03)
ax.tick_params(axis='x', labelrotation=55, labelsize=14)
ax.tick_params(axis='y', labelsize=15)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.02f'))

# Eliminar bordes superiores y derechos
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
