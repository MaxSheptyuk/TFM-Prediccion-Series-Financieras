import pandas as pd
import matplotlib.pyplot as plt

# Cargar archivo CSV
df = pd.read_csv("Resultados/Resultado_Ejecucion_GA_Feature_Selection.csv")
df.columns = df.columns.str.strip()

# Crear figura y ejes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Eje Y izquierdo para R²
color_r2 = 'tab:blue'
ax1.set_xlabel('Generación', fontsize=12)
ax1.set_ylabel('R²', color=color_r2, fontsize=12)
ax1.plot(df['Generacion'], df['R2'], color=color_r2, marker='o', linestyle='-', label='R²')
ax1.tick_params(axis='y', labelcolor=color_r2)
ax1.set_ylim(0.60, 0.72)

# Eje Y derecho para RMSE
ax2 = ax1.twinx()
color_rmse = 'tab:orange'
ax2.set_ylabel('RMSE', color=color_rmse, fontsize=12)
ax2.plot(df['Generacion'], df['RMSE'], color=color_rmse, marker='s', linestyle='--', label='RMSE')
ax2.tick_params(axis='y', labelcolor=color_rmse)
ax2.set_ylim(0.11, 0.13)

# Título y formato
plt.title('Evolución de R² y RMSE por generación (GA - NVDA)', fontsize=14)
fig.tight_layout()
plt.grid(True, linestyle='--', alpha=0.5)

# Exportar como imagen
#plt.savefig("Graficos/R2_RMSE_Evolucion_NVDA.png", dpi=300, bbox_inches='tight')
plt.show()
