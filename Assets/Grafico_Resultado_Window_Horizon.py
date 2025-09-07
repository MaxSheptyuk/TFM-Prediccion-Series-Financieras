import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", font_scale=1.15)

metrics_df = pd.read_csv("RESULTADOS/Resultados_Window_Horizon.csv")
metrics_df['Window'] = metrics_df['Target'].str.extract(r'_(\d+)_')[0].astype(int)
metrics_df['Horizon'] = metrics_df['Target'].str.extract(r'_(\d+)$')[0].astype(int)
df_win10 = metrics_df[metrics_df['Window'] == 15]

plt.figure(figsize=(10,6))
palette = sns.color_palette("Set2", n_colors=len(df_win10['Symbol'].unique()))

for idx, symbol in enumerate(df_win10['Symbol'].unique()):
    sub = df_win10[df_win10['Symbol'] == symbol].sort_values('Horizon')
    plt.plot(sub['Horizon'], sub['rmse'], marker='o', label=symbol, color=palette[idx], linewidth=2)
    # Numeritos solo donde hay punto
    for x, y in zip(sub['Horizon'], sub['rmse']):
        plt.text(x, y+0.003, f"{y:.3f}", fontsize=10, color=palette[idx], ha='center')

plt.title('RMSE vs Horizonte para cada stock (Ventana=15)', fontsize=17, fontweight='bold', pad=15)
plt.xlabel('Horizonte de predicción (días)', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.xticks(sorted(df_win10['Horizon'].unique()), fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Stock', fontsize=11, title_fontsize=12, loc='best', frameon=True)
plt.grid(axis='both', alpha=0.25, linestyle='--')
plt.tight_layout()
plt.gcf().set_facecolor('white')
plt.show()
