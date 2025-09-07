import pandas as pd

# Comprobamos la coherencia del dataset multi-activos:  dónde cada símbolo tiene que tener las mismas fechas y el mismo número de filas

# Cargamos el dataset
df = pd.read_csv("DATA/Dataset_All_Features_Transformado.csv", sep=";", parse_dates=["Fecha"])

# 1. Chequear número de filas por símbolo
group_counts = df.groupby("Symbol")["Fecha"].count()
print("Filas por símbolo:")
print(group_counts)
print("\n¿Todos tienen el mismo número?", group_counts.nunique() == 1)

# 2. Obtener la lista ordenada de fechas de referencia (usamos el primer símbolo como referencia)
symbols = sorted(df["Symbol"].unique())
ref_symbol = symbols[0]
ref_dates = df[df["Symbol"] == ref_symbol].sort_values("Fecha")["Fecha"].reset_index(drop=True)

# 3. Comprobar para cada símbolo que las fechas cuadran exactamente
coherente = True
for symbol in symbols:
    dates = df[df["Symbol"] == symbol].sort_values("Fecha")["Fecha"].reset_index(drop=True)
    if not ref_dates.equals(dates):
        print(f" **INCOHERENCIA EN FECHAS para {symbol}**")
        coherente = False

if coherente:
    print("\n ¡Todos los símbolos tienen exactamente las mismas fechas y número de filas!")
else:
    print("\n Algún símbolo tiene fechas diferentes. ¡Revisa el log arriba!")

# Opcional: Mostramos símbolos con incoherencia de fechas o de longitud
if group_counts.nunique() != 1:
    print("\n Símbolos con número de filas diferente:")
    print(group_counts[group_counts != group_counts.max()])
