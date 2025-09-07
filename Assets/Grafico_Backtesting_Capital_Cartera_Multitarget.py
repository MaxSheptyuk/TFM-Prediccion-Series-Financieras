"""
GRÁFICO DE EVOLUCIÓN DE CAPITAL DE UNA CARTERA (25 ACTIVOS)
Comparando distintas configuraciones de TARGET_TREND_ANG (W, H)

Versión: "muy documentada" para perfiles junior
Objetivo: Dibujar en un mismo gráfico la evolución del capital total de la cartera
         a partir de varios logs de backtesting (uno por configuración W,H).

Resumen de cómo funciona:
1) Cargamos el histórico de precios (solo para construir una "línea de tiempo" estándar).
2) Para cada log (CSV) de trading:
   - Filtramos los símbolos de la cartera.
   - Nos quedamos con las filas de tipo SELL (cierre de operaciones), que suelen tener:
       * 'Profit'       -> Ganancia/Pérdida cerrada en esa fecha
       * 'Capital_Actual' (por símbolo) -> Capital acumulado para ese símbolo tras ese SELL
   - Para **cada símbolo**:
       * Creamos una serie de fechas (según precios) y la unimos con sus SELL.
       * Rellenamos hacia adelante (ffill) el capital de ese símbolo.
   - Sumamos el capital de los 25 símbolos en cada fecha -> "Capital_Total" de la cartera.
3) Dibujamos todas las curvas en el mismo gráfico.

⚠️ Nota importante:
   Este enfoque suma 'Capital_Actual' por símbolo y re-llena con ffill. Es correcto si:
   - 'Capital_Actual' refleja el capital acumulado por símbolo (no solo PnL de la operación).
   - No hay conflictos de fechas (p. ej., distintos símbolos cerrando en días muy diferentes).
   Si algún día necesitas que el "final del gráfico" coincida EXACTO con una tabla de métricas
   basada en cumsum de 'Profit' diario, conviene cambiar la función de curva para usar
   "capital_inicial + cumsum(daily Profit)". (Podemos hacerlo si lo quieres.)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

# ===================== CONFIG =====================
# Capital inicial por activo (ej. 10.000 $ por cada uno de los 25 símbolos).
CAPITAL_INICIAL = 10_000.0

# Lista fija de símbolos que componen la cartera (25 en este ejemplo).
# Tip: Si algún CSV trae otros símbolos, los filtramos con esta lista.
SYMBOLS_CARTERA = [
    'NVDA','AAPL','AMZN','LRCX','SBUX','REGN','KLAC','BKNG','AMD','VRTX',
    'MAR','CDNS','CAT','INTU','GILD','MU','EBAY','AXP','AMAT','COST','MSFT',
    'ORCL','ADI','MS','NKE'
]

# Rango temporal a representar (debe cubrir el periodo de backtesting).
F_FECHA_DESDE = "2020-01-01"
F_FECHA_HASTA = "2024-12-31"

# Ruta al fichero con histórico de precios consolidado (necesario para construir la "línea de tiempo").
# Este CSV debe tener al menos las columnas: ['Fecha', 'Symbol', ...]
PATH_PRECIOS_TODOS = "DATA/AllStocksHistoricalData.csv"

# Diccionario { etiqueta_legenda : ruta_csv_log }
# Cada "ruta_csv_log" debe ser un log de trading unificado que contenga TODOS los símbolos.
# Campos mínimos esperados en esos CSV: ['Fecha', 'Symbol', 'Accion', 'Capital_Actual']
# * 'Accion' se usa para filtrar SELL. * 'Profit' puede existir, pero aquí no lo usamos.
LOGS = {
    "W5_H5":  "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_5_5.csv",
    "W5_H10": "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_5_10.csv",
    "W5_H15": "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_5_15.csv",
    "W10_H5": "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_10_5.csv",
    "W10_H10":"Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_10_10.csv",
    "W10_H15":"Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_10_15.csv",
    "W15_H5": "Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_15_5.csv",
    "W15_H10":"Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_15_10.csv",
    "W15_H15":"Resultados/Trading_Log_AllStocks_TARGET_TREND_ANG_15_15.csv",
    # Ejemplos alternativos:
    # "XGB_ALLFEATS": "Resultados/Trading_Log_AllStocks_XGB_ALLFEATS.csv",
    # "ARIMA":        "Resultados/Trading_Log_ARIMA_AllStocks.csv",
}

# ===================== FUNCIONES AUXILIARES =====================

def build_timeline(df_prices_all: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Construye la "línea de tiempo" común (índice de fechas) a partir del histórico de precios,
    acotada al rango [F_FECHA_DESDE, F_FECHA_HASTA].

    ¿Por qué necesitamos esto?
    - Para que todas las curvas (de todos los símbolos y de todos los logs) queden alineadas
      en las MISMAS fechas. Si no, habría "huecos" o desalineaciones.

    Parámetros:
        df_prices_all : DataFrame con al menos ['Fecha'] y (recomendado) ['Symbol'].

    Retorna:
        DatetimeIndex ordenado y sin duplicados, representando todas las fechas del rango.
    """
    timeline = (
        df_prices_all[
            (df_prices_all["Fecha"] >= F_FECHA_DESDE) &
            (df_prices_all["Fecha"] <= F_FECHA_HASTA)
        ]
        .sort_values("Fecha")[["Fecha"]]
        .drop_duplicates()
        .set_index("Fecha")
        .index
    )
    return timeline


def curva_cartera_desde_log(
    path_log: str,
    df_prices_all: pd.DataFrame,
    timeline: pd.DatetimeIndex
) -> pd.Series:
    """
    Genera la curva de capital TOTAL de la cartera para un log de trading concreto.

    PASOS (por símbolo):
    1) Filtramos el log al símbolo.
    2) Nos quedamos con las filas SELL (cierres), donde suele venir 'Capital_Actual'
       que refleja el capital acumulado para ese símbolo tras ese SELL.
    3) Creamos un DataFrame con todas las 'Fecha' (según precios) para ese símbolo.
    4) Hacemos un merge por 'Fecha' para pegar 'Capital_Actual' en esas fechas.
    5) ffill() para "arrastrar" el último capital conocido hasta la siguiente fecha.
       - Si no hay dato aún, usamos CAPITAL_INICIAL como valor por defecto.
    6) Reindexamos a la línea de tiempo global (timeline) y volvemos a ffill()
       para asegurar que todas las fechas tienen capital.
    7) Guardamos esa serie (una por símbolo) y al final sumamos todas -> "Capital_Total".

    Parámetros:
        path_log       : Ruta al CSV de log (unificado de todos los símbolos).
        df_prices_all  : DataFrame de precios (usado para fechas por símbolo).
        timeline       : DatetimeIndex común para reindexar las curvas.

    Retorna:
        pd.Series con índice = timeline y valores = capital total de la cartera en cada fecha.
    """
    # 0) Seguridad: verificar que el archivo existe antes de leer
    if not os.path.exists(path_log):
        raise FileNotFoundError(f"No existe el archivo de log: {path_log}")

    # 1) Cargar el log y parsear fechas
    trade_log = pd.read_csv(path_log, sep=";", parse_dates=["Fecha"])

    # 2) Filtrar SOLO los símbolos de la cartera (por si el log trae extras)
    trade_log = trade_log[trade_log["Symbol"].isin(SYMBOLS_CARTERA)].copy()

    # 3) Lista para guardar la curva por símbolo
    curvas_por_symbol = []

    # 4) Recorremos cada símbolo de la cartera, generamos su curva
    for sym in SYMBOLS_CARTERA:
        # 4.1) Filtrar el log del símbolo
        tl_sym = trade_log[trade_log["Symbol"] == sym].copy()

        # 4.2) Filtrar SOLO las filas SELL (cierres de trade)
        #      Esto es importante: en SELL es cuando se "cristaliza" el capital.
        tl_sym = tl_sym[tl_sym["Accion"].astype(str).str.startswith("SELL")]

        # 4.3) Extraer las fechas disponibles para este símbolo según precios (no del log).
        #      Usamos precios para tener TODAS las fechas del rango (aunque no haya SELL ese día).
        precios_sym = df_prices_all[
            (df_prices_all["Symbol"] == sym) &
            (df_prices_all["Fecha"] >= F_FECHA_DESDE) &
            (df_prices_all["Fecha"] <= F_FECHA_HASTA)
        ].sort_values("Fecha")[["Fecha"]]

        # 4.4) Unir por 'Fecha' para "pegar" el Capital_Actual del SELL en la fecha correspondiente.
        cap_curve = precios_sym.merge(
            tl_sym[["Fecha", "Capital_Actual"]],
            on="Fecha", how="left"
        )

        # 4.5) Rellenar el capital hacia adelante:
        #      - Si ese día hubo SELL, usamos su 'Capital_Actual'.
        #      - Si no, arrastramos el último capital conocido (ffill).
        #      - Si aún no hay ninguno (principio de la serie), ponemos CAPITAL_INICIAL.
        cap_curve["Capital_Actual"] = (
            cap_curve["Capital_Actual"]
            .ffill()                       # arrastrar último valor conocido
            .fillna(CAPITAL_INICIAL)       # valor inicial si seguimos sin datos
        )

        # 4.6) Pasar a Series con índice Fecha (así es más fácil reindexar al timeline global).
        cap_curve = cap_curve.set_index("Fecha")["Capital_Actual"]

        # 4.7) Alinear la serie al timeline global y volver a ffill (por si faltan días)
        cap_curve = cap_curve.reindex(timeline).ffill().fillna(CAPITAL_INICIAL)

        # 4.8) Guardar la curva de este símbolo
        curvas_por_symbol.append(cap_curve.rename(sym))

    # 5) Concatenar todas las curvas de símbolos por columnas y sumar por filas -> Capital_Total
    df_cartera = pd.concat(curvas_por_symbol, axis=1)
    df_cartera["Capital_Total"] = df_cartera.sum(axis=1)

    # 6) Devolver solo la columna final (serie con el capital de la cartera)
    return df_cartera["Capital_Total"]


# ===================== LECTURA DEL HISTÓRICO Y TIMELINE =====================

# 1) Cargar el histórico de precios (necesario para las fechas por símbolo)
if not os.path.exists(PATH_PRECIOS_TODOS):
    raise FileNotFoundError(
        f"No existe el fichero de precios: {PATH_PRECIOS_TODOS}\n"
        "Asegúrate de que la ruta es correcta y que el CSV contiene al menos las columnas ['Fecha','Symbol']."
    )

df_prices_all = pd.read_csv(PATH_PRECIOS_TODOS, sep=";", parse_dates=["Fecha"])

# 2) Construir la línea de tiempo común (índice de fechas)
timeline = build_timeline(df_prices_all)

# ===================== CÁLCULO DE CURVAS PARA CADA LOG =====================

# Diccionario donde guardaremos { etiqueta : Serie_de_capital_total }
series_por_log = {}

# Recorremos cada configuración definida en LOGS
for etiqueta, ruta in LOGS.items():
    if not os.path.exists(ruta):
        # Aviso claro para diagnóstico si algún CSV no está.
        print(f"[AVISO] No se encontró el log '{etiqueta}': {ruta}. Se omite.")
        continue

    # Generar la curva de capital total de la cartera para ese log
    serie_capital_total = curva_cartera_desde_log(ruta, df_prices_all, timeline)

    # Guardar para graficar más adelante
    series_por_log[etiqueta] = serie_capital_total

# ===================== GRÁFICO =====================

# 1) Crear la figura y el eje
fig, ax = plt.subplots(figsize=(16, 6))

# 2) Dibujar una curva por cada log disponible
for etiqueta, serie in series_por_log.items():
    # 'serie' es una pd.Series con índice de fechas y valores de capital total
    ax.plot(serie.index, serie.values, linewidth=1.7, label=etiqueta)

# 3) Títulos y ejes
ax.set_title("Evolución de capital de cartera — Comparativa por TARGET_TREND_ANG (W,H)")
ax.set_ylabel("Capital ($)")
ax.set_xlim(pd.to_datetime(F_FECHA_DESDE), pd.to_datetime(F_FECHA_HASTA))
ax.grid(True, alpha=0.3)

# 4) Leyenda (ncol=3 para que quepa mejor si hay muchas etiquetas)
ax.legend(ncol=3, frameon=True)

# 5) Eje X: formateo por meses (trimestral) y rotación
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))              # tick cada 3 meses
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))              # formato 'YYYY-MM'
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")                  # girar etiquetas para que no se monten

# 6) Eje Y: formateo de miles con punto (estilo europeo)
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(round(x)):,}".replace(",", "."))
)

# 7) Ajustar márgenes para que no se corte nada
plt.tight_layout()

# 8) Mostrar en pantalla (si quieres guardar, descomenta las siguientes líneas)
# plt.savefig("Resultados/Fig_Comparativa_Cartera_TARGET_TREND_ANG.png", dpi=300, bbox_inches="tight")
# plt.savefig("Resultados/Fig_Comparativa_Cartera_TARGET_TREND_ANG.pdf", bbox_inches="tight")
plt.show()


# ===================== PREGUNTAS FRECUENTES (FAQ) =====================
# P: ¿Por qué usamos 'SELL' y no 'BUY'?
# R: Porque en SELL es cuando se consolida el PnL y se actualiza el 'Capital_Actual'
#    del símbolo. Usar BUY no reflejaría cambios de capital realizados.

# P: ¿Qué pasa si un símbolo no tiene ningún SELL en el periodo?
# R: Gracias a ffill() + fillna(CAPITAL_INICIAL), su capital se mantiene igual (capital inicial)
#    durante todo el periodo, y la suma de la cartera sigue siendo consistente.

# P: ¿Por qué puede salir un "shape" muy parecido entre curvas?
# R: Si la lógica de gestión (TP/SL, tamaño de posición, ventanas) es muy parecida,
#    y compartimos mismo universo y fechas, es normal que las curvas se parezcan
#    en las caídas/recuperaciones, aunque difieran en magnitud final.

# P: Quiero que el punto final del gráfico coincida EXACTO con una tabla basada en cumsum de Profit, ¿qué hago?
# R: Cambia la función 'curva_cartera_desde_log' por una basada en:
#    equity = capital_inicial_total + cumsum(Profit diario agregado a nivel cartera).
#    (Te puedo pasar esa versión si la necesitas; ya la tenemos preparada.)
