import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from stock_indicators import indicators
from stock_indicators import Quote
from Feature_Generator import Feature_Generator
import traceback
import os 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time

# ---------------------------
# Función parcial para ejecución en paralelo
# ---------------------------
def procesar_tupla(tupla):
    symbol, df_sub = tupla
    return procesar_symbol(df_sub, symbol)

# ---------------------------
# Función para procesar cada símbolo 
# ---------------------------
def procesar_symbol(df_sub, symbol ):
    try:
        
        df_sub = df_sub.sort_values(by="Fecha")
        fg = Feature_Generator(df_sub, date_col="Fecha")

        # EMA con periodos comunes
        ema_periods = [5, 7, 10, 12, 14]
        
        # RSI con periodos comunes
        rsi_periods = [8, 10, 12, 14]

        # MACD con los conjuntos de parámetros (Fast, Slow, Signal)
        macd_periods = [[12, 26, 9], [7, 18, 9], [10, 20, 7]]
        
        # Stochastic Oscillator con los conjuntos de parámetros (K, D, Smooth)
        stochastic_periods = [[14, 3, 3]]  
 
        # Williams %R con periodos comunes
        williams_periods = [10, 14, 17]

        # CCI con periodos estándar
        cci_periods = [9, 12, 14, 15, 20]

        # Rate of change con periodos estándar
        roc_periods = [1, 2, 3, 5, 7, 10, 12, 14]
        
        # Adx con periodos comunes
        adx_periods = [8, 9, 10, 12, 14]  

        # ATR con periodos comunes
        atr_periods = [8, 10, 12, 14] 

        # Bollinger Bands con periodos y desviaciones estándar
        bb_periods = [
                        [10, 1.5],   # Más corto plazo, bandas más ajustadas
                        [10, 2],     # Corto plazo, bandas "normales"
                        [20, 1],     # Clásico pero menos sensibilidad a volatilidad
                        [20, 2],     # La clásica
                        [20, 2.5],   # Más exigente para señales extremas
                        [30, 2],     # Medio plazo, señales más "limpias"
                        [50, 2],     # Largo plazo, menos ruido
                        [20, 3],     # Bandas ultra anchas para tendencias fuertes
                    ]

        # Añadimos Rate of Change sobre columnas específicas           
        # On-Balance Volume con periodos de EMA
        obv_ema_periods = [5, 10]

        # Connors RSI con parámetros específicos
        # Ejemplo: [rsi_periods, up_down_periods, lookback_periods]
        connors_params = [[3, 2, 25]] 
        
        # Ultimate Oscillator con periodos típicos
        uo_periods = [[7, 14, 28]]
        
        # Añadimos MFI (Money Flow Index) con periodos comunes
        mfi_periods = [7, 10, 14]  # puedes modificar los periodos

        # Añadimos ADL (Accumulation/Distribution Line)
        # ADL no requiere periodos, se calcula directamente sobre los datos de volumen y precio 
        adl_ema_periods = [10, 12, 14, 15, 17,  20]
        
        # Añadimos Chaikin Oscillator con periodos comunes
        # Chaikin Oscillator se calcula con dos periodos, típicamente 3 y 10, o 5 y 20        
        cho_periods = [[3, 10],  [5, 20]]  # Los periodos que quieras probar

        # Añadimos TEMA (Triple Exponential Moving Average)
        # TEMA no es tan común, pero puedes usarlo con periodos típicos 
        tema_periods = [7, 10, 12, 14]
        
        
        # Añadimos Parabolic SAR con parámetros típicos
        # Parabolic SAR no tiene periodos como tal, pero sí parámetros de aceleración y
        # máximo, que puedes ajustar según tu estrategia
        # Aquí puedes usar un par de valores comunes, como 0.02 y 0.2 para el acelerador y 0.2 para el máximo
        # Puedes añadir más combinaciones si lo deseas
        sar_params = [[0.02, 0.2]]  
        
        # Añadimos Ulcer Index con distintos periodos
        # El Ulcer Index mide la profundidad y duración de las caídas del precio    
        ui_periods = [5, 9, 14] 


        # Añadimos todas las características al generador
        fg.add_ema(ema_periods)
        fg.add_rsi(rsi_periods)
        fg.add_macd(macd_periods)
        fg.add_stochastic(stochastic_periods)
        fg.add_williams_r(williams_periods)
        fg.add_cci(cci_periods)
        fg.add_roc(roc_periods)
        fg.add_roc_on_columns(periods=roc_periods, columns=["Open", "High", "Low"])
        fg.add_adx(adx_periods)
        fg.add_atr(atr_periods)
        fg.add_bollinger_bands(bb_periods)
        fg.add_obv(obv_ema_periods)
        fg.add_connors_rsi(connors_params)
        fg.add_ultimate_oscillator(uo_periods)
        fg.add_mfi(mfi_periods)
        fg.add_adl(adl_ema_periods) 
        fg.add_chaikin_osc(cho_periods)
        fg.add_tema(tema_periods)
        fg.add_parabolic_sar(sar_params)
        fg.add_ulcer_index(ui_periods)      


        # Añadimos las características binarias
        # Aquí puedes ajustar los periodos y parámetros según tu estrategia
        fg.add_binary_features(
            ema_periods=ema_periods,
            rsi_periods=rsi_periods,
            macd_periods=macd_periods,
            stochastic_periods=stochastic_periods,
            williams_periods=williams_periods,
            cci_periods=cci_periods,
            roc_periods=roc_periods,
            adx_periods=adx_periods,
            atr_periods=atr_periods,
            bb_periods=bb_periods,
            obv_ema_periods=obv_ema_periods,
            connors_params=connors_params,
            uo_periods=uo_periods,
            mfi_periods=mfi_periods,
            adl_ema_periods=adl_ema_periods,
            cho_periods=cho_periods,
            tema_periods=tema_periods,
            sar_params=sar_params,
            ui_periods=ui_periods)




        # Añadirmos características de tendencia
        
        
        # Añadimos targets con ventana y horizonte
        fg.add_trend_angle(column='Close', window=10, horizon=1)
        fg.add_trend_angle(column='Close', window=10, horizon=2)
        fg.add_trend_angle(column='Close', window=10, horizon=3)
        fg.add_trend_angle(column='Close', window=10, horizon=4)
        fg.add_trend_angle(column='Close', window=10, horizon=5)
        fg.add_trend_angle(column='Close', window=10, horizon=6)
        fg.add_trend_angle(column='Close', window=10, horizon=7)
        fg.add_trend_angle(column='Close', window=10, horizon=8)
        fg.add_trend_angle(column='Close', window=10, horizon=9)
        fg.add_trend_angle(column='Close', window=10, horizon=10)
        fg.add_trend_angle(column='Close', window=10, horizon=12)
        fg.add_trend_angle(column='Close', window=10, horizon=15)


        # Añadimos targets con ventana y horizonte
        fg.add_trend_angle(column='Close', window=15, horizon=1)
        fg.add_trend_angle(column='Close', window=15, horizon=2)
        fg.add_trend_angle(column='Close', window=15, horizon=3)
        fg.add_trend_angle(column='Close', window=15, horizon=4)
        fg.add_trend_angle(column='Close', window=15, horizon=5)
        fg.add_trend_angle(column='Close', window=15, horizon=6)
        fg.add_trend_angle(column='Close', window=15, horizon=7)
        fg.add_trend_angle(column='Close', window=15, horizon=8)
        fg.add_trend_angle(column='Close', window=15, horizon=9)
        fg.add_trend_angle(column='Close', window=15, horizon=10)
        fg.add_trend_angle(column='Close', window=15, horizon=12)
        fg.add_trend_angle(column='Close', window=15, horizon=15)


        
        # Añadimos targets con ventana y horizonte
        fg.add_trend_angle(column='Close', window=5, horizon=1)
        fg.add_trend_angle(column='Close', window=5, horizon=2)
        fg.add_trend_angle(column='Close', window=5, horizon=3)
        fg.add_trend_angle(column='Close', window=5, horizon=4)
        fg.add_trend_angle(column='Close', window=5, horizon=5)
        fg.add_trend_angle(column='Close', window=5, horizon=8)
        fg.add_trend_angle(column='Close', window=5, horizon=10)
        fg.add_trend_angle(column='Close', window=5, horizon=12)
        fg.add_trend_angle(column='Close', window=5, horizon=15)
        




        df_feat = fg.get_df()
        return df_feat
    
    except Exception as e:
        print(f" Error procesando {symbol}: {e}")
        traceback.print_exc()
        return None


# ---------------------------
# MAIN Puno de entrada al programa
# ---------------------------
if __name__ == "__main__":
    
    # El CSV con los datos de movimientos diarios de precios históricos de 50  stocks
    data_path = "DATA/AllStocksHistoricalData.csv"

    # Cargamos el archivo completo
    df_all = pd.read_csv(data_path, sep=';', parse_dates=['Fecha'])
    

    # Definimos los  filtros del tamaño de dataset
    fecha_minima = pd.Timestamp("2010-01-01")
    
    
    symbols_seleccionados = pd.read_csv("DATA/stocks_seleccionados.csv")['Symbol'].unique().tolist()

    # Aplicar filtro combinado (fecha + símbolos)
    df_all = df_all[
        (df_all['Fecha'] >= fecha_minima) &
        (df_all['Symbol'].isin(symbols_seleccionados))
    ]


    # Agrupar manualmente por símbolo, creando lista de (symbol, df)
    lista_dataframes = []
    for symbol in df_all['Symbol'].unique():
        df_symbol = df_all[df_all['Symbol'] == symbol].copy()
        lista_dataframes.append((symbol, df_symbol))

    print(f" Detectados {len(lista_dataframes)} símbolos. Iniciando procesamiento paralelo...")


    # Calcular número óptimo de workers
    num_nucleos = os.cpu_count() or 4  # fallback en caso de que devuelva None
    max_workers = max(2, num_nucleos - 2)
    print(f"⚙️ Ejecutando con {max_workers} procesos paralelos (de {num_nucleos} núcleos lógicos disponibles)")

    
    t_inicio = time.time()
    print(f"⏱️ Inicio del procesamiento: {time.strftime('%H:%M:%S')}")

    # Procesamiento paralelo
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        resultados = list(executor.map(procesar_tupla, lista_dataframes))

    print(" Procesamiento paralelo completado.")
    
    t_fin = time.time()
    print(f"⏱️ Fin del procesamiento: {time.strftime('%H:%M:%S')}")
    print(f"⏱️ Tiempo total: {t_fin - t_inicio:.2f} segundos")
    
    # Concatenamos resultados geneados por los hilos 
    df_final = pd.concat([df for df in resultados if df is not None], ignore_index=True)

    # Eliminar las primeras N_SKIP filas de cada símbolo
    N_SKIP = 60  # Número de filas a omitir al inicio por periodos iniciales de indicadores
    df_final = (
        df_final.groupby('Symbol', group_keys=False)
                .apply(lambda g: g.iloc[N_SKIP:])
                .reset_index(drop=True)
    )

    
    # Elimina filas con NaN en cualquier columna TARGET_.
    # Esto limpia los nulos generados al final de cada símbolo por falta de datos futuros en los targets.
    df_final = df_final.dropna(subset=[c for c in df_final.columns if c.startswith('TARGET_')])

    # Guardar resultado final
    output_path = Path(__file__).parent.parent  / "DATA" / "Dataset_All_Features.csv"
    df_final.to_csv(output_path, index=False, sep=';')


    print(f" Dataset final guardado en: {output_path}")


