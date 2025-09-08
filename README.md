# TFM – Predicción de tendencias en series financieras (ML + GA + Backtesting)

Sistema end-to-end para generar señales de inversión a partir de indicadores técnicos, con **selección evolutiva de características (GA)** y comparación de modelos (XGBoost, MLP, ARIMA). El foco está en **utilidad práctica**: backtesting, métricas de rentabilidad–riesgo y reproducibilidad.

## 🎯 Objetivo
Construir y validar un pipeline completo: **descarga y ETL de datos → feature engineering masivo → selección de features con GA → tuning → backtesting → análisis y resultados**.

---

## 📁 Estructura del repositorio
```
.
├─ BuildDataSet/
│  ├─ Download_Historical_Data.py
│  ├─ Build_Dataset_All_Features.py
│  ├─ Comprobar_Coherencia_Dataset.py
│  └─ Build_Dataset_Auxiliar_Backtesting.py
│
├─ BackTesting/
│  ├─ Backtester.py
│  ├─ BacktesterSignalExit.py
│  └─ BacktesterCalidadML.py
│
├─ GA/
│  └─ GA_Feature_Selection.py
│
├─ Tests_Prediccion/
│  ├─ Test_Prediccion_Metricas_KFoldCV.py
│  └─ Test_Prediccion_Target_Vs_Predicted.py
│
├─ Analisis/
│  ├─ Analisis_Metricas_Trading_Target.py
│  ├─ Analisis_Metricas_Trading_Arquitecturas_MLP.py
│  ├─ Analisis_Impacto_Tuning.py
│  ├─ Analisis_Directional_Acccuracy_AllStocks.py
│  ├─ Analisis_Feature_Engineering_GA.py
│  ├─ Analisis_Features_GA_vs_Aleatorias.py
│  ├─ Analisis_GA_Features_MasSeleccionas.py
│  ├─ Analisis_GA_Seleccion_Modelo.py
│  └─ Analisis_Variacion_Precios.py
│
├─ (raíz)
│  ├─ Feature_Generator.py
│  ├─ Adaptive_Feature_Normalizer.py
│  ├─ Pipeline_GA_Seleccion_Features_AllStocks.py
│  ├─ Pipeline_Ajuste_Hiperparametros.py
│  ├─ Pipeline_BackTesting_GAXGBoost.py
│  ├─ Pipeline_BackTesting_GAXGBoost_Calidad.py
│  ├─ Pipeline_BackTesting_XGB_AllFeatures.py
│  ├─ Pipeline_BackTesting_MLP_OHLC.py
│  ├─ Pipeline_BackTesting_MLP_OHLC_Original.py
│  └─ Pipeline_BackTesting_ARIMA.py
│
├─ DATA/              # datasets (entrada/salida)
├─ Resultados/        # tablas, figuras, logs
└─ Tests_Prediccion/  # predicciones por fecha/activo (si aplica)
```
> *Nota:* El pipeline LSTM no se incluye en este README por decisión del autor.

---

## 🧩 Ficheros clave (rápido)

**BuildDataSet/**
- `Download_Historical_Data.py` – Descarga OHLCV (yfinance) + ETL (imputación, redondeos, control de % nulos). **Salida:** `Data/AllStocksHistoricalData.csv` *(ojo a la carpeta: `Data/` con mayúscula/minúscula)*.
- `Build_Dataset_All_Features.py` – Construye el **dataset final** por símbolo con indicadores. **Salida:** `DATA/Dataset_All_Features.csv`.
- `Adaptive_Feature_Normalizer.py` – **Normalización adaptativa** según *skewness* (evita *leakage*). **Salida:** `DATA/Dataset_All_Features_Transformado.csv`.
- `Build_Dataset_Auxiliar_Backtesting.py` – Auxiliares para backtesting (ATR simple). **Salida:** `DATA/AllStocksHistoricalData_Auxiliar_Backtesting.csv`.
- `Comprobar_Coherencia_Dataset.py` – Chequeos de calidad (nulos, duplicados, rangos).

**BackTesting/**
- `Backtester.py` – **Motor base** de backtesting (señales, posiciones, métricas, logs). **Salida:** `Resultados/Backtesting_TradeLog_{symbol}.csv`.
- `BacktesterSignalExit.py` – Backtesting con **salida por señal**. **Salida:** `Resultados/Backtesting_TradeLog_{symbol}_SignalExit.csv`.
- `BacktesterCalidadML.py` – Backtesting de **calidad de señal** (umbrales, Spearman robusto).

**GA/**
- `GA_Feature_Selection.py` – **Algoritmo Genético** para seleccionar subconjuntos de *features* (torneo, crossover, mutación), compatible con ElasticNet/SVM/MLP/XGBoost.

**Tests_Prediccion/**
- `Test_Prediccion_Metricas_KFoldCV.py` – **K-Fold por años** para evaluar modelos. **Salida:** `Resultados/Predicciones_KFoldsCV_{model_name}.csv`.
- `Test_Prediccion_Target_Vs_Predicted.py` – Comparación **target vs. predicho** (sesgos/lag). **Salida:** `Resultados/Predicciones_Test_{model_name}_{symbol}.csv`.

**Analisis/**
- `Analisis_Metricas_Trading_Target.py` – Calcula métricas agregadas por *target* y cartera. **Salida:** `Resultados/Resumen_Metricas_Trading.csv`.
- `Analisis_Metricas_Trading_Arquitecturas_MLP.py` – Resumen por arquitecturas MLP. **Salida:** `Resultados/Resumen_Metricas_Trading_Arquitecturas.csv`.
- `Analisis_Impacto_Tuning.py` – Impacto del *tuning*. **Salidas:** `Resultados/Comparativa_Tuning_2020_2024.csv`, `Resultados/Grafico_Mejora_Tuning.png`.
- `Analisis_GA_Features_MasSeleccionas.py` – Frecuencia de *features* seleccionadas por GA. **Salida:** `Resultados/GA_Frecuencia_Features.csv`.
- `Analisis_Features_GA_vs_Aleatorias.py` – GA vs selección aleatoria. **Salida:** `Resultados/Comparativa_GA_vs_Aleatorias.csv`.
- `Analisis_GA_Seleccion_Modelo.py` – Selección GA y evaluación multi–modelo (K-Fold temporal). **Salida:** `Resultados/Comparativa_GA_Multimodelo_KFold.csv`.
- `Analisis_Directional_Acccuracy_AllStocks.py` – Accuracy direccional diario (close-only, umbral u). **Salida:** `Resultados/DA_Binary_Daily_AllStocks_SIMPLE.csv`.
- `Analisis_Feature_Engineering_GA.py` – Estudia el impacto del GA (exploratorio, sin guardados por defecto).
- `Analisis_Variacion_Precios.py` – Variaciones y resumen estadístico (exploratorio, sin guardados por defecto).

---

## ▶️ Flujo recomendado
1) **Datos y dataset**  
   `BuildDataSet/Download_Historical_Data.py` → `BuildDataSet/Build_Dataset_All_Features.py` →  
   `Adaptive_Feature_Normalizer.py` → `BuildDataSet/Build_Dataset_Auxiliar_Backtesting.py` → `BuildDataSet/Comprobar_Coherencia_Dataset.py`  
   **Salidas esperadas:** `Data/AllStocksHistoricalData.csv`, `DATA/Dataset_All_Features.csv`, `DATA/Dataset_All_Features_Transformado.csv`, `DATA/AllStocksHistoricalData_Auxiliar_Backtesting.csv`.

2) **Selección de features (GA)**  
   `Pipeline_GA_Seleccion_Features_AllStocks.py`  
   **Salidas esperadas:** `Resultados/GA/…` *(según configuración; features seleccionadas y logs).*

3) **Ajuste de hiperparámetros**  
   `Pipeline_Ajuste_Hiperparametros.py`  
   **Salidas esperadas:** `Resultados/Hiperparametros_{TARGET}.csv`.

4) **Backtesting (principal: GA + XGBoost)**  
   `Pipeline_BackTesting_GAXGBoost.py`  
   **Salidas esperadas:** `Resultados/Backtesting_TradeLog_{symbol}.csv` *(y métricas que consolida la fase de análisis).*

5) **Análisis y tablas finales**  
   Ejecutar `Analisis/Analisis_Metricas_Trading_Target.py` y otros `Analisis_*` según necesidad (ver tabla anterior).

> Si cambias **W/H** o la cartera de símbolos, repite (2) y (3) antes del (4).

---

## ▶️ Ejecución rápida (solo `.py`)

```bash
# 1) Datos + dataset + normalización + auxiliares + control de calidad
python BuildDataSet/Download_Historical_Data.py
python BuildDataSet/Build_Dataset_All_Features.py
python Adaptive_Feature_Normalizer.py
python BuildDataSet/Build_Dataset_Auxiliar_Backtesting.py
python BuildDataSet/Comprobar_Coherencia_Dataset.py

# 2) Selección de features con GA (por símbolos / cartera)
python Pipeline_GA_Seleccion_Features_AllStocks.py

# 3) Ajuste de hiperparámetros (en base a las features seleccionadas)
python Pipeline_Ajuste_Hiperparametros.py

# 4) Backtesting principal (XGBoost + GA)
python Pipeline_BackTesting_GAXGBoost.py

# 5) Análisis de métricas por target / cartera
python Analisis/Analisis_Metricas_Trading_Target.py
```

---

## 📦 Requisitos (mínimos sugeridos)
`pandas`, `numpy`, `scikit-learn`, `scipy`, `xgboost`, `deap`, `matplotlib`, `tqdm`, `yfinance`, `pandas-ta` *(o `ta`)*, `statsmodels`, `python-dateutil`.  
*(Si usas deep learning: `torch`/`torchvision`/`torchaudio` o `tensorflow`/`keras`.)*

**Python:** 3.12.x (CPython). Para auditar versiones reales del entorno, ver `audit_versions_strict.py` o `check_env_strict.py` (opcional).
