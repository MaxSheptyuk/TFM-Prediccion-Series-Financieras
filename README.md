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

> *Nota:* Por elección del autor, el pipeline LSTM no se incluye en este README.

---

## 🧩 Ficheros clave (rápido)

**BuildDataSet/**
- `Download_Historical_Data.py` – Descarga OHLCV (yfinance) + ETL (imputación, redondeos, control de % nulos).
- `Build_Dataset_All_Features.py` – Construye el **dataset final** por símbolo con indicadores.
- `Comprobar_Coherencia_Dataset.py` – Chequeos de calidad (nulos, duplicados, rangos).
- `Build_Dataset_Auxiliar_Backtesting.py` – Utilidades para backtesting (p. ej., ATR simple).

**BackTesting/**
- `Backtester.py` – **Motor base** de backtesting (señales, posiciones, métricas, logs).
- `BacktesterSignalExit.py` – Backtesting con **salida gobernada por señal**.
- `BacktesterCalidadML.py` – Backtesting de **calidad de señal** (umbrales, Spearman robusto).

**GA**
- `GA_Feature_Selection.py` – **Algoritmo Genético** para seleccionar subconjuntos de *features* (torneo, crossover, mutación), compatible con ElasticNet/SVM/MLP/XGBoost.

**Tests_Prediccion/**
- `Test_Prediccion_Metricas_KFoldCV.py` – **K-Fold por años** para evaluar modelos.
- `Test_Prediccion_Target_Vs_Predicted.py` – Comparación **target vs. predicho** (sesgos/lag).

**Analisis/**
- `Analisis_Metricas_Trading_Target.py` – Métricas por (W, H): ROI, CAGR, MaxDD, payoff, expectancy, % ganadoras.
- `Analisis_Impacto_Tuning.py` – Impacto del *tuning* por stock vs baseline.
- (Resto `Analisis_*` aportan comparativas adicionales: MLP, GA vs aleatorias, frecuencia de features, DA, volatilidad, etc.)

**Raíz (utilidades y pipelines)**
- `Feature_Generator.py` – Indicadores técnicos masivos (EMA, RSI, **MACD con naming por salida**, Stochastic, Williams %R, CCI, ADX/DI±, ATR, Bollinger, OBV, …).
- `Adaptive_Feature_Normalizer.py` – **Normalización adaptativa** según *skewness* (evita *leakage*).
- `Pipeline_GA_Seleccion_Features_AllStocks.py` – Selección **GA** de features por símbolo/cartera.
- `Pipeline_Ajuste_Hiperparametros.py` – *Tuning* con validación temporal.
- `Pipeline_BackTesting_GAXGBoost.py` – **Principal**: XGBoost + GA (entrenamiento + backtesting).
- `Pipeline_BackTesting_GAXGBoost_Calidad.py` – Variante centrada en **calidad de señal**.
- `Pipeline_BackTesting_XGB_AllFeatures.py` – **Baseline** XGBoost con todas las *features* (sin GA).
- `Pipeline_BackTesting_MLP_OHLC.py` / `…_Original.py` – Pipelines **MLP** (scikit-learn).
- `Pipeline_BackTesting_ARIMA.py` – **Benchmark ARIMA**.

---

## ▶️ Flujo recomendado
1) **Datos y dataset**  
   `BuildDataSet/Download_Historical_Data.py` → `BuildDataSet/Build_Dataset_All_Features.py` →  
   `Adaptive_Feature_Normalizer.py` → `BuildDataSet/Build_Dataset_Auxiliar_Backtesting.py` → `BuildDataSet/Comprobar_Coherencia_Dataset.py`  
   **Salidas esperadas:** `DATA/` (históricos, dataset con features y versión normalizada), `Resultados/Checks/` (informes de coherencia).

2) **Selección de features (GA)**  
   `Pipeline_GA_Seleccion_Features_AllStocks.py`  
   **Salidas esperadas:** `Resultados/GA/` (p. ej., `selected_features_*.csv|json` + logs).

3) **Ajuste de hiperparámetros**  
   `Pipeline_Ajuste_Hiperparametros.py`  
   **Salidas esperadas:** `Resultados/Tuning/` (p. ej., `best_params_*.csv|json` + métricas CV).

4) **Backtesting (principal: GA + XGBoost)**  
   `Pipeline_BackTesting_GAXGBoost.py`  
   **Salidas esperadas:** `Resultados/Backtesting/` (curvas de equity `*.png`, `trades_log_*.csv`, `Resumen_Metricas_Trading.csv`).

5) **Análisis y tablas finales**  
   `Analisis/Analisis_Metricas_Trading_Target.py` (y otros `Analisis_*`)  
   **Salidas esperadas:** `Resultados/Analisis/` (tablas comparativas y figuras).

> Si cambias **W/H** o la cartera de símbolos, repite (2) y (3) antes del (4).

---

## 🖼️ Figuras y tablas: **qué genera cada script**
> Dónde mirar para **gráficos (figuras)** y **tablas (CSV)** de tu memoria.

- **Curvas de equity por target y comparativas de ROI/DD**  
  **Script:** `Analisis/Analisis_Metricas_Trading_Target.py`  
  **Figuras:** `Resultados/Analisis/Equity_*.png`, `Resultados/Analisis/ROI_*.png` *(nombres orientativos)*  
  **Tablas:** `Resultados/Analisis/Resumen_Metricas_Trading.csv` y, si está habilitado, `Resumen_Cartera_Anual.csv` (ROI anual, DD, payoff, expectancy, % ganadoras).

- **Barras/tabla de rendimiento de cartera anual**  
  **Script:** `Analisis/Analisis_Metricas_Trading_Target.py`  
  **Tablas:** `Resultados/Analisis/Resumen_Cartera_Anual.csv` *(siempre que el bloque anual esté activo en el script)*  
  **Figuras:** `Resultados/Analisis/ROI_Anual_Barras_*.png` *(si la sección de plotting está activa)*

- **Comparativa de arquitecturas MLP**  
  **Script:** `Analisis/Analisis_Metricas_Trading_Arquitecturas_MLP.py`  
  **Figuras:** `Resultados/Analisis/MLP_*.png`  
  **Tablas:** `Resultados/Analisis/Resumen_Metricas_Trading_Arquitecturas.csv`

- **Impacto del tuning**  
  **Script:** `Analisis/Analisis_Impacto_Tuning.py`  
  **Figuras:** `Resultados/Analisis/Grafico_Mejora_Tuning.png`  
  **Tablas:** `Resultados/Analisis/Comparativa_Tuning_*.csv`

- **Frecuencia de features seleccionadas por GA**  
  **Script:** `Analisis/Analisis_GA_Features_MasSeleccionas.py`  
  **Figuras:** `Resultados/Analisis/GA_Frecuencia_Features.png`  
  **Tablas:** `Resultados/Analisis/GA_Frecuencia_Features.csv`

- **GA vs selección aleatoria**  
  **Script:** `Analisis/Analisis_Features_GA_vs_Aleatorias.py`  
  **Figuras:** `Resultados/Analisis/GA_vs_Aleatorias_*.png`  
  **Tablas:** `Resultados/Analisis/Comparativa_GA_vs_Aleatorias.csv`

- **Accuracy direccional (DA) por símbolo**  
  **Script:** `Analisis/Analisis_Directional_Acccuracy_AllStocks.py`  
  **Figuras:** `Resultados/Analisis/DA_*.png`  
  **Tablas:** `Resultados/Analisis/DA_Binary_Daily_AllStocks_*.csv`

- **Variación de precios / volatilidad**  
  **Script:** `Analisis/Analisis_Variacion_Precios.py`  
  **Figuras:** `Resultados/Analisis/Variacion_*.png`  
  **Tablas:** `Resultados/Analisis/Variacion_*.csv`

> **Assets/**: si tu memoria usa una carpeta `Assets/`, basta con **copiar** desde `Resultados/Analisis/*.png` a `Assets/` (sólo las figuras definitivas).

---

## ▶️ Ejecución rápida (solo `.py`)

> Edita parámetros en la cabecera de cada script (símbolos, fechas, W/H, costes, seeds).

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

**Python:** 3.12.x (CPython). Para auditar versiones reales del entorno, ver `audit_versions_strict.py` o `versions_auditoria.py` (opcional).
