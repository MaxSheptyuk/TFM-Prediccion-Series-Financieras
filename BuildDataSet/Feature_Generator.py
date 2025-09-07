import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import LinearRegression
from stock_indicators import indicators
from stock_indicators import Quote
import traceback
import warnings

# ---------------------------
# Clase FeatureGenerator 
# Genera datos de características para análisis financiero
# Genera señales de indicadores técnicos como EMA, RSI , etc
# ---------------------------

class Feature_Generator:
    def __init__(self, df, date_col='Fecha'):
        self.df = df.copy()
        self.date_col = date_col
        self._quotes = self._build_quotes()

    def _build_quotes(self):
        fechas = np.array(pd.to_datetime(self.df[self.date_col]).dt.to_pydatetime())
        opens = self.df['Open'].values
        highs = self.df['High'].values
        lows = self.df['Low'].values
        closes = self.df['Close'].values
        volumes = self.df['Volume'].values

        return [
            Quote(
                date=fechas[i],
                open=opens[i],
                high=highs[i],
                low=lows[i],
                close=closes[i],
                volume=volumes[i]
                )
                for i in range(len(closes))
            ]


    def add_ema(self, periods_ema=[10, 20, 50]):
        new_cols = {}
        for p_ema in periods_ema:
            ema_result = indicators.get_ema(self._quotes, p_ema)
            ema = np.array([x.ema if x.ema is not None else np.nan for x in ema_result])
            new_cols[f'EMA_{p_ema}'] = ema
        if new_cols:
            self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)
        return self

    
    def add_rsi(self, periods_rsi=[14]):
        new_cols = {}
        for p_rsi in periods_rsi:
            rsi_result = indicators.get_rsi(self._quotes, p_rsi)
            rsi = np.array([x.rsi if x.rsi is not None else np.nan for x in rsi_result])
            new_cols[f'RSI_{p_rsi}'] = rsi
        if new_cols:
            self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)
        return self


    def add_macd(self, fast_slow_signals=[[12, 26, 9]]):
        new_cols = {}
        for fast, slow, signal in fast_slow_signals:
            macd_result = indicators.get_macd(self._quotes, fast, slow, signal)
            new_cols[f'MACD_{fast}_{slow}_{signal}'] = [x.macd if x.macd is not None else np.nan for x in macd_result]
            new_cols[f'MACD_SIGNAL_{fast}_{slow}_{signal}'] = [x.signal if x.signal is not None else np.nan for x in macd_result]
            new_cols[f'MACD_HISTOGRAM_{fast}_{slow}_{signal}'] = [x.histogram if x.histogram is not None else np.nan for x in macd_result]
        if new_cols:
            self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)
        return self
    


    def add_stochastic(self, stochastic_periods=[[14, 3, 3]]):
        """
        Añade columnas de %K, %D y Signal para cada configuración de periodos.
        Formato: [periodo, smooth_k, smooth_d]
        """
        new_cols = {}
        for p in stochastic_periods:
            period, smooth_k, smooth_d = p
            stoch_result = indicators.get_stoch(self._quotes, period, smooth_k, smooth_d)

            k_values = np.array([x.k if x.k is not None else np.nan for x in stoch_result])
            d_values = np.array([x.d if x.d is not None else np.nan for x in stoch_result])
            sig_values = np.array([x.signal if x.signal is not None else np.nan for x in stoch_result])

            new_cols[f'STOCH_K_{period}_{smooth_k}_{smooth_d}'] = k_values
            new_cols[f'STOCH_D_{period}_{smooth_k}_{smooth_d}'] = d_values
            new_cols[f'STOCH_SIGNAL_{period}_{smooth_k}_{smooth_d}'] = sig_values

        if new_cols:
            self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)
        return self
    

    def add_williams_r(self, williams_periods=[14]):
        new_cols = {}
        for p in williams_periods:
            try:
                willr_result = indicators.get_williams_r(self._quotes, p)
                willr = np.array([x.williams_r if x.williams_r is not None else np.nan for x in willr_result])
                new_cols[f'WILLIAMS_R_{p}'] = willr
            except Exception as e:
                print(f" Error calculando Williams %R ({p}): {e}")
        if new_cols:
            self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)
        return self


    def add_cci(self, cci_periods=[14, 20]):
        new_cols = {}
        for p_cci in cci_periods:
            try:
                cci_result = indicators.get_cci(self._quotes, p_cci)
                cci = np.array([x.cci if x.cci is not None else np.nan for x in cci_result])
                new_cols[f'CCI_{p_cci}'] = cci
            except Exception as e:
                print(f" Error calculando CCI ({p_cci}): {e}")
        if new_cols:
            self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)
        return self

    
    def add_roc(self, roc_periods=[5, 10, 14]):
        """
        Añade columnas ROC_N (Rate of Change porcentual) para cada periodo dado.
        ROC_N = (Close - Close_N) / Close_N * 100
        """
        new_cols = {}
        for p in roc_periods:
            prev_close = self.df['Close'].shift(p)
            roc = ((self.df['Close'] - prev_close) / prev_close) * 100
            new_cols[f'ROC_{p}'] = roc
        if new_cols:
            self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)
        return self
    
    
    def add_roc_on_columns(self, columns=["Open", "High", "Low"], periods=[5, 21]):
        new_cols = {}
        for col_base in columns:
            for p in periods:
                col = f"ROC_COL_{col_base}_{p}"
                new_cols[col] = self.df[col_base].pct_change(periods=p) * 100
        if new_cols:
            self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)
        return self
                

    def add_adx(self, adx_periods=[14]):
        """
        Añade columnas de ADX, PDI (DI+), MDI (DI-) para cada periodo.
        """
        new_cols = {}
        for p in adx_periods:
            try:
                adx_result = indicators.get_adx(self._quotes, p)
                adx = np.array([x.adx if x.adx is not None else np.nan for x in adx_result])
                pdi = np.array([x.pdi if x.pdi is not None else np.nan for x in adx_result])
                mdi = np.array([x.mdi if x.mdi is not None else np.nan for x in adx_result])
                new_cols[f'ADX_{p}'] = adx
                new_cols[f'PDI_{p}'] = pdi
                new_cols[f'MDI_{p}'] = mdi
            except Exception as e:
                print(f" Error calculando ADX ({p}): {e}")
        if new_cols:
            self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)
        return self

    def add_atr(self, atr_periods=[14]):
        """
        Añade columnas de ATR para cada periodo especificado.
        """
        new_cols = {}
        for p in atr_periods:
            try:
                atr_result = indicators.get_atr(self._quotes, p)
                atr = np.array([x.atr if x.atr is not None else np.nan for x in atr_result])
                new_cols[f'ATR_{p}'] = atr
            except Exception as e:
                print(f" Error calculando ATR ({p}): {e}")
        if new_cols:
            self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)
        return self



    def add_bollinger_bands(self, periods_stddevs=[[20, 2]]):
        """
        Añade columnas de Bollinger Bands para cada configuración de periodos y desviaciones estándar.
        Formato: [[periodo, std_dev], ...]
        """
        new_cols = {}
        for period, std in periods_stddevs:
            try:
                bb_result = indicators.get_bollinger_bands(self._quotes, period, std)
                # Extrae cada valor
                sma = np.array([x.sma if x.sma is not None else np.nan for x in bb_result])
                upper = np.array([x.upper_band if x.upper_band is not None else np.nan for x in bb_result])
                lower = np.array([x.lower_band if x.lower_band is not None else np.nan for x in bb_result])
                percent_b = np.array([x.percent_b if x.percent_b is not None else np.nan for x in bb_result])
                zscore = np.array([x.z_score if x.z_score is not None else np.nan for x in bb_result])
                width = np.array([x.width if x.width is not None else np.nan for x in bb_result])
                # Añade al diccionario
                new_cols[f'BB_SMA_{period}_{std}'] = sma
                new_cols[f'BB_UPPER_{period}_{std}'] = upper
                new_cols[f'BB_LOWER_{period}_{std}'] = lower
                new_cols[f'BB_PERCENT_B_{period}_{std}'] = percent_b
                new_cols[f'BB_ZSCORE_{period}_{std}'] = zscore
                new_cols[f'BB_WIDTH_{period}_{std}'] = width
            except Exception as e:
                print(f" Error calculando Bollinger Bands ({period}, {std}): {e}")
        if new_cols:
            self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)
        return self

    
    def add_obv(self, ema_periods=None):
        new_cols = {}
        # Calcula OBV puro
        obv_result = indicators.get_obv(self._quotes)
        obv = np.array([x.obv if x.obv is not None else np.nan for x in obv_result])
        new_cols['OBV'] = obv

        # Si se especifican periodos de EMA, calcula EMAs del OBV
        if ema_periods:
            for p in ema_periods:
                obv_ema = pd.Series(obv).ewm(span=p, adjust=False).mean().values
                new_cols[f'OBV_EMA_{p}'] = obv_ema

        if new_cols:
            self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)
        return self

    


    def add_connors_rsi(self, connors_params=[[3, 2, 100]]):
        """
        Añade columnas de Connors RSI y sus componentes para cada conjunto de parámetros.
        Formato de cada set: [rsi_periods, streak_periods, rank_periods]
        """
        new_cols = {}
        for params in connors_params:
            rsi_p, streak_p, rank_p = params
            try:
                connors_result = indicators.get_connors_rsi(self._quotes, rsi_p, streak_p, rank_p)
                new_cols[f'CONNORS_RSI_{rsi_p}_{streak_p}_{rank_p}'] = [
                    x.connors_rsi if x.connors_rsi is not None else np.nan for x in connors_result
                ]
                new_cols[f'CONNORS_RSI_CLOSE_{rsi_p}'] = [
                    x.rsi_close if x.rsi_close is not None else np.nan for x in connors_result
                ]
                new_cols[f'CONNORS_RSI_STREAK_{streak_p}'] = [
                    x.rsi_streak if x.rsi_streak is not None else np.nan for x in connors_result
                ]
                new_cols[f'CONNORS_PERCENT_RANK_{rank_p}'] = [
                    x.percent_rank if x.percent_rank is not None else np.nan for x in connors_result
                ]
            except Exception as e:
                print(f" Error calculando Connors RSI ({params}): {e}")
        if new_cols:
            self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)
        return self

    def add_ultimate_oscillator(self, periods_list=[[7, 14, 28]]):
        """
        Añade Ultimate Oscillator (UO) con distintas configuraciones de periodos.
        Cada configuración es una lista de 3 periodos [short, medium, long].
        """
        new_cols = {}
        for p in periods_list:
            try:
                short, medium, long = p
                uo_result = indicators.get_ultimate(self._quotes, short, medium, long)
                uo = np.array([x.ultimate if x.ultimate is not None else np.nan for x in uo_result])
                new_cols[f'UO_{short}_{medium}_{long}'] = uo
            except Exception as e:
                print(f" Error calculando Ultimate Oscillator ({p}): {e}")
        if new_cols:
            self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)
        return self
    

    def add_mfi(self, mfi_periods=[14]):
        new_cols = {}
        for p in mfi_periods:
            try:
                mfi_result = indicators.get_mfi(self._quotes, p)
                mfi = np.array([x.mfi if x.mfi is not None else np.nan for x in mfi_result])
                new_cols[f'MFI_{p}'] = mfi
            except Exception as e:
                print(f" Error calculando MFI ({p}): {e}")
        if new_cols:
            self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)
        return self
    

    def add_adl(self, ema_periods=[10, 20, 50]):
        """
        Añade la Accumulation/Distribution Line (ADL) y su EMA suavizada para cada periodo.
        """
        new_cols = {}
        try:
            adl_result = indicators.get_adl(self._quotes)
            adl = np.array([x.adl if x.adl is not None else np.nan for x in adl_result])
            new_cols['ADL'] = adl

            # EMA suavizada de ADL para cada periodo
            for p in ema_periods:
                new_cols[f'ADL_EMA_{p}'] = pd.Series(adl).ewm(span=p, adjust=False).mean().values

        except Exception as e:
            print(f" Error calculando ADL: {e}")
        if new_cols:
            self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)
        return self


    def add_chaikin_osc(self, cho_periods=[[3, 10]]):
        """
        Añade columnas de Chaikin Oscillator (CHO) para cada configuración de periodos.
        Formato: [fast, slow]
        """
        new_cols = {}
        for p in cho_periods:
            fast, slow = p
            try:
                cho_result = indicators.get_chaikin_osc(self._quotes, fast, slow)
                cho = np.array([x.oscillator if x.oscillator is not None else np.nan for x in cho_result])
                new_cols[f'CHO_{fast}_{slow}'] = cho
            except Exception as e:
                print(f" Error calculando Chaikin Oscillator ({fast}, {slow}): {e}")
        if new_cols:
            self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)
        return self


    def add_tema(self, tema_periods=[9, 21]):
        """
        Añade columnas de TEMA para cada periodo.
        """
        new_cols = {}
        for period in tema_periods:
            try:
                tema_result = indicators.get_tema(self._quotes, period)
                tema = np.array([x.tema if x.tema is not None else np.nan for x in tema_result])
                new_cols[f'TEMA_{period}'] = tema
            except Exception as e:
                print(f" Error calculando TEMA ({period}): {e}")
        if new_cols:
            self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)
        return self
    

    def add_parabolic_sar(self, sar_params=[[0.02, 0.2]]):
        """
        Añade columna de Parabolic SAR.
        sar_params: lista de pares [acceleration_step, max_acceleration_factor]
        """
        new_cols = {}
        for acc_step, max_acc in sar_params:
            try:
                sar_result = indicators.get_parabolic_sar(
                    self._quotes,
                    acceleration_step=acc_step,
                    max_acceleration_factor=max_acc
                )
                sar = [x.sar if x.sar is not None else np.nan for x in sar_result]
                new_cols[f'SAR_{acc_step}_{max_acc}'] = sar
            except Exception as e:
                print(f" Error calculando Parabolic SAR ({acc_step},{max_acc}): {e}")
        if new_cols:
            self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)
        return self



    def add_ulcer_index(self, ui_periods=[14]):
        """
        Añade columnas de Ulcer Index (UI) para los periodos indicados.
        """
        new_cols = {}
        for period in ui_periods:
            try:
                ui_result = indicators.get_ulcer_index(self._quotes, period)
                ui = [x.ui if x.ui is not None else np.nan for x in ui_result]
                new_cols[f'ULCER_INDEX_{period}'] = ui
            except Exception as e:
                print(f" Error calculando Ulcer Index ({period}): {e}")
        if new_cols:
            self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)
        return self




    # Método para obtener el DataFrame final con las características generadas
    def add_binary_features(
            self,
            ema_periods, rsi_periods, macd_periods, stochastic_periods,
            williams_periods, cci_periods, roc_periods, adx_periods, atr_periods,
            bb_periods, obv_ema_periods, connors_params, uo_periods, mfi_periods,
            adl_ema_periods, cho_periods, tema_periods, sar_params, ui_periods
        ):
        binary_cols = {}

        # EMA (Uptrend si EMA > Close)
        for p in ema_periods:
            binary_cols[f'BINARY_EMA_{p}_UPTREND'] = (self.df[f'EMA_{p}'] > self.df['Close']).astype(int)

        # RSI (Overbought si RSI > 70, Oversold si RSI < 30)
        for p in rsi_periods:
            binary_cols[f'BINARY_RSI_{p}_OVERBOUGHT'] = (self.df[f'RSI_{p}'] > 70).astype(int)
            binary_cols[f'BINARY_RSI_{p}_OVERSOLD'] = (self.df[f'RSI_{p}'] < 30).astype(int)

        # MACD
        for fast, slow, signal in macd_periods:
            binary_cols[f'BINARY_MACD_{fast}_{slow}_{signal}_UP'] = (self.df[f'MACD_{fast}_{slow}_{signal}'] > 0).astype(int)
            binary_cols[f'BINARY_MACD_SIGNAL_{fast}_{slow}_{signal}_UP'] = (self.df[f'MACD_SIGNAL_{fast}_{slow}_{signal}'] > 0).astype(int)
            binary_cols[f'BINARY_MACD_HISTOGRAM_{fast}_{slow}_{signal}_UP'] = (self.df[f'MACD_HISTOGRAM_{fast}_{slow}_{signal}'] > 0).astype(int)

        # Stochastic
        for period, smooth_k, smooth_d in stochastic_periods:
            k_col = f'STOCH_K_{period}_{smooth_k}_{smooth_d}'
            d_col = f'STOCH_D_{period}_{smooth_k}_{smooth_d}'
            binary_cols[f'BINARY_STOCH_{period}_{smooth_k}_{smooth_d}_K_ABOVE_D'] = (self.df[k_col] > self.df[d_col]).astype(int)

        # Williams %R
        for p in williams_periods:
            binary_cols[f'BINARY_WILLIAMS_R_{p}_OVERBOUGHT'] = (self.df[f'WILLIAMS_R_{p}'] > -20).astype(int)
            binary_cols[f'BINARY_WILLIAMS_R_{p}_OVERSOLD'] = (self.df[f'WILLIAMS_R_{p}'] < -80).astype(int)

        # CCI
        for p in cci_periods:
            binary_cols[f'BINARY_CCI_{p}_OVERBOUGHT'] = (self.df[f'CCI_{p}'] > 100).astype(int)
            binary_cols[f'BINARY_CCI_{p}_OVERSOLD'] = (self.df[f'CCI_{p}'] < -100).astype(int)

        # ROC
        for p in roc_periods:
            binary_cols[f'BINARY_ROC_{p}_UPTREND'] = (self.df[f'ROC_{p}'] > 0).astype(int)

        # ADX
        for p in adx_periods:
            binary_cols[f'BINARY_ADX_{p}_STRONG_TREND'] = (self.df[f'ADX_{p}'] > 20).astype(int)
            binary_cols[f'BINARY_ADX_{p}_UPTREND'] = (self.df[f'PDI_{p}'] > self.df[f'MDI_{p}']).astype(int)
            binary_cols[f'BINARY_ADX_{p}_DOWNTREND'] = (self.df[f'MDI_{p}'] > self.df[f'PDI_{p}']).astype(int)

        # ATR
        for p in atr_periods:
            col = f'ATR_{p}'
            binary_cols[f'BINARY_{col}_VOL_UP'] = (self.df[col] > self.df[col].shift(1)).astype(int)

        # Bollinger Bands
        for period, std in bb_periods:
            binary_cols[f'BINARY_BB_{period}_{std}_ABOVE'] = (self.df['Close'] > self.df[f'BB_UPPER_{period}_{std}']).astype(int)
            binary_cols[f'BINARY_BB_{period}_{std}_BELOW'] = (self.df['Close'] < self.df[f'BB_LOWER_{period}_{std}']).astype(int)

        # OBV
        binary_cols['BINARY_OBV_UP'] = (self.df['OBV'] > self.df['OBV'].shift(1)).astype(int)
        for p in obv_ema_periods:
            binary_cols[f'BINARY_OBV_EMA_{p}_UP'] = (self.df['OBV'] > self.df[f'OBV_EMA_{p}']).astype(int)
            binary_cols[f'BINARY_OBV_EMA_{p}_DOWN'] = (self.df['OBV'] < self.df[f'OBV_EMA_{p}']).astype(int)

        # Connors RSI
        for rsi, streak, rank in connors_params:
            binary_cols[f'BINARY_CONNORS_RSI_{rsi}_{streak}_{rank}_OVERBOUGHT'] = (self.df[f'CONNORS_RSI_{rsi}_{streak}_{rank}'] > 70).astype(int)
            binary_cols[f'BINARY_CONNORS_RSI_{rsi}_{streak}_{rank}_OVERSOLD'] = (self.df[f'CONNORS_RSI_{rsi}_{streak}_{rank}'] < 30).astype(int)

        # Ultimate Oscillator
        for p1, p2, p3 in uo_periods:
            binary_cols[f'BINARY_UO_{p1}_{p2}_{p3}_UP'] = (self.df[f'UO_{p1}_{p2}_{p3}'] > 50).astype(int)
            binary_cols[f'BINARY_UO_{p1}_{p2}_{p3}_DOWN'] = (self.df[f'UO_{p1}_{p2}_{p3}'] < 50).astype(int)

        # MFI
        for p in mfi_periods:
            binary_cols[f'BINARY_MFI_{p}_OVERBOUGHT'] = (self.df[f'MFI_{p}'] > 80).astype(int)
            binary_cols[f'BINARY_MFI_{p}_OVERSOLD'] = (self.df[f'MFI_{p}'] < 20).astype(int)

        # ADL
        binary_cols['BINARY_ADL_UP'] = (self.df['ADL'] > self.df['ADL'].shift(1)).astype(int)
        for p in adl_ema_periods:
            binary_cols[f'BINARY_ADL_EMA_{p}_UP'] = (self.df['ADL'] > self.df[f'ADL_EMA_{p}']).astype(int)
            binary_cols[f'BINARY_ADL_EMA_{p}_DOWN'] = (self.df['ADL'] < self.df[f'ADL_EMA_{p}']).astype(int)

        # Chaikin Oscillator
        for fast, slow in cho_periods:
            binary_cols[f'BINARY_CHO_{fast}_{slow}_UP'] = (self.df[f'CHO_{fast}_{slow}'] > 0).astype(int)

        # TEMA
        for p in tema_periods:
            binary_cols[f'BINARY_TEMA_{p}_UPTREND'] = (self.df[f'TEMA_{p}'] > self.df['Close']).astype(int)

        # Parabolic SAR
        for acc, max_acc in sar_params:
            col = f'SAR_{acc}_{max_acc}'
            binary_cols[f'BINARY_{col}_UP'] = (self.df['Close'] > self.df[col]).astype(int)
            binary_cols[f'BINARY_{col}_DOWN'] = (self.df['Close'] < self.df[col]).astype(int)

        # Ulcer Index
        for p in ui_periods:
            col = f'ULCER_INDEX_{p}'
            binary_cols[f'BINARY_{col}_UP'] = (self.df[col] > self.df[col].shift(1)).astype(int)
            binary_cols[f'BINARY_{col}_DOWN'] = (self.df[col] < self.df[col].shift(1)).astype(int)

        # Concatenar todas las columnas binarias de una vez
        self.df = pd.concat([self.df, pd.DataFrame(binary_cols, index=self.df.index)], axis=1)
        return self
    
    

    
    def add_trend_angle(
        self,
        column='Close',
        window=10,
        horizon=5,
        max_angle=75.0,
        target_col_name=None
         ):
        """
        Añade columna TARGET_TREND_ANG_{window}_{horizon} con el ángulo de tendencia universal.

        Descripción:
        - Para cada fila t, se ajusta una regresión lineal sobre los precios en una ventana centrada en t.
        - Si window > horizon: incluye puntos pasados y futuros.
        - Si window == horizon: incluye solo puntos futuros.
        - A partir de la pendiente (slope), se calcula un ángulo universal en grados:
            ang = arctan(slope / (precio_actual * r))
        donde r es un cambio de referencia (por defecto 1% diario → r = 0.01)
        - Este ángulo se transforma linealmente al rango [0,1]:
            -75° → 0.0
            0°  → 0.5
            +75° → 1.0
        Valores fuera del rango ±75° se recortan (clipped).
        """

        vals = self.df[column].values
        n = len(vals)
        angles = np.full(n, np.nan)
        r = 0.01  # Cambio relativo de referencia: 1% diario

        for i in range(n):
            if window > horizon:
                start = i - (window - horizon - 1)
                end = i + horizon + 1
            else:
                start = i + 1
                end = i + horizon + 1
                
            
            if window > horizon:
                # Caso clásico: hay pasado extra
                start = i - (window - horizon - 1)
                end   = i + horizon + 1
            else:
                # Caso nuevo: manda el horizon
                # Tomamos exactamente W puntos dentro del rango [t .. t+H]
                start = i + (horizon - window + 1)
                end   = i + horizon + 1

            if start < 0 or end > n:
                continue

            y = vals[start:end]
            x = np.arange(len(y))
            if np.any(np.isnan(y)):
                continue

            slope = np.polyfit(x, y, 1)[0]
            precio_actual = vals[i]
            if precio_actual == 0 or np.isnan(precio_actual):
                continue

            # Ángulo universal (en grados)
            ang_rad = np.arctan(slope / (precio_actual * r))
            ang_deg = np.degrees(ang_rad)

            # Normalizar linealmente al rango [0, 1]
            ang_deg = np.clip(ang_deg, -max_angle, max_angle)
            norm_angle = (ang_deg + max_angle) / (2 * max_angle)

            angles[i] = norm_angle

        if not target_col_name:
            target_col_name = f'TARGET_TREND_ANG_{window}_{horizon}'

        self.df[target_col_name] = angles
        return self




    
    def add_target_roc(self, horizons=[5, 10, 20]):
        """
        Añade columnas TARGET_ROC_{h} para cada horizonte de predicción en 'horizons'.
        El ROC es el % de variación desde t hasta t+h (futuro).
        """
        for h in horizons:
            # Cálculo del ROC futuro: % cambio de Close entre t y t+h
            self.df[f'TARGET_ROC_{h}'] = 100 * (self.df['Close'].shift(-h) / self.df['Close'] - 1)
        return self
    
    

    def add_targets(
        self,
        column='Close',    # Columna de precios a usar
        window=10,         # Tamaño total de ventana para regresión lineal
        horizon=5,         # Cuántos puntos al futuro (horizon >= 1)
        target_col_name=None
        ):
        """
        Añade columna TARGET_SLOPE_{window}_{horizon} :
        - Para cada fila t, calcula la pendiente de la recta que ajusta los valores de precios desde
        t - (window - horizon - 1) hasta t + horizon (ambos incluidos).
        - Si window == horizon, solo usa los puntos futuros (desde t+1 hasta t+horizon).
        - Si window > horizon, usa (window-horizon) puntos al pasado (incluyendo t), y horizon puntos al futuro.
        - Aplica el suavizado sigmoidal "bipolar" a la pendiente.
        - La salida está alineada para ser usada como y(t).
        """
        vals = self.df[column].values
        n = len(vals)
        slopes = np.full(n, np.nan)
        for i in range(n):
            # Definir índices para la ventana
            if window > horizon:
                # Incluye pasado y futuro: [t-(window-horizon-1), ..., t, ..., t+horizon]
                start = i - (window - horizon - 1)
                end = i + horizon + 1  # Python: excluye el final
            else:
                # Solo futuro: [t+1, ..., t+horizon]
                start = i + 1
                end = i + horizon + 1

            if start < 0 or end > n:
                continue

            y = vals[start:end]
            x = np.arange(len(y))
            if np.any(np.isnan(y)):
                continue

            m = np.polyfit(x, y, 1)[0]  # Pendiente
            slopes[i] = m

        targets = self.sigmoid_bipolar(slopes)

        if not target_col_name:
            target_col_name = f"TARGET_SLOPE_{window}_{horizon}"

        self.df[target_col_name] = targets
        return self


    # Función auxiliar para aplicar la normalización sigmoidal bipolar
    def sigmoid_bipolar(x):
        return 0.5 + (((2.0 / (1.0 + np.exp(-x))) - 1.0) / 2.0)




    def get_df(self):

        #return self.df
       
        # Redondeo a 5 decimales para reducir tamaño de dataset
        return self.df.round(5)
