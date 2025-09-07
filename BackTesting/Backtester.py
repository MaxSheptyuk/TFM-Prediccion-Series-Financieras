import pandas as pd
import os

class Backtester:
    """
    Clase para ejecutar un backtesting de trading discrecional sobre una serie temporal de precios y señales de trading.

    Esta clase permite simular una estrategia de trading tipo “compra-venta” en base a señales generadas
    (por ejemplo, por un modelo ML), aplicando reglas de entrada y salida parametrizables.

    Resultados: ROI, curva de capital, drawdown, log de operaciones, etc.
    """

    def __init__(self, threshold_buy=0.51, paciencia_max_dias=5, capital_inicial=10000,
                 tp_pct=0.015, sl_pct=0.03, save_trades=False):
        """
        Inicializa el backtester con los parámetros principales de la estrategia.

        Parámetros:
        -----------
        threshold_buy : float
            Umbral mínimo de señal para abrir una compra (por ejemplo, una predicción de tendencia).
        paciencia_max_dias : int
            Días máximos que se mantiene una posición abierta si no se alcanza ni TP ni SL.
        capital_inicial : float
            Capital inicial con el que comienza la simulación de trading.
        tp_pct : float
            Take profit (beneficio máximo) expresado como porcentaje sobre el precio de entrada (ej: 0.015 = 1.5%).
        sl_pct : float
            Stop loss (pérdida máxima) como porcentaje sobre el precio de entrada (ej: 0.03 = 3%).
        save_trades : bool
            Si es True, guarda el log completo de operaciones en un CSV en la carpeta Resultados/.
        """
        self.threshold_buy = threshold_buy
        self.paciencia_max_dias = paciencia_max_dias
        self.capital_inicial = capital_inicial
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.save_trades = save_trades
        self.trade_log = []

    def run(self, df_test, y_pred, symbol):
        """
        Ejecuta el backtesting sobre un conjunto de datos y señales dadas.

        Parámetros:
        -----------
        df_test : pd.DataFrame
            DataFrame con las columnas originales de precios (debe incluir 'Fecha' y 'Close').
        y_pred : array-like
            Array o Serie con las señales/predicciones generadas por el modelo.
        symbol : str
            Nombre del activo o ticker (para guardar los logs).

        Retorna:
        --------
        dict con métricas principales: ROI, curva de capital, drawdown, log de operaciones, etc.
        """
        # Convierte la señal de umbral normalizado [0, 1] a grados [-75, +75] (solo para log)
        THRESHOLD_BUY_GRADOS = (self.threshold_buy * 150) - 75
        df_test = df_test.copy()
        df_test['y_pred'] = y_pred

        # Variables para el control del estado de la posición y capital
        position = 0           # 0 = sin posición, 1 = comprado
        entry_price = 0        # Precio de entrada de la última compra
        num_shares = 0         # Número de acciones compradas en la posición actual
        capital_actual = self.capital_inicial
        self.trade_log = []    # Lista para guardar el log de operaciones
        patience_counter = 0   # Contador de días desde la compra
        take_profit_price = None
        stop_loss_price = None

        # --- Bucle principal: simulamos cada día/trade ---
        for idx, row in df_test.iterrows():
            fecha = row['Fecha']
            signal = row['y_pred']
            price = row['Close']

            # Si no tenemos ninguna operación abierta, evaluamos si abrir una compra
            if position == 0:
                if signal > self.threshold_buy:
                    num_shares = capital_actual / price    # Compra todo el capital disponible
                    position = 1
                    entry_price = price
                    patience_counter = 0
                    take_profit_price = entry_price * (1 + self.tp_pct)
                    stop_loss_price  = entry_price * (1 - self.sl_pct)
                    self.trade_log.append([
                        fecha, 'BUY', price, 0, num_shares, 0, capital_actual, signal,
                        take_profit_price, stop_loss_price, entry_price, '',
                        self.threshold_buy, THRESHOLD_BUY_GRADOS
                    ])
            # Si ya tenemos una posición abierta, evaluamos si hay que vender por TP, SL o paciencia
            elif position == 1:
                patience_counter += 1
                profit = 0
                cause = ''
                # Cierre por take profit (precio alcanza objetivo de beneficio)
                if price >= take_profit_price:
                    profit = num_shares * (price - entry_price)
                    capital_actual += profit
                    cause = 'SELL_TAKEPROFIT'
                # Cierre por stop loss (precio baja hasta pérdida máxima aceptada)
                elif price <= stop_loss_price:
                    profit = num_shares * (price - entry_price)
                    capital_actual += profit
                    cause = 'SELL_STOPLOSS'
                # Cierre por paciencia máxima (días límite sin alcanzar TP ni SL)
                elif patience_counter >= self.paciencia_max_dias:
                    profit = num_shares * (price - entry_price)
                    capital_actual += profit
                    cause = 'SELL_PATIENCE'

                # Si se cierra la operación por cualquier motivo, registramos el trade y reseteamos el estado
                if cause:
                    self.trade_log.append([
                        fecha, cause, price, profit, num_shares, patience_counter, capital_actual, signal,
                        take_profit_price, stop_loss_price, entry_price, '',
                        self.threshold_buy, THRESHOLD_BUY_GRADOS
                    ])
                    position = 0
                    num_shares = 0
                    entry_price = 0
                    take_profit_price = None
                    stop_loss_price = None

        # --- Construcción del log de operaciones como DataFrame ---
        columns = [
            'Fecha', 'Accion', 'Precio', 'Profit', 'Num_Shares', 'Dias_Posicion',
            'Capital_Actual', 'Signal', 'TakeProfit', 'StopLoss', 'Entry_Price', 'Comentario',
            'Threshold_Buy', 'Threshold_Buy_Grados'
        ]
        trade_log_df = pd.DataFrame(self.trade_log, columns=columns)

        # Guarda el log en CSV si está activado
        if self.save_trades:
            os.makedirs("Resultados", exist_ok=True)
            trade_log_path = f"Resultados/Backtesting_TradeLog_{symbol}.csv"
            trade_log_df.to_csv(trade_log_path, index=False, sep=';')
            print(f"\nTrade log guardado en: {trade_log_path}\n")

        # --- Cálculo de métricas clave del backtest ---
        profits = trade_log_df[trade_log_df['Accion'].str.startswith('SELL')]['Profit'].tolist()
        num_trades = len(profits)
        num_winners = sum(1 for p in profits if p > 0)
        num_losers = sum(1 for p in profits if p <= 0)
        total_profit = sum(profits)
        win_loss_ratio = num_winners / num_losers if num_losers > 0 else float('inf')

        # Cálculo de la curva de capital y drawdown
        balance_curve = []
        running_balance = self.capital_inicial
        max_balance = self.capital_inicial
        max_drawdown = 0

        for profit in profits:
            running_balance += profit
            balance_curve.append(running_balance)
            if running_balance > max_balance:
                max_balance = running_balance
            drawdown = max_balance - running_balance
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # ROI total obtenido en el periodo
        roi = ((capital_actual - self.capital_inicial) / self.capital_inicial) * 100

        # Devuelve todas las métricas y logs principales
        return {
            "capital_inicial": self.capital_inicial,
            "capital_final": capital_actual,
            "roi": roi,
            "num_trades": num_trades,
            "num_winners": num_winners,
            "num_losers": num_losers,
            "total_profit": total_profit,
            "win_loss_ratio": win_loss_ratio,
            "max_drawdown": max_drawdown,
            "trade_log": trade_log_df,
            "balance_curve": balance_curve,
            "fechas": list(df_test['Fecha'])
        }
