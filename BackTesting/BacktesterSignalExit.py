import pandas as pd
import os

class BacktesterSignalExit:
    """
    Backtester con salida adaptativa:
    - Entrada por señal (> threshold_buy)
    - Salida por: take profit, stop loss, o reversión de señal del modelo (< threshold_exit)
    """
    def __init__(self, threshold_buy=0.51, threshold_exit=0.4, capital_inicial=10000,
                 tp_pct=0.015, sl_pct=0.03, save_trades=False):
        self.threshold_buy = threshold_buy
        self.threshold_exit = threshold_exit
        self.capital_inicial = capital_inicial
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.save_trades = save_trades

    def run(self, df_test, y_pred, symbol):
        THRESHOLD_BUY_GRADOS = (self.threshold_buy * 150) - 75
        THRESHOLD_EXIT_GRADOS = (self.threshold_exit * 150) - 75
        df_test = df_test.copy()
        df_test['y_pred'] = y_pred

        position = 0
        entry_price = 0
        num_shares = 0
        capital_actual = self.capital_inicial
        trade_log = []
        take_profit_price = None
        stop_loss_price = None

        for idx, row in df_test.iterrows():
            fecha = row['Fecha']
            signal = row['y_pred']
            price = row['Close']

            if position == 0:
                if signal > self.threshold_buy:
                    num_shares = capital_actual / price
                    position = 1
                    entry_price = price
                    take_profit_price = entry_price * (1 + self.tp_pct)
                    stop_loss_price  = entry_price * (1 - self.sl_pct)
                    trade_log.append([
                        fecha, 'BUY', price, 0, num_shares, 0, capital_actual, signal,
                        take_profit_price, stop_loss_price, entry_price, '',
                        self.threshold_buy, THRESHOLD_BUY_GRADOS
                    ])
            elif position == 1:
                profit = 0
                cause = ''
                # Salida por take profit (Aqui no se utiliza, salimos por señal de modelo o stop loss)
                
                # if price >= take_profit_price:
                #     profit = num_shares * (price - entry_price)
                #     capital_actual += profit
                #     cause = 'SELL_TAKEPROFIT'
                
                # Salida por stop loss (salvaguardamos la posición si el precio cae por debajo del stop loss)
                if price <= stop_loss_price:
                    profit = num_shares * (price - entry_price)
                    capital_actual += profit
                    cause = 'SELL_STOPLOSS'
                
                # Salida por reversión de señal del modelo (ángulo < threshold_exit)
                elif signal < self.threshold_exit:
                    profit = num_shares * (price - entry_price)
                    capital_actual += profit
                    cause = 'SELL_SIGNAL_EXIT'

                if cause:
                    trade_log.append([
                        fecha, cause, price, profit, num_shares, 0, capital_actual, signal,
                        take_profit_price, stop_loss_price, entry_price, '',
                        self.threshold_buy, THRESHOLD_BUY_GRADOS
                    ])
                    position = 0
                    num_shares = 0
                    entry_price = 0
                    take_profit_price = None
                    stop_loss_price = None

        columns = [
            'Fecha', 'Accion', 'Precio', 'Profit', 'Num_Shares', 'Dias_Posicion',
            'Capital_Actual', 'Signal', 'TakeProfit', 'StopLoss', 'Entry_Price', 'Comentario',
            'Threshold_Buy', 'Threshold_Buy_Grados'
        ]
        trade_log_df = pd.DataFrame(trade_log, columns=columns)

        if self.save_trades:
            os.makedirs("Resultados", exist_ok=True)
            trade_log_path = f"Resultados/Backtesting_TradeLog_{symbol}_SignalExit.csv"
            trade_log_df.to_csv(trade_log_path, index=False, sep=';')
            print(f"\nTrade log guardado en: {trade_log_path}\n")

        profits = trade_log_df[trade_log_df['Accion'].str.startswith('SELL')]['Profit'].tolist()
        num_trades = len(profits)
        num_winners = sum(1 for p in profits if p > 0)
        num_losers = sum(1 for p in profits if p <= 0)
        total_profit = sum(profits)
        win_loss_ratio = num_winners / num_losers if num_losers > 0 else float('inf')

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

        roi = ((capital_actual - self.capital_inicial) / self.capital_inicial) * 100

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
