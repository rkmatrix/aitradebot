import backtrader as bt
import pandas as pd
import sqlite3
from datetime import datetime
import collections

# --- Strategy Definition (Copied from backtester.py) ---
# We use the exact same strategy class here.
class MA_RSI_Strategy(bt.Strategy):
    params = (
        ('ma_short', 50),
        ('ma_long', 200),
        ('rsi_low', 30),
        ('rsi_high', 70),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.sma_short = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.ma_short)
        self.sma_long = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.ma_long)
        self.rsi = bt.indicators.RSI(self.datas[0], period=14)
        self.ma_crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        # Suppress logging during optimization for cleaner output
        # print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None

    def next(self):
        if self.order:
            return
        if not self.position:
            if self.ma_crossover > 0 and self.rsi < self.params.rsi_low:
                self.order = self.buy()
        else:
            if self.ma_crossover < 0 or self.rsi > self.params.rsi_high:
                self.order = self.sell()
    
    def stop(self):
        # This method is called at the end of a backtest run
        # We can use it to store the final value of the portfolio
        self.log(f'(MA Short {self.params.ma_short}, MA Long {self.params.ma_long}, RSI Low {self.params.rsi_low}, RSI High {self.params.rsi_high}) Final Value: {self.broker.getvalue():.2f}', 
                 self.datas[0].datetime.date(0))


# --- Data Loading ---
def load_data(ticker, db_file):
    """ Loads data for a specific ticker from the SQLite database. """
    table_name = f"{ticker.replace('.', '_')}_ohlcv"
    conn = sqlite3.connect(db_file)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# --- Optimization Execution ---
def run_optimization(ticker, db_file):
    """ Sets up and runs the backtrader optimization. """
    cerebro = bt.Cerebro()

    # Add the strategy to optimize
    # We provide a range of values for the parameters we want to test.
    cerebro.optstrategy(
        MA_RSI_Strategy,
        ma_short=range(20, 71, 10),  # Test short MAs from 20 to 70 in steps of 10
        ma_long=range(100, 201, 20), # Test long MAs from 100 to 200 in steps of 20
        rsi_low=range(20, 36, 5),   # Test RSI low from 20 to 35 in steps of 5
        # rsi_high is kept constant for this example to reduce complexity
    )

    # Load data
    dataframe = load_data(ticker, db_file)
    if dataframe.empty:
        return
        
    data = bt.feeds.PandasData(dataname=dataframe)
    cerebro.adddata(data)

    # Set our starting cash
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.001)

    # Add an analyzer to evaluate the results
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    print(f'\n--- Starting Optimization for {ticker} ---')
    print("This may take a while...")

    # Run the optimization
    opt_results = cerebro.run(stdstats=False) # stdstats=False to keep output clean

    print(f'--- Optimization Complete for {ticker} ---')

    # --- Process and display results ---
    final_results = []
    for run in opt_results:
        for strategy in run:
            params = strategy.params
            returns = strategy.analyzers.returns.get_analysis()
            final_results.append({
                'ma_short': params.ma_short,
                'ma_long': params.ma_long,
                'rsi_low': params.rsi_low,
                'rsi_high': params.rsi_high,
                'total_return': returns.get('rtot', 0)
            })

    if not final_results:
        print("No results to display.")
        return

    results_df = pd.DataFrame(final_results)
    # Sort the results by total return in descending order
    results_df.sort_values(by='total_return', ascending=False, inplace=True)

    print("\n--- Top 5 Best Performing Parameter Combinations ---")
    print(results_df.head(5))

    best_params = results_df.iloc[0]
    print("\n--- Best Parameters Found ---")
    print(f"Short MA Period: {best_params['ma_short']}")
    print(f"Long MA Period:  {best_params['ma_long']}")
    print(f"RSI Low Threshold: {best_params['rsi_low']}")
    print(f"Best Total Return: {best_params['total_return'] * 100:.2f}%")


if __name__ == '__main__':
    TICKERS_TO_OPTIMIZE = ['AAPL', 'ORCL', 'TSLA', 'SPY', 'AMZN', 'NVDA']
    DB_FILE = 'trading_data.db'
    
    for ticker in TICKERS_TO_OPTIMIZE:
        print(f"\n{'='*60}")
        print(f"               RUNNING OPTIMIZATION FOR: {ticker}")
        print(f"{'='*60}")
        run_optimization(ticker, DB_FILE)
        print(f"\nOptimization for {ticker} complete.")

