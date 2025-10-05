import backtrader as bt
import pandas as pd
import sqlite3
from datetime import datetime

# --- Step 1: Strategy Definition ---
# Define the trading strategy based on user requirements
class MA_RSI_Strategy(bt.Strategy):
    """
    A trading strategy that combines Moving Average Crossover and RSI.
    - Buy Signal: 50-day MA crosses above 200-day MA AND RSI is below 30.
    - Sell Signal: 50-day MA crosses below 200-day MA OR RSI is above 70.
    """
    params = (
        ('ma_short', 50),
        ('ma_long', 200),
        ('rsi_low', 30),
        ('rsi_high', 70),
    )

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders
        self.order = None

        # Add the indicators
        self.sma_short = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.ma_short)
        
        self.sma_long = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.ma_long)
        
        self.rsi = bt.indicators.RSI(self.datas[0], period=14)

        # Crossover signal
        self.ma_crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def next(self):
        # Simply log the closing price of the series from the reference
        # self.log(f'Close, {self.dataclose[0]:.2f}')

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            # Not in the market, look for a buy signal
            if self.ma_crossover > 0 and self.rsi < self.params.rsi_low:
                self.log(f'BUY CREATE, Close: {self.dataclose[0]:.2f}')
                self.order = self.buy()

        else:
            # Already in the market, look for a sell signal
            if self.ma_crossover < 0 or self.rsi > self.params.rsi_high:
                self.log(f'SELL CREATE, Close: {self.dataclose[0]:.2f}')
                self.order = self.sell()


# --- Step 2: Data Loading ---
def load_data(ticker, db_file):
    """ Loads data for a specific ticker from the SQLite database. """
    table_name = f"{ticker.replace('.', '_')}_ohlcv"
    conn = sqlite3.connect(db_file)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        print(f"Loaded {len(df)} rows of data for {ticker}.")
        return df
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# --- Step 3: Backtesting Execution ---
def run_backtest(ticker, db_file):
    """ Sets up and runs the backtrader engine. """
    cerebro = bt.Cerebro()

    # Add the strategy
    cerebro.addstrategy(MA_RSI_Strategy)

    # Load data
    dataframe = load_data(ticker, db_file)
    if dataframe.empty:
        return
        
    data = bt.feeds.PandasData(dataname=dataframe)
    cerebro.adddata(data)

    # Set our starting cash
    start_cash = 10000.0
    cerebro.broker.setcash(start_cash)
    
    # Set commission
    cerebro.broker.setcommission(commission=0.001) # 0.1% commission

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    print(f'\n--- Starting Backtest for {ticker} ---')
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')

    # Run over everything
    results = cerebro.run()
    strat = results[0]

    # Print out the final result
    end_cash = cerebro.broker.getvalue()
    print(f'Final Portfolio Value:   {end_cash:.2f}')
    print(f'Total Return:            {(end_cash - start_cash) / start_cash * 100:.2f}%')
    
    # Print analyzers
    print('\n--- Performance Metrics ---')
    print(f"Sharpe Ratio: {strat.analyzers.sharpe_ratio.get_analysis().get('sharperatio', 'N/A')}")
    print(f"Max Drawdown: {strat.analyzers.drawdown.get_analysis().max.drawdown:.2f}%")
    
    # Plot the results
    print("\nGenerating plot...")
    # The plot call is blocking, so the script will pause here until the plot window is closed.
    cerebro.plot(style='candlestick', barup='green', bardown='red')


if __name__ == '__main__':
    TICKERS_TO_TEST = ['AAPL', 'ORCL', 'TSLA', 'SPY', 'AMZN', 'NVDA']
    DB_FILE = 'trading_data.db'

    for ticker in TICKERS_TO_TEST:
        print(f"\n{'='*60}")
        print(f"               RUNNING BACKTEST FOR: {ticker}")
        print(f"{'='*60}")
        run_backtest(ticker, DB_FILE)
        print(f"\nBacktest for {ticker} complete.")
        print("-> Close the plot window to continue to the next ticker.")

