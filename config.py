import os

API_KEY = os.getenv('ALPACA_API_KEY', 'PKYVIZ6Y0GGUCQHGI97V')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '2vHaDIDPzfbtgn5aNVm5P3LEuhVzcRpd4CChOinS')

# --- Ticker Symbols ---
# This is the single source of truth for the tickers you want to trade
TICKERS = ['AAPL', 'TSLA', 'AMZN', 'NVDA', 'GOOG', 'MSFT']

# The ticker used to determine the overall market trend (regime).
# SPY (S&P 500 ETF) is the standard choice for the US market.
MARKET_REGIME_TICKER = 'SPY'

# --- File Paths ---
# The name of the SQLite database file where price data will be stored.
DB_FILE = "trading_data.db"

# The name of the file where the trained AI model will be saved.
MODEL_FILENAME = "ultimate_trading_model.pkl"