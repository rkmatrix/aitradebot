import os

# --- Securely load API Keys from Environment Variables ---
# These MUST be set in the Render dashboard's "Environment" tab
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

# --- File Paths for Persistent Storage ---
# This path points to the persistent disk we will create on Render.
RENDER_DISK_PATH = "/var/data"
MODEL_FILENAME = os.path.join(RENDER_DISK_PATH, "ultimate_trading_model.pkl")

# --- Bot & Model Configuration ---
TICKERS = ['AAPL', 'TSLA', 'AMZN', 'NVDA', 'GOOG', 'MSFT']
MARKET_REGIME_TICKER = 'SPY'
DB_FILE = "trading_data.db" # This is only used during the temporary setup process

FEATURE_COLUMNS = [
    'ma_crossover_signal', 'rsi_signal', 'bb_signal', 'macd_signal', 'market_regime'
]
TARGET_COLUMN = 'target'

# --- Sanity Check ---
# This helps diagnose configuration issues on server startup.
if not API_KEY or not SECRET_KEY:
    print("FATAL STARTUP ERROR: ALPACA_API_KEY and/or ALPACA_SECRET_KEY are not set in the environment.")

