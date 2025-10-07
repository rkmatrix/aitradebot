import os

# --- Securely load API Keys & Cloudflare R2 Config from Environment Variables ---
# These MUST be set in the Render dashboard
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

R2_ENDPOINT_URL = os.getenv('R2_ENDPOINT_URL')
R2_BUCKET_NAME = os.getenv('R2_BUCKET_NAME')
R2_ACCESS_KEY_ID = os.getenv('R2_ACCESS_KEY_ID')
R2_SECRET_ACCESS_KEY = os.getenv('R2_SECRET_ACCESS_KEY')

# --- File Paths & Names ---
# The model is no longer a local file path, but an object key in R2
MODEL_FILENAME = "ultimate_trading_model.pkl"

# --- Bot & Model Configuration ---
TICKERS = ['AAPL', 'TSLA', 'AMZN', 'NVDA', 'GOOG', 'MSFT']
MARKET_REGIME_TICKER = 'SPY'
DB_FILE = "trading_data.db" 

FEATURE_COLUMNS = [
    'ma_crossover_signal', 'rsi_signal', 'bb_signal', 'macd_signal', 'market_regime'
]
TARGET_COLUMN = 'target'

# Options Strategy Parameters (centralized for easy access)
DAYS_TO_EXPIRATION_MIN = 30
DAYS_TO_EXPIRATION_MAX = 45
STRIKE_PRICE_OFFSET = 3
POSITION_CLOSE_DTE = 5

