import os

# --- Securely load API Keys from Environment Variables ---
# For local testing, you can temporarily set them here if not using environment variables
# For Render, these will be set in the dashboard

#### API_KEY = os.getenv('ALPACA_API_KEY', 'YOUR_ALPACA_API_KEY_ID')
#### SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', 'YOUR_ALPACA_SECRET_KEY')

# --- Cloudflare R2 Cloud Storage Configuration ---
#### R2_ENDPOINT_URL = os.getenv('R2_ENDPOINT_URL')
#### R2_BUCKET_NAME = os.getenv('R2_BUCKET_NAME')
#### R2_ACCESS_KEY_ID = os.getenv('R2_ACCESS_KEY_ID')
#### R2_SECRET_ACCESS_KEY = os.getenv('R2_SECRET_ACCESS_KEY')


API_KEY = os.getenv('ALPACA_API_KEY', 'PKYVIZ6Y0GGUCQHGI97V')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '2vHaDIDPzfbtgn5aNVm5P3LEuhVzcRpd4CChOinS')

# --- Cloudflare R2 Cloud Storage Configuration ---
R2_ENDPOINT_URL = os.getenv('https://3b5309f44aeba454d6ca25b57e61ff55.r2.cloudflarestorage.com/aitradepro-model-storage')
R2_BUCKET_NAME = os.getenv('aitradepro-model-storage')
R2_ACCESS_KEY_ID = os.getenv('fc613b819ad539a95ca031a907c90e71')
R2_SECRET_ACCESS_KEY = os.getenv('28853a3433520136f2c139fe757422419ca90009227549146ea51bae67424b77')

# Set the operating environment.
# 'local': Saves the AI model to a local file.
# 'cloud': Uploads the AI model to Cloudflare R2.
ENVIRONMENT = "local" # Or "cloud"

# --- Bot & Model Configuration (DO NOT EDIT) ---
MODEL_FILENAME = "ultimate_trading_model.pkl" 
TICKERS = ['AAPL', 'TSLA', 'AMZN', 'NVDA', 'GOOG', 'MSFT']
MARKET_REGIME_TICKER = 'SPY'
DB_FILE = "trading_data_temp.db"

FEATURE_COLUMNS = [
    'ma_crossover_signal', 'rsi_signal', 'bb_signal', 'macd_signal', 'market_regime'
]
TARGET_COLUMN = 'target'

