import os
import time
import pandas as pd
import pandas_ta as ta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
import joblib
from datetime import datetime, timedelta
import config
from stacking_model import StackingEnsemble
import boto3
import io
import threading

# --- Options Strategy Parameters ---
DAYS_TO_EXPIRATION_MIN = 30
DAYS_TO_EXPIRATION_MAX = 45
STRIKE_PRICE_OFFSET = 3
POSITION_CLOSE_DTE = 5

# This will be loaded from R2 cloud storage
ultimate_model = None

def load_model_from_r2():
    """Downloads the model from R2, loads it into memory."""
    global ultimate_model
    if ultimate_model is not None: return True
    
    if not all([config.R2_ENDPOINT_URL, config.R2_BUCKET_NAME, config.R2_ACCESS_KEY_ID, config.R2_SECRET_ACCESS_KEY]):
        print("FATAL: R2 environment variables not set. Cannot download model.")
        return False
    try:
        print(f"Downloading model '{config.MODEL_FILENAME}' from R2 bucket...")
        s3_client = boto3.client(
            's3',
            endpoint_url=config.R2_ENDPOINT_URL,
            aws_access_key_id=config.R2_ACCESS_KEY_ID,
            aws_secret_access_key=config.R2_SECRET_ACCESS_KEY,
            region_name="auto"
        )
        with io.BytesIO() as buffer:
            s3_client.download_fileobj(config.R2_BUCKET_NAME, config.MODEL_FILENAME, buffer)
            buffer.seek(0)
            ultimate_model = joblib.load(buffer)
        print("Successfully loaded Ultimate AI Model from cloud storage.")
        return True
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to download or load model from R2: {e}")
        return False

def get_stock_market_data(data_client, ticker, limit=250):
    """Fetches historical stock data from Alpaca with error handling."""
    try:
        request_params = StockBarsRequest(symbol_or_symbols=[ticker], timeframe=TimeFrame.Day, limit=limit)
        bars = data_client.get_stock_bars(request_params).df
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.reset_index(level=0, drop=True)
        bars.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True, errors='ignore')
        return bars
    except Exception as e:
        print(f"  Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()

def get_live_features(data_client, ticker):
    """Computes advanced features with extreme defensive error handling."""
    try:
        df = get_stock_market_data(data_client, ticker)
        spy_df = get_stock_market_data(data_client, config.MARKET_REGIME_TICKER)
        if df.empty or spy_df.empty or len(df) < 201 or len(spy_df) < 201:
            return None

        # Calculate all indicators within individual try-except blocks
        try: df['ma_crossover_signal'] = (ta.sma(df['Close'], 50) > ta.sma(df['Close'], 200)).astype(int).replace(0, -1)
        except Exception: df['ma_crossover_signal'] = 0

        try:
            rsi = ta.rsi(df['Close']); df['rsi_signal'] = 0
            if rsi is not None: df.loc[rsi < 30, 'rsi_signal'] = 1; df.loc[rsi > 70, 'rsi_signal'] = -1
        except Exception: df['rsi_signal'] = 0

        try:
            bbands = ta.bbands(df['Close']); df['bb_signal'] = 0
            if bbands is not None and not bbands.empty:
                lower = next((c for c in bbands.columns if 'bbl' in c.lower()), None)
                upper = next((c for c in bbands.columns if 'bbu' in c.lower()), None)
                if lower and upper: df.loc[df['Close'] < bbands[lower], 'bb_signal'] = 1; df.loc[df['Close'] > bbands[upper], 'bb_signal'] = -1
        except Exception: df['bb_signal'] = 0
        
        try:
            macd = ta.macd(df['Close']); df['macd_signal'] = 0
            if macd is not None and not macd.empty:
                macd_line = next((c for c in macd.columns if 'macd_' in c.lower()), None)
                signal_line = next((c for c in macd.columns if 'macds' in c.lower()), None)
                if macd_line and signal_line: df['macd_signal'] = (macd[macd_line] > macd[signal_line]).astype(int).replace(0, -1)
        except Exception: df['macd_signal'] = 0
        
        try:
            spy_df['sma_200'] = ta.sma(spy_df['Close'], 200)
            df['market_regime'] = 1 if spy_df['Close'].iloc[-1] > spy_df['sma_200'].iloc[-1] else -1
        except Exception: df['market_regime'] = 0
        
        df.dropna(inplace=True)
        return df.iloc[-1] if not df.empty else None
    except Exception as e:
        print(f"  MAJOR ERROR in get_live_features for {ticker}: {e}")
        return None

def find_target_option(trading_client, data_client, ticker, prediction):
    """Finds a suitable options contract with defensive checks."""
    try:
        # ... (This logic is sound, but we ensure it fails gracefully)
        pass # Placeholder for the full find_target_option logic from previous versions
    except Exception as e:
        print(f"  An error occurred while finding an option for {ticker}: {e}")
    return None

def run_trader(stop_event):
    """Main bot loop, now accepts a stop_event and loads model from R2."""
    print("\n--- Starting Ultimate AI Options Trading Bot ---")
    
    if not load_model_from_r2(): 
        print("Shutting down bot due to model loading failure.")
        return
        
    trading_client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)
    data_client = StockHistoricalDataClient(config.API_KEY, config.SECRET_KEY)
    
    while not stop_event.is_set():
        try:
            print(f"\n[{time.ctime()}] Running AI Options trading cycle...")
            
            for ticker in config.TICKERS:
                if stop_event.is_set(): break
                try:
                    print(f"\n- Analyzing {ticker} for new entry")
                    
                    features = get_live_features(data_client, ticker)
                    if features is None:
                        print(f"  Could not generate live features for {ticker}. Skipping.")
                        continue
                    
                    if not all(col in features.index for col in config.FEATURE_COLUMNS):
                        print(f"  Feature set for {ticker} is incomplete. Skipping prediction.")
                        continue

                    feature_df = pd.DataFrame([features])[config.FEATURE_COLUMNS]
                    prediction = ultimate_model.predict(feature_df.values)[0]
                    
                    print(f"  Ultimate AI Prediction for {ticker}: {'UP (Buy Call)' if prediction == 1 else 'DOWN/STAY (Hold)'}")

                    if prediction == 1:
                        # (Find and place order logic here)
                        pass

                except Exception as e:
                    print(f"  An unhandled error occurred while analyzing {ticker}: {e}")

            if stop_event.is_set(): break

            print("\nCycle complete. Waiting for 15 minutes...")
            for _ in range(15 * 60):
                if stop_event.is_set(): break
                time.sleep(1)

        except Exception as e:
            print(f"A critical error occurred in the main loop: {e}")
            time.sleep(60)
    
    print("--- AI Trading Bot has received stop signal and is shutting down. ---")

if __name__ == '__main__':
    run_trader(threading.Event())

