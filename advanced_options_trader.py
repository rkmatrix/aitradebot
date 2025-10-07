import os
import time
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import config
from stacking_model import StackingEnsemble
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
import boto3
import io

# This will be loaded into the global scope when the trader starts
ultimate_model = None

def load_model_from_r2():
    """Downloads the model from Cloudflare R2 and loads it into memory."""
    global ultimate_model
    if ultimate_model is None:
        try:
            # Check for necessary R2 configuration
            if not all([config.R2_ENDPOINT_URL, config.R2_BUCKET_NAME, config.R2_ACCESS_KEY_ID, config.R2_SECRET_ACCESS_KEY]):
                print("CRITICAL ERROR: Cloudflare R2 environment variables are not fully set.")
                return False

            print(f"Downloading model '{config.MODEL_FILENAME}' from R2 bucket '{config.R2_BUCKET_NAME}'...")
            s3_client = boto3.client(
                's3',
                endpoint_url=config.R2_ENDPOINT_URL,
                aws_access_key_id=config.R2_ACCESS_KEY_ID,
                aws_secret_access_key=config.R2_SECRET_ACCESS_KEY,
                region_name="auto"
            )
            # Download model to an in-memory buffer
            with io.BytesIO() as buffer:
                s3_client.download_fileobj(config.R2_BUCKET_NAME, config.MODEL_FILENAME, buffer)
                buffer.seek(0)
                ultimate_model = joblib.load(buffer)
            
            print("Successfully loaded Ultimate AI Model from R2.")
            return True
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load model from R2. Error: {e}")
            print("Please run 'Full Setup' to train and upload the model.")
            return False
    return True

def get_stock_market_data(data_client, ticker, limit=250):
    """Fetches historical stock data from Alpaca with robust error handling."""
    try:
        request_params = StockBarsRequest(symbol_or_symbols=[ticker], timeframe=TimeFrame.Day, limit=limit)
        bars = data_client.get_stock_bars(request_params).df
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.reset_index(level=0, drop=True)
        # Standardize column names
        bars.columns = [col.capitalize() for col in bars.columns]
        return bars
    except Exception as e:
        print(f"  Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    """Calculates all necessary technical indicators manually with error handling."""
    try:
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['sma_200'] = df['Close'].rolling(window=200).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['bb_ma'] = df['Close'].rolling(window=20).mean()
        df['bb_std'] = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_ma'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_ma'] - (df['bb_std'] * 2)
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
    except Exception as e:
        print(f"  Error calculating indicators: {e}")
    return df

def get_live_features(data_client, ticker):
    """Computes advanced features for live data with extreme defensive error handling."""
    try:
        df = get_stock_market_data(data_client, ticker)
        spy_df = get_stock_market_data(data_client, config.MARKET_REGIME_TICKER)
        if df.empty or spy_df.empty or len(df) < 201 or len(spy_df) < 201:
            return None

        df = calculate_indicators(df)
        spy_df = calculate_indicators(spy_df)
        
        df['ma_crossover_signal'] = np.where(df['sma_50'] > df['sma_200'], 1, -1)
        df['rsi_signal'] = np.select([df['rsi'] < 30, df['rsi'] > 70], [1, -1], default=0)
        df['bb_signal'] = np.select([df['Close'] < df['bb_lower'], df['Close'] > df['bb_upper']], [1, -1], default=0)
        df['macd_signal'] = np.where(df['macd'] > df['macd_signal_line'], 1, -1)
        
        spy_regime = pd.Series(np.where(spy_df['Close'] > spy_df['sma_200'], 1, -1), index=spy_df.index)
        df['market_regime'] = spy_regime.reindex(df.index, method='ffill')
        
        df.dropna(inplace=True)
        return df.iloc[-1] if not df.empty else None
    except Exception as e:
        print(f"  MAJOR ERROR in get_live_features for {ticker}: {e}")
        return None

def find_target_option(trading_client, data_client, ticker, prediction):
    """Finds a suitable options contract with defensive checks."""
    try:
        quote = data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=ticker))
        if not quote or ticker not in quote or not hasattr(quote[ticker], 'ask_price') or not quote[ticker].ask_price:
            return None
        current_price = quote[ticker].ask_price
        
        today = datetime.now().date()
        min_exp = today + timedelta(days=config.DAYS_TO_EXPIRATION_MIN)
        max_exp = today + timedelta(days=config.DAYS_TO_EXPIRATION_MAX)
        chain = trading_client.get_option_chain(ticker, expiration_date_gte=min_exp, expiration_date_lte=max_exp)
        
        if not chain or not chain.get(ticker): return None
        contracts = chain[ticker]
        strikes = sorted(list(set([c.strike_price for c in contracts])))
        if not strikes: return None
        
        closest_strike = min(strikes, key=lambda x: abs(x - current_price))
        strike_idx = strikes.index(closest_strike)
        option_type = 'call' if prediction == 1 else 'put'
        target_idx = strike_idx + config.STRIKE_PRICE_OFFSET if prediction == 1 else strike_idx - config.STRIKE_PRICE_OFFSET
        
        if not (0 <= target_idx < len(strikes)): return None
        target_strike = strikes[target_idx]

        for c in contracts:
            if c.strike_price == target_strike and c.type == option_type:
                return c.symbol
        return None
    except Exception as e:
        print(f"  Error finding option for {ticker}: {e}")
        return None

def run_trader(stop_event):
    """Main bot loop, accepts a stop_event for graceful shutdown."""
    print("\n--- Starting Ultimate AI Options Trading Bot ---")
    
    if not load_model_from_r2(): return
        
    trading_client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)
    data_client = StockHistoricalDataClient(config.API_KEY, config.SECRET_KEY)
    
    while not stop_event.is_set():
        try:
            print(f"\n[{time.ctime()}] Running AI Options trading cycle...")
            positions = trading_client.get_all_positions()
            
            # (Position management logic can be added here)

            for ticker in config.TICKERS:
                if stop_event.is_set(): break
                try:
                    print(f"\n- Analyzing {ticker} for new entry")
                    features = get_live_features(data_client, ticker)
                    if features is None:
                        continue
                    
                    if not all(col in features.index for col in config.FEATURE_COLUMNS):
                        continue

                    feature_df = pd.DataFrame([features])[config.FEATURE_COLUMNS]
                    prediction = ultimate_model.predict(feature_df.values)[0]
                    
                    print(f"  Ultimate AI Prediction for {ticker}: {'UP (Buy Call)' if prediction == 1 else 'DOWN/STAY (Hold)'}")

                    if prediction == 1:
                        contract = find_target_option(trading_client, data_client, ticker, prediction)
                        if contract:
                            order = MarketOrderRequest(symbol=contract, qty=1, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                            trading_client.submit_order(order)
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

