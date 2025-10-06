import os
import time
import pandas as pd
import numpy as np
import threading
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

# --- Options Strategy Parameters ---
ultimate_model = None

def load_model():
    """Loads the AI model into the global scope."""
    global ultimate_model
    if ultimate_model is None:
        try:
            ultimate_model = joblib.load(config.MODEL_FILENAME)
            print("Successfully loaded Ultimate AI Model.")
            return True
        except FileNotFoundError:
            print(f"Error: Model file '{config.MODEL_FILENAME}' not found.")
            return False
    return True

def get_stock_market_data(data_client, ticker, limit=250):
    """Fetches historical stock data from Alpaca."""
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

def calculate_indicators_live(df):
    """Calculates all necessary technical indicators manually without pandas-ta."""
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
    return df

def get_live_features(data_client, ticker):
    """Computes advanced features for live data with robust error handling."""
    try:
        df = get_stock_market_data(data_client, ticker)
        spy_df = get_stock_market_data(data_client, config.MARKET_REGIME_TICKER)
        if df.empty or spy_df.empty or len(df) < 201 or len(spy_df) < 201:
            return None

        df = calculate_indicators_live(df)
        spy_df = calculate_indicators_live(spy_df)

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

def find_target_option(trading_client, data_client, ticker):
    """Finds a suitable options contract."""
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

        target_idx = strike_idx + config.STRIKE_PRICE_OFFSET
        if not (0 <= target_idx < len(strikes)): return None
        target_strike = strikes[target_idx]

        for c in contracts:
            if c.strike_price == target_strike and c.type == 'call':
                print(f"  Found target option: {c.symbol}")
                return c.symbol
        return None
    except Exception as e:
        print(f"  Error finding option for {ticker}: {e}")
        return None

def run_trader(stop_event):
    """Main bot loop."""
    print("\n--- Starting Ultimate AI Options Trading Bot (No pandas-ta) ---")
    
    if not load_model(): return
        
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
                        print(f"  Could not generate live features. Skipping.")
                        continue
                    
                    if not all(col in features.index for col in config.FEATURE_COLUMNS):
                        print(f"  Feature set is incomplete. Skipping.")
                        continue

                    feature_df = pd.DataFrame([features])[config.FEATURE_COLUMNS]
                    prediction = ultimate_model.predict(feature_df)[0]
                    
                    print(f"  Prediction: {'UP (Buy Call)' if prediction == 1 else 'DOWN/STAY (Hold)'}")

                    if prediction == 1:
                        contract = find_target_option(trading_client, data_client, ticker)
                        if contract:
                            order = MarketOrderRequest(symbol=contract, qty=1, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                            trading_client.submit_order(order)
                except Exception as e:
                    print(f"  Unhandled error for {ticker}: {e}")

            if stop_event.is_set(): break
            print("\nCycle complete. Waiting for 15 minutes...")
            for _ in range(15 * 60):
                if stop_event.is_set(): break
                time.sleep(1)

        except Exception as e:
            print(f"A critical error in main loop: {e}")
            time.sleep(60)
    
    print("--- AI Bot shutting down. ---")

if __name__ == '__main__':
    run_trader(threading.Event())

