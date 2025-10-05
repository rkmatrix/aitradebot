import os
import time
import pandas as pd
import pandas_ta as ta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TrailingStopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import joblib # To load our trained ML model
import math

# --- Configuration ---
# IMPORTANT: Replace with your actual Alpaca paper trading keys
API_KEY = os.getenv('ALPACA_API_KEY', 'PKYVIZ6Y0GGUCQHGI97V')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '2vHaDIDPzfbtgn5aNVm5P3LEuhVzcRpd4CChOinS')
MODEL_FILENAME = "trading_model.pkl"
FEATURES_FILENAME = "feature_columns.pkl"

# Risk Management Parameters
RISK_PER_TRADE = 0.02  # We will risk 2% of our equity on any single trade

TICKERS = ['AAPL', 'ORCL', 'TSLA', 'SPY', 'AMZN', 'NVDA']

# --- Clients and Models ---
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Load the trained machine learning model and feature columns
try:
    trading_model = joblib.load(MODEL_FILENAME)
    feature_columns = joblib.load(FEATURES_FILENAME)
    print(f"Successfully loaded model and {len(feature_columns)} feature columns.")
except FileNotFoundError:
    print(f"Error: Model or feature file not found. Please run 'ml_model_trainer.py' first.")
    exit()


def get_market_data(ticker, timeframe, limit=250):
    """Fetches historical market data and prepares it for feature calculation."""
    try:
        request_params = StockBarsRequest(symbol_or_symbols=[ticker], timeframe=timeframe, limit=limit)
        bars = data_client.get_stock_bars(request_params)
        df = bars.df.reset_index()
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching market data for {ticker}: {e}")
        return pd.DataFrame()

def create_features_for_prediction(df):
    """Calculates all necessary features for the ML model."""
    # This function must create all the same features as the training script
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.macd(append=True)
    df.ta.rsi(append=True)
    df.ta.bbands(append=True)
    df.ta.atr(append=True)
    df.ta.vwap(append=True)
    df.ta.obv(append=True)
    df.columns = [col.lower().replace('.', '_').replace('-', '_').replace(':', '') for col in df.columns]
    df.dropna(inplace=True)
    return df

def check_portfolio_and_log():
    """Logs current portfolio status."""
    print("\n--- Portfolio Status ---")
    account = trading_client.get_account()
    print(f"Equity: ${float(account.equity):,.2f} | Cash: ${float(account.cash):,.2f}")
    positions = trading_client.get_all_positions()
    if not positions: print("No open positions.")
    else:
        print("Open Positions:")
        for pos in positions:
            print(f"  - {pos.symbol}: {pos.qty} shares @ avg ${float(pos.avg_entry_price):,.2f} | P/L: ${float(pos.unrealized_pl):,.2f}")
    print("--------------------------")

def trading_bot_cycle():
    """Runs one cycle of the AI trading bot logic for all tickers."""
    print(f"\n[{time.ctime()}] Running AI trading cycle...")
    account_info = trading_client.get_account()
    
    for ticker in TICKERS:
        print(f"\n--- Analyzing {ticker} ---")
        
        data_df = get_market_data(ticker, TimeFrame.Day)
        if data_df.empty:
            print(f"No data for {ticker}. Skipping.")
            continue
            
        indicator_df = create_features_for_prediction(data_df)
        if indicator_df.empty or not all(elem in indicator_df.columns for elem in feature_columns):
            print(f"Not enough data or missing columns to create features for {ticker}. Skipping.")
            continue
            
        latest_data = indicator_df.iloc[-1]
        
        # Prepare the features for the model in the correct order
        features_for_model = latest_data[feature_columns]

        # --- Get AI Prediction ---
        prediction = trading_model.predict([features_for_model])[0] # Get a single prediction (0 or 1)
        
        in_position = False
        try:
            position = trading_client.get_open_position(ticker)
            in_position = True
        except Exception:
            pass

        print(f"Latest Close: ${latest_data['close']:.2f} | AI Prediction: {'UP (Buy)' if prediction == 1 else 'DOWN (Sell)'} | In Position: {in_position}")
        
        # --- Execute Orders based on AI Prediction ---
        if prediction == 1 and not in_position: # AI predicts UP, and we have no position
            equity = float(account_info.equity)
            amount_to_risk = equity * RISK_PER_TRADE
            
            # Use ATR for a dynamic stop loss distance
            stop_loss_distance = latest_data['atrr_14'] * 2.0 # Example: 2x ATR
            
            if stop_loss_distance <= 0: continue
            
            qty = math.floor(amount_to_risk / stop_loss_distance)
            if qty <= 0: continue

            print(f"AI BUY SIGNAL. Qty: {qty}, Stop Distance: ${stop_loss_distance:.2f}")
            # Use a trailing stop to lock in profits
            market_order_data = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC, # Good 'til Canceled
                order_class=OrderClass.TRAILING_STOP,
                trail_price=f"{stop_loss_distance:.2f}"
            )
            trading_client.submit_order(order_data=market_order_data)
            
        elif prediction == 0 and in_position: # AI predicts DOWN, and we have a position
            print(f"AI SELL SIGNAL DETECTED for {ticker}. Closing position.")
            trading_client.close_position(ticker)
        else:
            print("AI advises HOLD. No action taken.")

if __name__ == '__main__':
    if 'YOUR_API_KEY' in API_KEY or 'YOUR_SECRET_KEY' in SECRET_KEY:
        print("\n!!! WARNING: Alpaca API keys are not configured. Please edit the script. !!!\n")
    else:
        check_portfolio_and_log()
        while True:
            try:
                trading_bot_cycle()
                check_portfolio_and_log()
                print("\nCycle complete. Waiting for 15 minutes...")
                time.sleep(60 * 15)
            except KeyboardInterrupt:
                print("\nBot stopped by user. Exiting.")
                break
            except Exception as e:
                print(f"An error occurred in the main loop: {e}")
                print("Restarting cycle in 5 minutes...")
                time.sleep(60 * 5)

