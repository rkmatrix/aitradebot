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

# --- Configuration ---
API_KEY = os.getenv('ALPACA_API_KEY', 'PKYVIZ6Y0GGUCQHGI97V')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '2vHaDIDPzfbtgn5aNVm5P3LEuhVzcRpd4CChOinS')
MODEL_FILENAME = "trading_model.pkl"
FEATURES_FILENAME = "feature_columns.pkl"

# Options Strategy Parameters
DAYS_TO_EXPIRATION_MIN = 30
DAYS_TO_EXPIRATION_MAX = 45
STRIKE_PRICE_OFFSET = 3 # How many strike prices away from the current price to select
POSITION_CLOSE_DTE = 5 # Close position when DTE is 5 days or less

TICKERS = ['AAPL', 'TSLA', 'AMZN', 'NVDA'] # Options are best for volatile, liquid stocks

# --- Clients and Models ---
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
try:
    trading_model = joblib.load(MODEL_FILENAME)
    feature_columns = joblib.load(FEATURES_FILENAME)
    print(f"Successfully loaded model and {len(feature_columns)} feature columns.")
except FileNotFoundError:
    print(f"Error: Model or feature file not found. Please run 'ml_model_trainer.py' first.")
    exit()


def get_stock_market_data(ticker, timeframe, limit=250):
    """Fetches historical stock data."""
    try:
        request_params = StockBarsRequest(symbol_or_symbols=[ticker], timeframe=timeframe, limit=limit)
        bars = data_client.get_stock_bars(request_params)
        df = bars.df.reset_index()
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()

def create_features_for_prediction(df):
    """Calculates all necessary features for the ML model."""
    df.ta.sma(length=50, append=True); df.ta.sma(length=200, append=True)
    df.ta.ema(length=50, append=True); df.ta.ema(length=200, append=True)
    df.ta.macd(append=True); df.ta.rsi(append=True)
    df.ta.bbands(append=True); df.ta.atr(append=True)
    df.ta.vwap(append=True); df.ta.obv(append=True)
    df.columns = [col.lower().replace('.', '_').replace('-', '_').replace(':', '') for col in df.columns]
    df.dropna(inplace=True)
    return df

def find_target_option(ticker, prediction):
    """Finds a suitable options contract based on the AI's prediction."""
    # 1. Get current stock price
    quote = data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=ticker))
    current_price = quote[ticker].ask_price

    # 2. Define date range for expiration
    today = datetime.now().date()
    min_exp_date = today + timedelta(days=DAYS_TO_EXPIRATION_MIN)
    max_exp_date = today + timedelta(days=DAYS_TO_EXPIRATION_MAX)

    # 3. Fetch the option chain
    option_chain = trading_client.get_option_chain(
        symbol_or_symbols=ticker,
        expiration_date_gte=min_exp_date,
        expiration_date_lte=max_exp_date
    )
    
    if not option_chain or not option_chain.get(ticker):
        print(f"No suitable options found for {ticker} in the desired date range.")
        return None

    chain_data = option_chain[ticker]

    # 4. Determine target strike price
    strikes = sorted(list(set([c.strike_price for c in chain_data])))
    closest_strike = min(strikes, key=lambda x:abs(x-current_price))
    strike_index = strikes.index(closest_strike)

    if prediction == 1: # Bullish (UP) -> Buy a Call
        target_strike_index = strike_index + STRIKE_PRICE_OFFSET
        option_type = 'call'
    else: # Bearish (DOWN) -> Buy a Put
        target_strike_index = strike_index - STRIKE_PRICE_OFFSET
        option_type = 'put'

    if not (0 <= target_strike_index < len(strikes)):
        print("Could not find a suitable strike price offset.")
        return None
    target_strike = strikes[target_strike_index]

    # 5. Find the contract that matches our criteria
    for contract in chain_data:
        if contract.strike_price == target_strike and contract.type == option_type:
            print(f"Found target option: {contract.symbol} (Strike: {target_strike}, Type: {option_type})")
            return contract.symbol
    return None

def trading_bot_cycle():
    """Runs one cycle of the AI options trading bot logic."""
    print(f"\n[{time.ctime()}] Running AI Options trading cycle...")
    
    # Check existing positions first for exit signals
    positions = trading_client.get_all_positions()
    open_option_symbols = [p.symbol for p in positions if p.asset_class == 'us_option']

    for symbol in open_option_symbols:
        underlying = trading_client.get_asset(symbol).underlying_symbol
        print(f"\n--- Managing existing position in {symbol} (Underlying: {underlying}) ---")
        
        # Exit reason 1: Prediction flips
        data_df = get_stock_market_data(underlying, TimeFrame.Day)
        indicator_df = create_features_for_prediction(data_df)
        features = indicator_df.iloc[-1][feature_columns]
        prediction = trading_model.predict([features])[0]
        
        is_call = 'C' in symbol.split('_')[1]

        if (is_call and prediction == 0) or (not is_call and prediction == 1):
            print(f"AI signal for {underlying} flipped. Closing position in {symbol}.")
            trading_client.close_position(symbol)
            continue

        # Exit reason 2: Too close to expiration
        expiration_date = datetime.strptime(trading_client.get_asset(symbol).expiration_date, '%Y-%m-%d').date()
        if (expiration_date - datetime.now().date()).days <= POSITION_CLOSE_DTE:
            print(f"Position in {symbol} is too close to expiration. Closing.")
            trading_client.close_position(symbol)

    # Look for new entries
    for ticker in TICKERS:
        print(f"\n--- Analyzing {ticker} for new entry ---")
        
        positions = trading_client.get_all_positions() # Refresh positions
        has_position = any(p.asset_class == 'us_option' and trading_client.get_asset(p.symbol).underlying_symbol == ticker for p in positions)
        if has_position:
            print(f"Already have a position for underlying {ticker}. Skipping new entry analysis.")
            continue

        data_df = get_stock_market_data(ticker, TimeFrame.Day)
        if data_df.empty: continue
        indicator_df = create_features_for_prediction(data_df)
        if indicator_df.empty or not all(elem in indicator_df.columns for elem in feature_columns): continue
            
        features = indicator_df.iloc[-1][feature_columns]
        prediction = trading_model.predict([features])[0]
        
        print(f"AI Prediction for {ticker}: {'UP (Call)' if prediction == 1 else 'DOWN (Put)'}")

        target_contract_symbol = find_target_option(ticker, prediction)
        
        if target_contract_symbol:
            print(f"Placing order to BUY 1 contract of {target_contract_symbol}")
            order_data = MarketOrderRequest(
                symbol=target_contract_symbol,
                qty=1, # Buy 1 contract
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            trading_client.submit_order(order_data=order_data)

if __name__ == '__main__':
    if 'YOUR_API_KEY' in API_KEY or 'YOUR_SECRET_KEY' in SECRET_KEY:
        print("\n!!! WARNING: Alpaca API keys are not configured. Please edit the script. !!!\n")
    else:
        while True:
            try:
                trading_bot_cycle()
                print("\nCycle complete. Waiting for 15 minutes...")
                time.sleep(60 * 15)
            except KeyboardInterrupt:
                print("\nBot stopped by user. Exiting.")
                break
            except Exception as e:
                print(f"An error occurred in the main loop: {e}")
                time.sleep(60) # Wait a minute on error

