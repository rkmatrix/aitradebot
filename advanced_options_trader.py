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
# We need to import the StackingEnsemble class so joblib can load the custom model object
from meta_model_trainer import StackingEnsemble

# --- Options Strategy Parameters ---
DAYS_TO_EXPIRATION_MIN = 30
DAYS_TO_EXPIRATION_MAX = 45
STRIKE_PRICE_OFFSET = 3 # How many strike prices away from the current price to select
POSITION_CLOSE_DTE = 5 # Close position when Days To Expiration is 5 or less

# --- Clients ---
# Initialize clients at the module level
trading_client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(config.API_KEY, config.SECRET_KEY)


def get_stock_market_data(ticker, limit=250):
    """Fetches historical stock data from Alpaca."""
    try:
        request_params = StockBarsRequest(symbol_or_symbols=[ticker], timeframe=TimeFrame.Day, limit=limit)
        bars = data_client.get_stock_bars(request_params).df
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.reset_index(level=0, drop=True)
        bars.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        return bars
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()

def get_live_features(ticker):
    """Fetches live data and computes the advanced features the model was trained on."""
    df = get_stock_market_data(ticker)
    spy_df = get_stock_market_data(config.MARKET_REGIME_TICKER)
    if df.empty or spy_df.empty:
        return None

    df['ma_crossover_signal'] = (ta.sma(df['Close'], 50) > ta.sma(df['Close'], 200)).astype(int).replace(0, -1)
    rsi = ta.rsi(df['Close'])
    df['rsi_signal'] = 0; df.loc[rsi < 30, 'rsi_signal'] = 1; df.loc[rsi > 70, 'rsi_signal'] = -1
    bbands = ta.bbands(df['Close'])
    df['bb_signal'] = 0; df.loc[df['Close'] < bbands[f'BBL_20_2.0'], 'bb_signal'] = 1; df.loc[df['Close'] > bbands[f'BBU_20_2.0'], 'bb_signal'] = -1
    macd = ta.macd(df['Close'])
    df['macd_signal'] = (macd['MACD_12_26_9'] > macd['MACDs_12_26_9']).astype(int).replace(0, -1)
    
    spy_df['sma_200'] = ta.sma(spy_df['Close'], 200)
    df['market_regime'] = (spy_df['Close'].iloc[-1] > spy_df['sma_200'].iloc[-1]).astype(int).replace(0, -1)
    
    df.dropna(inplace=True)
    if df.empty:
        return None
    return df.iloc[-1]

def find_target_option(ticker, prediction):
    """Finds a suitable options contract based on the AI's prediction."""
    try:
        quote = data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=ticker))
        current_price = quote[ticker].ask_price
        today = datetime.now().date()
        min_exp = today + timedelta(days=DAYS_TO_EXPIRATION_MIN)
        max_exp = today + timedelta(days=DAYS_TO_EXPIRATION_MAX)

        chain = trading_client.get_option_chain(ticker, expiration_date_gte=min_exp, expiration_date_lte=max_exp)
        if not chain or not chain.get(ticker):
            print(f"  No options chain found for {ticker} in the desired date range.")
            return None
        
        contracts = chain[ticker]
        strikes = sorted(list(set([c.strike_price for c in contracts])))
        closest_strike = min(strikes, key=lambda x: abs(x - current_price))
        strike_idx = strikes.index(closest_strike)

        option_type = 'call' if prediction == 1 else 'put'
        target_idx = strike_idx + STRIKE_PRICE_OFFSET if prediction == 1 else strike_idx - STRIKE_PRICE_OFFSET
        if not (0 <= target_idx < len(strikes)):
            print(f"  Could not find a suitable strike price offset.")
            return None
        target_strike = strikes[target_idx]

        for c in contracts:
            if c.strike_price == target_strike and c.type == option_type:
                print(f"  Found target option: {c.symbol}")
                return c.symbol
        return None
    except Exception as e:
        print(f"  An error occurred while finding an option for {ticker}: {e}")
        return None

def run_trader():
    """Main function to run the live options trading bot loop."""
    print("\n--- Starting Ultimate AI Options Trading Bot ---")

    # --- Load the AI Model ---
    # Moved here from the top level to prevent errors during the setup process.
    try:
        ultimate_model = joblib.load(config.MODEL_FILENAME)
        print(f"Successfully loaded Ultimate AI Model from '{config.MODEL_FILENAME}'.")
    except FileNotFoundError:
        print(f"Error: Model file '{config.MODEL_FILENAME}' not found. Please run 'main.py --action setup' first.")
        return # Use return instead of exit for cleaner modular code

    while True:
        try:
            print(f"\n[{time.ctime()}] Running AI Options trading cycle...")
            positions = trading_client.get_all_positions()

            for pos in [p for p in positions if p.asset_class == 'us_option']:
                underlying_symbol = trading_client.get_asset(pos.symbol).underlying_symbol
                expiration_date = datetime.strptime(trading_client.get_asset(pos.symbol).expiration_date, '%Y-%m-%d').date()
                if (expiration_date - datetime.now().date()).days <= POSITION_CLOSE_DTE:
                    print(f"- Position in {pos.symbol} is too close to expiration. Closing.")
                    trading_client.close_position(pos.symbol)

            positions = trading_client.get_all_positions()
            for ticker in config.TICKERS:
                print(f"\n- Analyzing {ticker} for new entry")
                if any(trading_client.get_asset(p.symbol).underlying_symbol == ticker for p in positions if p.asset_class == 'us_option'):
                    print(f"  Already have a position for {ticker}. Skipping.")
                    continue

                features = get_live_features(ticker)
                if features is None:
                    print(f"  Could not generate live features for {ticker}. Skipping.")
                    continue

                prediction = ultimate_model.predict(pd.DataFrame([features]))[0]
                print(f"  Ultimate AI Prediction for {ticker}: {'UP (Buy Call)' if prediction == 1 else 'DOWN/STAY (Hold)'}")

                if prediction == 1:
                    contract = find_target_option(ticker, prediction)
                    if contract:
                        print(f"  Placing order to BUY 1 contract of {contract}")
                        order = MarketOrderRequest(symbol=contract, qty=1, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                        trading_client.submit_order(order)

            print("\nCycle complete. Waiting for 15 minutes...")
            time.sleep(60 * 15)
        except KeyboardInterrupt:
            print("\nBot stopped by user.")
            break
        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
            time.sleep(60)

if __name__ == '__main__':
    run_trader()

