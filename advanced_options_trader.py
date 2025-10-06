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
# CRITICAL FIX: Import from the new, lightweight model definition file
from stacking_model import StackingEnsemble

# --- Options Strategy Parameters ---
DAYS_TO_EXPIRATION_MIN = 30
DAYS_TO_EXPIRATION_MAX = 45
STRIKE_PRICE_OFFSET = 3 # How many strike prices away from the current price to select
POSITION_CLOSE_DTE = 5 # Close position when Days To Expiration is 5 or less

# This will be loaded by the run_trader function
ultimate_model = None

def load_model():
    """Helper function to load the model when the trader starts."""
    global ultimate_model
    if ultimate_model is None:
        try:
            ultimate_model = joblib.load(config.MODEL_FILENAME)
            print("Successfully loaded Ultimate AI Model.")
            return True
        except FileNotFoundError:
            print(f"Error: Model file '{config.MODEL_FILENAME}' not found. Please run 'main.py --action setup' first.")
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

def get_live_features(data_client, ticker):
    """Fetches live data and computes the advanced features the model was trained on."""
    try:
        df = get_stock_market_data(data_client, ticker)
        spy_df = get_stock_market_data(data_client, config.MARKET_REGIME_TICKER)
        if df.empty or spy_df.empty or len(df) < 200 or len(spy_df) < 200:
            print("  Not enough historical data to calculate all features.")
            return None

        # --- Generate Strategy Signals (More Robustly) ---
        df['ma_crossover_signal'] = (ta.sma(df['Close'], 50) > ta.sma(df['Close'], 200)).astype(int).replace(0, -1)
        
        rsi = ta.rsi(df['Close'])
        df['rsi_signal'] = 0
        if rsi is not None and not rsi.empty:
            df.loc[rsi < 30, 'rsi_signal'] = 1
            df.loc[rsi > 70, 'rsi_signal'] = -1
        
        bbands = ta.bbands(df['Close'])
        if bbands is not None and not bbands.empty:
            lower_band_col = next((col for col in bbands.columns if 'bbl' in col.lower()), None)
            upper_band_col = next((col for col in bbands.columns if 'bbu' in col.lower()), None)
            if lower_band_col and upper_band_col:
                df['bb_signal'] = 0
                df.loc[df['Close'] < bbands[lower_band_col], 'bb_signal'] = 1
                df.loc[df['Close'] > bbands[upper_band_col], 'bb_signal'] = -1
            else:
                df['bb_signal'] = 0
        else:
            df['bb_signal'] = 0
            
        macd = ta.macd(df['Close'])
        if macd is not None and not macd.empty:
            macd_line_col = next((col for col in macd.columns if 'macd_' in col.lower()), None)
            signal_line_col = next((col for col in macd.columns if 'macds' in col.lower()), None)
            if macd_line_col and signal_line_col:
                df['macd_signal'] = (macd[macd_line_col] > macd[signal_line_col]).astype(int).replace(0, -1)
            else:
                df['macd_signal'] = 0
        else:
            df['macd_signal'] = 0
            
        # --- Market Regime Detection ---
        spy_df['sma_200'] = ta.sma(spy_df['Close'], 200)
        last_spy_close = spy_df['Close'].iloc[-1] if not spy_df['Close'].empty else None
        last_spy_sma = spy_df['sma_200'].iloc[-1] if not spy_df['sma_200'].empty else None

        if pd.notna(last_spy_close) and pd.notna(last_spy_sma):
            df['market_regime'] = 1 if last_spy_close > last_spy_sma else -1
        else:
            df['market_regime'] = 0
        
        # Return the last row if it's not empty
        return df.iloc[-1] if not df.empty else None
    except Exception as e:
        print(f"  CRITICAL ERROR in get_live_features for {ticker}: {e}")
        return None


def find_target_option(trading_client, data_client, ticker, prediction):
    """Finds a suitable options contract based on the AI's prediction."""
    try:
        quote_req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
        quote = data_client.get_stock_latest_quote(quote_req)
        if not quote or ticker not in quote or not quote[ticker].ask_price:
            print(f"  Could not get a valid current price for {ticker}.")
            return None
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
        if not strikes: return None
        
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
    
    if not load_model():
        return
        
    trading_client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)
    data_client = StockHistoricalDataClient(config.API_KEY, config.SECRET_KEY)
    
    while True:
        try:
            print(f"\n[{time.ctime()}] Running AI Options trading cycle...")
            positions = trading_client.get_all_positions()
            
            # --- Manage Existing Positions ---
            for pos in [p for p in positions if p.asset_class == 'us_option']:
                try:
                    asset = trading_client.get_asset(pos.symbol)
                    if asset and asset.expiration_date:
                        expiration_date = datetime.strptime(asset.expiration_date, '%Y-%m-%d').date()
                        if (expiration_date - datetime.now().date()).days <= POSITION_CLOSE_DTE:
                            print(f"- Position in {pos.symbol} is too close to expiration. Closing.")
                            trading_client.close_position(pos.symbol)
                except Exception as e:
                    print(f"  Error managing position {pos.symbol}: {e}")


            # --- Look for New Entries ---
            positions = trading_client.get_all_positions() 
            for ticker in config.TICKERS:
                 print(f"\n- Analyzing {ticker} for new entry")
                 if any(hasattr(trading_client.get_asset(p.symbol), 'underlying_symbol') and trading_client.get_asset(p.symbol).underlying_symbol == ticker for p in positions if p.asset_class == 'us_option'):
                     print(f"  Already have a position for {ticker}. Skipping.")
                     continue

                 features = get_live_features(data_client, ticker)
                 if features is None or features.empty:
                     print(f"  Could not generate live features for {ticker}. Skipping.")
                     continue
                 
                 # Ensure all required columns are present before prediction
                 if not all(col in features.index for col in config.FEATURE_COLUMNS):
                     print(f"  Feature set for {ticker} is incomplete. Skipping prediction.")
                     continue

                 feature_df = pd.DataFrame([features])[config.FEATURE_COLUMNS]
                 prediction = ultimate_model.predict(feature_df)[0]
                 
                 print(f"  Ultimate AI Prediction for {ticker}: {'UP (Buy Call)' if prediction == 1 else 'DOWN/STAY (Hold)'}")

                 if prediction == 1: # Only trade on bullish signals
                     contract = find_target_option(trading_client, data_client, ticker, prediction)
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

