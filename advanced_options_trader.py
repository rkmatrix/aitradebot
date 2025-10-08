import os
import time
import pandas as pd
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
# --- NEW imports for options market data ---
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest, OptionChainRequest
from alpaca.data.timeframe import TimeFrame
import joblib
from datetime import datetime, timedelta
import config
from stacking_model import StackingEnsemble
import boto3
import io
import threading
import sys

# --- Options Strategy Parameters ---
DAYS_TO_EXPIRATION_MIN = 30
DAYS_TO_EXPIRATION_MAX = 45
STRIKE_PRICE_OFFSET = 3
POSITION_CLOSE_DTE = 5

ultimate_model = None
trading_client = None
data_client = None
# --- NEW: option historical data client ---
option_data_client = None

def load_model():
    """
    Loads the model, trying the cloud first, or falling back to local file.
    """
    global ultimate_model
    if ultimate_model is not None: return True

    if config.ENVIRONMENT == "cloud":
        if not all([config.R2_ENDPOINT_URL, config.R2_BUCKET_NAME, config.R2_ACCESS_KEY_ID, config.R2_SECRET_ACCESS_KEY]):
            print("FATAL: R2 environment variables not set for cloud mode. Cannot download model.")
            return False
        try:
            print(f"Attempting to download model '{config.MODEL_FILENAME}' from R2 bucket...")
            s3 = boto3.client('s3', endpoint_url=config.R2_ENDPOINT_URL, aws_access_key_id=config.R2_ACCESS_KEY_ID, aws_secret_access_key=config.R2_SECRET_ACCESS_KEY, region_name="auto")
            with io.BytesIO() as buffer:
                s3.download_fileobj(config.R2_BUCKET_NAME, config.MODEL_FILENAME, buffer)
                buffer.seek(0)
                ultimate_model = joblib.load(buffer)
            print("Successfully loaded Ultimate AI Model from cloud storage.")
            return True
        except Exception as e:
            print(f"FATAL: Could not load model from R2: {e}. The bot cannot run.")
            return False
    else: # Local environment
        try:
            print(f"Attempting to load model from local path: '{config.MODEL_FILENAME}'")
            ultimate_model = joblib.load(config.MODEL_FILENAME)
            print("Successfully loaded Ultimate AI Model from local file.")
            return True
        except FileNotFoundError:
            print("="*50)
            print(f"CRITICAL ERROR: Model file not found locally at '{config.MODEL_FILENAME}'.")
            print("Please run the 'Full Setup' to train and save the model.")
            print("="*50)
            return False
        except Exception as e:
            print(f"An unexpected error occurred while loading the local model: {e}")
            return False

def initialize_clients():
    """
    Initializes Alpaca clients inside a protected function for stability.
    """
    global trading_client, data_client, option_data_client
    try:
        trading_client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)
        data_client = StockHistoricalDataClient(config.API_KEY, config.SECRET_KEY)
        # --- NEW: initialize option historical data client ---
        option_data_client = OptionHistoricalDataClient(config.API_KEY, config.SECRET_KEY)

        # Test the keys by getting account details
        account = trading_client.get_account()

        # Force a non-critical API call to warm up the data connection
        trading_client.get_clock() 

        print(f"\n--- CLIENTS READY ---")
        print(f"Account Status: {account.status}")
        print(f"Equity: ${float(account.equity):,.2f}")
        print("---")
        return True
    except Exception as e:
        print("="*50)
        print("FATAL CLIENT INIT ERROR: Alpaca client failed to initialize.")
        print(f"Check API Keys in config.py. Error: {type(e).__name__} - {e}")
        print("="*50)
        return False

# ... keep get_stock_market_data, calculate_live_indicators, get_live_features unchanged ...

def find_target_option(trading_client, data_client, ticker):
    """Finds a suitable options contract with defensive checks."""
    global option_data_client
    try:
        # Get a live quote (same as before)
        quote_req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
        quote = data_client.get_stock_latest_quote(quote_req)

        if not quote or ticker not in quote or not hasattr(quote[ticker], 'ask_price') or not quote[ticker].ask_price: 
            print(f"  Could not get a live quote price for {ticker}.")
            return None

        current_price = quote[ticker].ask_price

        today = datetime.now().date()
        min_exp = today + timedelta(days=DAYS_TO_EXPIRATION_MIN)
        max_exp = today + timedelta(days=DAYS_TO_EXPIRATION_MAX)

        # --- FIX: Use OptionHistoricalDataClient with OptionChainRequest (correct for options data) ---
        if option_data_client is None:
            print("  Option data client not initialized. Cannot fetch option chain.")
            return None

        req = OptionChainRequest(
            underlying_symbol=ticker,
            expiration_date_gte=min_exp.strftime('%Y-%m-%d'),
            expiration_date_lte=max_exp.strftime('%Y-%m-%d'),
        )

        chain_resp = option_data_client.get_option_chain(req)

        # The response format can vary by SDK version - handle common cases
        contracts = None
        if isinstance(chain_resp, dict):
            # docs/examples sometimes return a dict keyed by underlying symbol
            contracts = chain_resp.get(ticker) or chain_resp.get('contracts') or chain_resp.get('data')
        else:
            # might be a list-like or object with attribute
            try:
                contracts = chain_resp.contracts  # some wrappers
            except Exception:
                contracts = chain_resp

        if not contracts:
            print(f"  No option contracts returned for {ticker} between {min_exp} and {max_exp}.")
            return None

        # Normalize to list of objects
        if hasattr(contracts, 'items') and not isinstance(contracts, list):
            # sometimes it's a dict of symbol->contract; flatten it
            possible = []
            for v in contracts.values():
                if isinstance(v, list):
                    possible.extend(v)
                else:
                    possible.append(v)
            contracts = possible

        # Extract strikes
        strikes = sorted(list(set([getattr(c, 'strike_price', None) for c in contracts if getattr(c, 'strike_price', None) is not None])))
        if not strikes:
            print("  No strikes extracted from option contracts.")
            return None

        closest_strike = min(strikes, key=lambda x: abs(x - current_price))
        strike_idx = strikes.index(closest_strike)

        # choose OTM call (one strike above)
        option_type = 'call'
        target_idx = strike_idx + STRIKE_PRICE_OFFSET if STRIKE_PRICE_OFFSET else strike_idx + 1

        if not (0 <= target_idx < len(strikes)):
            print("  Target strike index out of bounds for the strikes list.")
            return None
        target_strike = strikes[target_idx]

        final_symbol = None
        for contract in contracts:
            # contract.type might be an enum or string
            c_type = getattr(contract, 'type', None)
            c_strike = getattr(contract, 'strike_price', None)
            c_symbol = getattr(contract, 'symbol', None) or getattr(contract, 'contract_symbol', None)
            # Accept 'call' or OptionType.CALL-like enums
            if c_strike == target_strike and (str(c_type).lower().endswith('call') or str(c_type).lower() == 'call'):
                final_symbol = c_symbol
                break

        if final_symbol:
            print(f"  Found target option: {final_symbol} @ Strike {target_strike}")
            return final_symbol
        else:
            print(f"  Could not find specific option symbol for strike {target_strike}.")
            return None

    except Exception as e:
        print(f"  An error occurred while finding an option for {ticker}: {e}")
        return None

def run_trader(stop_event):
    """Main bot loop, using manual indicator calculations."""
    print("\n--- Starting Ultimate AI Options Trading Bot ---")

    if not load_model(): return

    if not initialize_clients():
        return # If clients fail, gracefully exit the thread

    while not stop_event.is_set():
        try:
            print(f"\n[{time.ctime()}] Running AI Options trading cycle...")
            for ticker in config.TICKERS:
                if stop_event.is_set(): break
                try:
                    print(f"\n- Analyzing {ticker} for new entry")

                    # 1. Get Prediction from Historical Data (Features)
                    features = get_live_features(data_client, ticker)

                    if features is None:
                        # Log if historical data (up to yesterday) is missing
                        print(f"  Skipping prediction for {ticker} due to missing data.")
                        continue

                    if not all(col in features.index for col in config.FEATURE_COLUMNS): continue

                    feature_df = pd.DataFrame([features])[config.FEATURE_COLUMNS]
                    prediction = ultimate_model.predict(feature_df.values)[0]

                    print(f"  Ultimate AI Prediction for {ticker}: {'UP (Buy Call)' if prediction == 1 else 'DOWN/STAY (Hold)'}")

                    if prediction == 1:
                        # 2. Find and Execute Trade
                        contract = find_target_option(trading_client, data_client, ticker)
                        if contract:
                            print(f"  Placing order to BUY 1 contract of {contract}")
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
