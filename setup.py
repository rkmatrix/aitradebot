import os
import sqlite3
import pandas as pd
import numpy as np
import time
import sys
import joblib
import boto3
import io
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from stacking_model import StackingEnsemble
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import config

# --- UTILITY FUNCTIONS ---
def clean_old_files():
    """Deletes old data and model files to ensure a fresh run."""
    print("\n--- Cleaning up old files ---")
    files_to_delete = [config.DB_FILE] + [f"{t}_advanced_features.csv" for t in config.TICKERS]
    if config.ENVIRONMENT == "local":
        files_to_delete.append(config.MODEL_FILENAME)
        
    for f in files_to_delete:
        if os.path.exists(f):
            os.remove(f)
            print(f"  Removed old file: {f}")

# --- STAGE 1: DATA COLLECTION ---
def fetch_data(ticker, start_date, end_date, data_client):
    """Fetches historical data from the Alpaca Market Data API."""
    print(f"  Fetching data for {ticker} from {start_date} to {end_date} via Alpaca...")
    request_params = StockBarsRequest(symbol_or_symbols=[ticker], timeframe=TimeFrame.Day, start=start_date, end=end_date)
    retries = 3
    for i in range(retries):
        try:
            bars = data_client.get_stock_bars(request_params).df
            if not bars.empty:
                if isinstance(bars.index, pd.MultiIndex): bars = bars.reset_index(level='symbol', drop=True)
                bars.columns = [col.lower().replace(' ', '_') for col in bars.columns]
                bars.dropna(inplace=True)
                print(f"    Successfully fetched {len(bars)} rows for {ticker}.")
                return bars
        except Exception as e:
            print(f"    Attempt {i+1}/{retries} for {ticker} failed: {e}")
            if i < retries - 1: time.sleep(2 * (i + 1))
    print(f"  All attempts to fetch data for {ticker} failed.")
    return pd.DataFrame()

def run_collection(data_client):
    """Orchestrates the data collection stage."""
    print("\n--- STAGE 1: DATA COLLECTION ---")
    conn = sqlite3.connect(config.DB_FILE)
    START_DATE, END_DATE = "2018-01-01", pd.to_datetime('today').strftime('%Y-%m-%d')
    tickers_to_fetch = config.TICKERS + [config.MARKET_REGIME_TICKER]
    
    for ticker in set(tickers_to_fetch):
        data = fetch_data(ticker, START_DATE, END_DATE, data_client)
        if data.empty:
            print(f"  FATAL: Could not fetch data for critical ticker {ticker}. Halting.")
            conn.close()
            return False
        data.index.name = 'date'
        data.to_sql(f"{ticker.upper()}_ohlcv", conn, if_exists='replace', index=True)
    conn.close()
    return True

# --- STAGE 2: FEATURE ENGINEERING ---
def calculate_indicators(df):
    """Calculates all necessary technical indicators manually."""
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    df['sma_200'] = df['Close'].rolling(window=200).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    bb_ma = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['bb_upper'] = bb_ma + (bb_std * 2)
    df['bb_lower'] = bb_ma - (bb_std * 2)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
    return df

def run_feature_engineering(conn):
    """Orchestrates the feature engineering stage."""
    print("\n--- STAGE 2: FEATURE ENGINEERING ---")
    try:
        spy_df = pd.read_sql(f"SELECT * FROM {config.MARKET_REGIME_TICKER}_ohlcv", conn, index_col='date', parse_dates=['date'])
        spy_df.columns = [col.capitalize() for col in spy_df.columns]
    except Exception as e:
        print(f"FATAL: Could not load SPY data. Error: {e}")
        return False

    for ticker in config.TICKERS:
        try:
            df = pd.read_sql(f"SELECT * FROM {ticker}_ohlcv", conn, index_col='date', parse_dates=['date'])
            df.columns = [col.capitalize() for col in df.columns]

            df = calculate_indicators(df.copy())
            spy_df_calc = calculate_indicators(spy_df.copy())

            df['ma_crossover_signal'] = np.where(df['sma_50'] > df['sma_200'], 1, -1)
            df['rsi_signal'] = np.select([df['rsi'] < 30, df['rsi'] > 70], [1, -1], default=0)
            df['bb_signal'] = np.select([df['Close'] < df['bb_lower'], df['Close'] > df['bb_upper']], [1, -1], default=0)
            df['macd_signal'] = np.where(df['macd'] > df['macd_signal_line'], 1, -1)
            spy_regime = pd.Series(np.where(spy_df_calc['Close'] > spy_df_calc['sma_200'], 1, -1), index=spy_df_calc.index)
            df['market_regime'] = spy_regime.reindex(df.index, method='ffill')
            df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            
            final_df = df[config.FEATURE_COLUMNS + [config.TARGET_COLUMN]].dropna()
            final_df.to_csv(f"{ticker}_advanced_features.csv")
            print(f"  Successfully processed and saved features for {ticker}")
        except Exception as e:
            print(f"  FATAL: An error occurred while processing {ticker}: {e}")
            return False
    return True

# --- STAGE 3: MODEL TRAINING ---
def save_model(model_object):
    """Saves the model either locally or to the cloud based on config."""
    if config.ENVIRONMENT == "cloud":
        if not all([config.R2_ENDPOINT_URL, config.R2_BUCKET_NAME, config.R2_ACCESS_KEY_ID, config.R2_SECRET_ACCESS_KEY]):
            print("\nFATAL: Cloudflare R2 environment variables not set. Cannot upload model.")
            return False
        try:
            print(f"\nUploading model '{config.MODEL_FILENAME}' to R2 bucket...")
            s3 = boto3.client('s3', endpoint_url=config.R2_ENDPOINT_URL, aws_access_key_id=config.R2_ACCESS_KEY_ID, aws_secret_access_key=config.R2_SECRET_ACCESS_KEY, region_name="auto")
            with io.BytesIO() as buffer:
                joblib.dump(model_object, buffer)
                buffer.seek(0)
                s3.upload_fileobj(buffer, config.R2_BUCKET_NAME, config.MODEL_FILENAME)
            print("Model uploaded successfully to cloud storage.")
        except Exception as e:
            print(f"FATAL: Failed to upload model to R2: {e}")
            return False
    else: # Local environment
        try:
            print(f"\nSaving model locally to '{config.MODEL_FILENAME}'...")
            joblib.dump(model_object, config.MODEL_FILENAME)
            print("Model saved successfully locally.")
        except Exception as e:
            print(f"FATAL: Failed to save model locally: {e}")
            return False
    return True

def run_training():
    """Orchestrates the model training stage."""
    print("\n--- STAGE 3: MODEL TRAINING ---")
    all_dfs = []
    for ticker in config.TICKERS:
        try:
            df = pd.read_csv(f"{ticker}_advanced_features.csv", index_col='date', parse_dates=True)
            all_dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: Feature file for {ticker} not found. Skipping.")
    if not all_dfs:
        print("FATAL: No feature data found to train on.")
        return False
    
    dataset = pd.concat(all_dfs).dropna()
    X, y = dataset[config.FEATURE_COLUMNS], dataset[config.TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    base_models = [('rf', RandomForestClassifier(n_estimators=100, random_state=42)), ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)), ('svc', SVC(probability=True, random_state=42))]
    meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
    stacking_model = StackingEnsemble(base_models=base_models, meta_model=meta_model)
    
    print("\n--- Fitting the Stacking Ensemble ---")
    stacking_model.fit(X_train.values, y_train.values)
    
    print("\n--- Evaluating Model ---")
    predictions = stacking_model.predict(X_test.values)
    print(classification_report(y_test, predictions, target_names=['Down/Stay (0)', 'Up (1)'], zero_division=0))
    
    if not save_model(stacking_model):
        return False
        
    return True

# --- MAIN ORCHESTRATOR ---
def run_full_setup():
    """Executes the entire setup process as a single, transactional script."""
    clean_old_files()

    if not config.API_KEY or 'YOUR_ALPACA' in config.API_KEY:
        print("\nFATAL: Alpaca API keys are not configured in config.py.")
        return False
        
    data_client = StockHistoricalDataClient(config.API_KEY, config.SECRET_KEY)
    
    if not run_collection(data_client):
        return False
    
    conn = sqlite3.connect(config.DB_FILE)
    if not run_feature_engineering(conn):
        conn.close()
        return False
    conn.close()
    
    if not run_training():
        return False
        
    print("\n\n--- ✅ FULL SETUP IS COMPLETE! You are now ready to trade. ---")
    return True

if __name__ == '__main__':
    if not run_full_setup():
        sys.exit(1) # Exit with an error code if setup fails

