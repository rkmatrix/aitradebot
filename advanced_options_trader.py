import pandas as pd
import numpy as np
import sqlite3
import config

def calculate_indicators(df):
    """
    Calculates all necessary technical indicators manually without pandas-ta.
    """
    # Moving Averages
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    df['sma_200'] = df['Close'].rolling(window=200).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['bb_ma'] = df['Close'].rolling(window=20).mean()
    df['bb_std'] = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_ma'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_ma'] - (df['bb_std'] * 2)

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    return df

def generate_signals(df, spy_df):
    """Generates strategy signals from the calculated indicators."""
    print("  Generating strategy signals...")
    df = calculate_indicators(df)
    spy_df = calculate_indicators(spy_df)

    # Generate signals based on the new columns
    df['ma_crossover_signal'] = np.where(df['sma_50'] > df['sma_200'], 1, -1)
    df['rsi_signal'] = np.select([df['rsi'] < 30, df['rsi'] > 70], [1, -1], default=0)
    df['bb_signal'] = np.select([df['Close'] < df['bb_lower'], df['Close'] > df['bb_upper']], [1, -1], default=0)
    df['macd_signal'] = np.where(df['macd'] > df['macd_signal_line'], 1, -1)
    
    # Market Regime
    # Ensure spy_df index is aligned with df's for proper market regime assignment
    spy_regime = pd.Series(np.where(spy_df['Close'] > spy_df['sma_200'], 1, -1), index=spy_df.index)
    df['market_regime'] = spy_regime.reindex(df.index, method='ffill')
    
    return df

def create_target(df):
    """Creates the target variable for the model."""
    print("  Creating target variable...")
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df

def run_feature_engineering():
    """Main function to run the feature engineering process."""
    print("\n--- Starting Advanced Feature Engineering (No pandas-ta) ---")
    conn = sqlite3.connect(config.DB_FILE)
    
    try:
        spy_df = pd.read_sql(f"SELECT * FROM {config.MARKET_REGIME_TICKER}_ohlcv", conn, index_col='date', parse_dates=['date'])
        # Rename columns for consistency
        spy_df.columns = [col.capitalize() for col in spy_df.columns]
    except Exception as e:
        print(f"FATAL: Could not load SPY data for market regime analysis. Error: {e}")
        conn.close()
        return

    for ticker in config.TICKERS:
        try:
            print(f"\nProcessing ticker: {ticker}")
            df = pd.read_sql(f"SELECT * FROM {ticker}_ohlcv", conn, index_col='date', parse_dates=['date'])
            # Rename columns for consistency
            df.columns = [col.capitalize() for col in df.columns]

            if df.empty:
                print(f"  No data found for {ticker}. Skipping.")
                continue

            df = generate_signals(df, spy_df)
            df = create_target(df)
            
            # Select only the final feature columns and the target
            final_df = df[config.FEATURE_COLUMNS + [config.TARGET_COLUMN]]
            final_df.dropna(inplace=True)

            output_filename = f"{ticker}_advanced_features.csv"
            final_df.to_csv(output_filename)
            print(f"  Successfully saved advanced features to '{output_filename}'")

        except Exception as e:
            print(f"  An error occurred while processing {ticker}: {e}")
            
    conn.close()
    print("\n--- Advanced Feature Engineering Complete ---")

if __name__ == '__main__':
    run_feature_engineering()

