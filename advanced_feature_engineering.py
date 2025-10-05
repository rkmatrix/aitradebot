import pandas as pd
import pandas_ta as ta
import sqlite3
import config

def generate_signals(df, spy_df):
    """
    Generates high-level strategy signals and market regime data.
    """
    print("  Generating strategy signals...")
    
    # Strategy 1: Moving Average Crossover
    df['ma_crossover_signal'] = (ta.sma(df['close'], length=50) > ta.sma(df['close'], length=200)).astype(int).replace(0, -1)

    # Strategy 2: RSI Mean Reversion
    rsi = ta.rsi(df['close'])
    df['rsi_signal'] = 0
    df.loc[rsi < 30, 'rsi_signal'] = 1
    df.loc[rsi > 70, 'rsi_signal'] = -1

    # Strategy 3: Bollinger Bands Breakout/Reversal
    bbands = ta.bbands(df['close'], length=20, std=2.0)
    # Access columns by position to be robust against naming changes.
    if bbands is not None and not bbands.empty and len(bbands.columns) >= 3:
        lower_band_col = bbands.columns[0] # e.g., 'BBL_20_2.0'
        upper_band_col = bbands.columns[2] # e.g., 'BBU_20_2.0'
        df['bb_signal'] = 0
        df.loc[df['close'] < bbands[lower_band_col], 'bb_signal'] = 1
        df.loc[df['close'] > bbands[upper_band_col], 'bb_signal'] = -1
    else:
        df['bb_signal'] = 0 # Default to neutral if bbands can't be calculated

    # Strategy 4: MACD Signal Cross
    macd = ta.macd(df['close'])
    if macd is not None and 'MACD_12_26_9' in macd.columns and 'MACDs_12_26_9' in macd.columns:
        df['macd_signal'] = (macd['MACD_12_26_9'] > macd['MACDs_12_26_9']).astype(int).replace(0, -1)
    else:
        df['macd_signal'] = 0 # Default to neutral if MACD calculation fails

    # Market Regime Detection
    spy_df['sma_200'] = ta.sma(spy_df['close'], length=200)
    # CRITICAL FIX: Use a standard conditional expression for single values.
    # This prevents the "'numpy.int64' object has no attribute 'replace'" error.
    regime_value = 1 if spy_df['close'].iloc[-1] > spy_df['sma_200'].iloc[-1] else -1
    df['market_regime'] = regime_value
    
    df.dropna(inplace=True)
    return df

def create_target(df):
    """Creates the target variable for our machine learning model."""
    print("  Creating target variable...")
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)
    return df

def run_feature_engineering():
    """Main function to load data, create advanced features, and save to CSV."""
    print("\n--- Starting Advanced Feature Engineering ---")
    conn = sqlite3.connect(config.DB_FILE)
    
    try:
        spy_df = pd.read_sql(f"SELECT * FROM {config.MARKET_REGIME_TICKER}_ohlcv", conn, index_col='date', parse_dates=['date'])
    except Exception as e:
        print(f"Could not load market regime data for {config.MARKET_REGIME_TICKER}: {e}")
        conn.close()
        return

    for ticker in config.TICKERS:
        try:
            print(f"\nProcessing ticker: {ticker}")
            query = f"SELECT * FROM {ticker.upper()}_ohlcv"
            df = pd.read_sql(query, conn, index_col='date', parse_dates=['date'])
            if df.empty:
                print(f"  No data found for {ticker}. Skipping.")
                continue
                
            df = generate_signals(df, spy_df)
            df = create_target(df)
            
            output_filename = f"{ticker}_advanced_features.csv"
            df.to_csv(output_filename)
            print(f"  Successfully saved advanced features to '{output_filename}'")
            print(f"  Dataset shape: {df.shape}")
        except Exception as e:
            print(f"  An error occurred while processing {ticker}: {e}")
            
    conn.close()
    print("\n--- Advanced Feature Engineering Complete ---")

if __name__ == "__main__":
    run_feature_engineering()

