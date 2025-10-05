import pandas as pd
import pandas_ta as ta
import sqlite3

# --- Configuration ---
DB_FILE = "trading_data.db"
TICKERS = ['AAPL', 'ORCL', 'TSLA', 'SPY', 'AMZN', 'NVDA']

def verify_db(conn):
    """Checks the database for available tables and prints them."""
    print("\n--- Verifying Database ---")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    if not tables:
        print("No tables found in the database.")
    else:
        print("Tables found in database:")
        for table in tables:
            print(f"  - {table[0]}")
    print("--------------------------\n")

def create_features(df):
    """
    Generates a rich set of technical indicators for the given DataFrame.
    """
    print("  Calculating technical indicators...")
    
    # Manually calculate indicators to ensure compatibility
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
    
    # CRITICAL FIX: Clean up ALL column names to be standardized
    # This replaces special characters like '.' with '_' to prevent errors
    df.columns = [col.lower().replace('.', '_').replace('-', '_').replace(':', '') for col in df.columns]
    
    df.dropna(inplace=True)
    return df

def create_target(df):
    """
    Creates the target variable for our machine learning model.
    """
    print("  Creating target variable...")
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)
    return df

def main():
    """
    Main function to load data, create features and target, and save to CSV.
    """
    print("Starting feature engineering process...")
    conn = sqlite3.connect(DB_FILE)
    
    verify_db(conn)
    
    for ticker in TICKERS:
        try:
            print(f"\nProcessing ticker: {ticker}")
            
            # Use the correct table name format
            query = f"SELECT * FROM {ticker}_ohlcv"
            df = pd.read_sql(query, conn, index_col='date', parse_dates=['date'])
            
            if df.empty:
                print(f"  No data found for {ticker}. Skipping.")
                continue
                
            df = create_features(df)
            df = create_target(df)
            
            output_filename = f"{ticker}_features.csv"
            df.to_csv(output_filename)
            print(f"  Successfully saved features and target to '{output_filename}'")
            print(f"  Dataset shape: {df.shape}")
            
        except Exception as e:
            print(f"  An error occurred while processing {ticker}: {e}")
            
    conn.close()
    print("\nFeature engineering process completed for all tickers.")


if __name__ == "__main__":
    main()

