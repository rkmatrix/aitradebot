import yfinance as yf
import pandas as pd
import sqlite3
import config # Import settings from our config file

def fetch_data(ticker, start_date, end_date):
    """Fetches historical data from Yahoo Finance."""
    print(f"  Fetching data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    # CRITICAL FIX: Handle cases where column names are tuples
    cleaned_columns = []
    for col in df.columns:
        if isinstance(col, tuple):
            cleaned_columns.append(col[0].lower().replace(' ', '_'))
        else:
            cleaned_columns.append(col.lower().replace(' ', '_'))
    df.columns = cleaned_columns
    
    df.dropna(inplace=True)
    return df

def save_to_db(df, ticker, conn):
    """Saves a DataFrame to the SQLite database."""
    df.index.name = 'date'
    table_name = f"{ticker.upper()}_ohlcv"
    print(f"  Saving data to table '{table_name}'...")
    df.to_sql(table_name, conn, if_exists='replace', index=True)
    print(f"  Successfully saved {len(df)} rows.")

def run_collection():
    """Main function for the data collection process."""
    print("\n--- Starting Data Collection ---")
    conn = sqlite3.connect(config.DB_FILE)
    
    START_DATE = "2018-01-01"
    END_DATE = pd.to_datetime('today').strftime('%Y-%m-%d')
    
    for ticker in config.TICKERS:
        try:
            print(f"\nProcessing ticker: {ticker}")
            data = fetch_data(ticker, START_DATE, END_DATE)
            if not data.empty:
                save_to_db(data, ticker, conn)
            else:
                print(f"  No data fetched for {ticker}. Skipping.")
        except Exception as e:
            print(f"  An error occurred while processing {ticker}: {e}")
            
    conn.close()
    print("\n--- Data Collection Complete ---")

if __name__ == '__main__':
    run_collection()

