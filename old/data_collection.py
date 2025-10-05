import yfinance as yf
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import os

# --- Configuration ---
TICKERS = ['AAPL', 'ORCL', 'TSLA', 'SPY', 'AMZN', 'NVDA']
START_DATE = "2019-01-01"
END_DATE = pd.to_datetime('today').strftime('%Y-%m-%d')
DB_FILE = "trading_data.db"

def create_database_connection():
    """ Creates a SQLAlchemy engine for the SQLite database. """
    return create_engine(f'sqlite:///{DB_FILE}')

def fetch_data(ticker, start, end):
    """ Fetches historical stock data from Yahoo Finance. """
    print(f"Fetching data for {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, interval="1d")
    if df.empty:
        print(f"No data found for {ticker}. Check the ticker symbol and date range.")
        return None
    df.reset_index(inplace=True)
    
    # Robustly clean column names, handling potential tuples from yfinance
    cleaned_columns = []
    for col in df.columns:
        # If a column name is a tuple (e.g., ('Adj Close', '')), take the first part
        name = col[0] if isinstance(col, tuple) else col
        cleaned_columns.append(name.lower().replace(' ', '_').replace('-', '_'))
    df.columns = cleaned_columns

    print("Data fetched successfully.")
    return df

def store_data(df, table_name, engine):
    """ Stores a DataFrame in the SQLite database. """
    if df is None or df.empty:
        print("Dataframe is empty. Nothing to store.")
        return
    print(f"Storing data in table '{table_name}'...")
    try:
        # Use 'replace' to overwrite the table if it exists, ensuring fresh data
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print("Data stored successfully.")
    except Exception as e:
        print(f"An error occurred while storing data: {e}")

def main():
    """ Main function to run the data collection and storage process for multiple tickers. """
    engine = create_database_connection()
    for ticker in TICKERS:
        print(f"\n----- Processing Ticker: {ticker} -----")
        # Sanitize ticker for table name (e.g., BRK.B -> BRK_B)
        table_name = f"{ticker.replace('.', '_')}_ohlcv"
        data = fetch_data(ticker, START_DATE, END_DATE)
        store_data(data, table_name, engine)
    
    print(f"\nProcess complete. Data for all tickers is stored in '{DB_FILE}'.")


if __name__ == "__main__":
    main()

