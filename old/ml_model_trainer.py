import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib # For saving the trained model

# --- Configuration ---
TICKERS = ['AAPL', 'ORCL', 'TSLA', 'SPY', 'AMZN', 'NVDA']
# NOTE: The static FEATURE_COLUMNS list has been removed.
# The script will now dynamically determine the features from the loaded data.
TARGET_COLUMN = 'target'
TEST_SIZE = 0.2 # Use 20% of the data for testing
MODEL_FILENAME = "trading_model.pkl"

def load_and_combine_data(tickers):
    """Loads feature data for all tickers and combines them into one DataFrame."""
    all_dfs = []
    for ticker in tickers:
        filename = f"{ticker}_features.csv"
        try:
            df = pd.read_csv(filename, index_col='date', parse_dates=True)
            df['ticker'] = ticker
            all_dfs.append(df)
            print(f"Successfully loaded {filename}, shape: {df.shape}")
        except FileNotFoundError:
            print(f"Warning: {filename} not found. Skipping.")
    
    if not all_dfs:
        print("Error: No data files found. Cannot proceed with training.")
        return pd.DataFrame()
        
    return pd.concat(all_dfs, axis=0)

def main():
    """Main function to train, evaluate, and save the trading model."""
    print("--- Starting ML Model Training ---")
    
    # 1. Load and prepare the data
    full_dataset = load_and_combine_data(TICKERS)
    if full_dataset.empty:
        return
        
    # Standardize column names to handle any format (e.g., '.', '_')
    full_dataset.columns = [col.lower().replace('.', '_').replace('-', '_') for col in full_dataset.columns]

    # Clean up any potential infinite values
    full_dataset.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
    full_dataset.dropna(inplace=True)
    
    # 2. Define Features (X) and Target (y)
    # CRITICAL FIX: Dynamically determine feature columns
    # This removes dependency on a hardcoded list and prevents KeyErrors.
    feature_columns = [col for col in full_dataset.columns if col not in [TARGET_COLUMN, 'ticker']]
    print(f"\nDynamically identified {len(feature_columns)} feature columns.")

    X = full_dataset[feature_columns]
    y = full_dataset[TARGET_COLUMN]
    
    if len(X) == 0:
        print("Error: No data available for training after cleaning. Exiting.")
        return

    # 3. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False
    )
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # 4. Train the Random Forest model
    print("\nTraining the RandomForestClassifier model...")
    model = RandomForestClassifier(
        n_estimators=200,     # More trees can improve performance
        min_samples_split=10, # Prevent overfitting
        random_state=42,      # For reproducibility
        n_jobs=-1             # Use all available CPU cores
    )
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 5. Evaluate the model's performance
    print("\nEvaluating model performance on the test set...")
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=['Down (0)', 'Up (1)'])
    
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(report)

    # 6. Save the trained model to a file
    print(f"\nSaving the trained model to '{MODEL_FILENAME}'...")
    joblib.dump(model, MODEL_FILENAME)
    # Also save the list of columns the model was trained on
    joblib.dump(feature_columns, "feature_columns.pkl")
    print("Model and feature columns saved successfully.")
    print("---------------------------------")


if __name__ == "__main__":
    main()

