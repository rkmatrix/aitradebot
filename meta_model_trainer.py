import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import config
from stacking_model import StackingEnsemble
from xgboost import XGBClassifier
from sklearn.svm import SVC
import os

def load_data():
    """Loads and combines all feature CSVs into a single DataFrame."""
    all_dfs = []
    for ticker in config.TICKERS:
        filename = f"{ticker}_advanced_features.csv"
        try:
            # The setup process runs on the ephemeral disk, so we read from the current directory
            df = pd.read_csv(filename, index_col='date', parse_dates=True)
            all_dfs.append(df)
            print(f"Successfully loaded {filename}")
        except FileNotFoundError:
            print(f"Warning: {filename} not found. Skipping.")
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs)

def run_training():
    """Trains the model and saves it to the persistent disk path."""
    print("\n--- Starting Ultimate AI Model Training ---")
    
    dataset = load_data()
    if dataset.empty:
        print("No data found to train on. Exiting.")
        return

    dataset.dropna(inplace=True)
    
    X = dataset[config.FEATURE_COLUMNS]
    y = dataset[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # --- Define the "Committee of Experts" ---
    base_models = [
        ('random_forest', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('gradient_boosting', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)),
        ('svc', SVC(gamma='auto', probability=True, random_state=42))
    ]

    # The "Master AI Strategist" that learns from the experts
    meta_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # --- Create and Train the Ultimate Model ---
    stacking_model = StackingEnsemble(base_models=base_models, meta_model=meta_model)
    print("\n--- Fitting the Stacking Ensemble ---")
    stacking_model.fit(X_train, y_train)

    # --- Evaluate Performance ---
    print("\nEvaluating Ultimate Model performance on the test set...")
    predictions = stacking_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=['Down/Stay (0)', 'Up (1)'], zero_division=0)
    
    print(f"Ultimate Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(report)

    # --- Save the Model to the Persistent Disk ---
    # This is the only save operation needed.
    try:
        print(f"\nSaving the trained Ultimate Model to '{config.MODEL_FILENAME}'...")
        # Ensure the directory for the model exists on the persistent disk
        os.makedirs(os.path.dirname(config.MODEL_FILENAME), exist_ok=True)
        joblib.dump(stacking_model, config.MODEL_FILENAME)
        print("Ultimate Model saved successfully to persistent storage.")
    except Exception as e:
        print(f"FATAL: Failed to save the model to persistent storage: {e}")
        # This will now be the final error message if something is wrong with the disk.

    print("--- Ultimate AI Model Training Complete ---")

if __name__ == '__main__':
    run_training()
    

