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
import boto3
import io

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

def upload_model_to_r2(model_object):
    """Serializes the model and uploads it to a Cloudflare R2 bucket."""
    # Check for necessary R2 configuration
    if not all([config.R2_ENDPOINT_URL, config.R2_BUCKET_NAME, config.R2_ACCESS_KEY_ID, config.R2_SECRET_ACCESS_KEY]):
        print("\nFATAL: Cloudflare R2 environment variables are not fully set. Cannot upload model.")
        print("Setup failed because the model could not be uploaded to cloud storage.")
        return False

    try:
        print(f"\nUploading model '{config.MODEL_FILENAME}' to R2 bucket '{config.R2_BUCKET_NAME}'...")
        s3_client = boto3.client(
            's3',
            endpoint_url=config.R2_ENDPOINT_URL,
            aws_access_key_id=config.R2_ACCESS_KEY_ID,
            aws_secret_access_key=config.R2_SECRET_ACCESS_KEY,
            region_name="auto" # Required for Cloudflare R2
        )
        
        # Serialize the model to an in-memory buffer
        with io.BytesIO() as buffer:
            joblib.dump(model_object, buffer)
            buffer.seek(0) # Rewind the buffer to the beginning
            
            # Upload the buffer content to R2
            s3_client.upload_fileobj(buffer, config.R2_BUCKET_NAME, config.MODEL_FILENAME)
        
        print("Ultimate Model uploaded successfully to cloud storage.")
        return True
    except Exception as e:
        print(f"FATAL: Failed to upload the model to R2 cloud storage: {e}")
        return False


def run_training():
    """Trains the model and then uploads it to cloud storage."""
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

    # The "Master AI Strategist"
    meta_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # --- Create and Train the Ultimate Model ---
    stacking_model = StackingEnsemble(base_models=base_models, meta_model=meta_model)
    print("\n--- Fitting the Stacking Ensemble ---")
    stacking_model.fit(X_train.values, y_train.values) # Use .values to avoid potential column name mismatches

    # --- Evaluate Performance ---
    print("\nEvaluating Ultimate Model performance on the test set...")
    predictions = stacking_model.predict(X_test.values)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=['Down/Stay (0)', 'Up (1)'], zero_division=0)
    
    print(f"Ultimate Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(report)

    # --- Upload the Model to Cloud Storage ---
    upload_model_to_r2(stacking_model)

    print("--- Ultimate AI Model Training Complete ---")

if __name__ == '__main__':
    run_training()

