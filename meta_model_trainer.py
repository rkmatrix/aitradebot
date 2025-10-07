import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import config
from stacking_model import StackingEnsemble
from xgboost import XGBClassifier
from sklearn.svm import SVC
import boto3
import io
import os

def upload_model_to_r2(model_object, bucket, key):
    """Serializes a model object and uploads it to Cloudflare R2."""
    try:
        # Check if credentials are set
        if not all([config.R2_ENDPOINT_URL, config.R2_BUCKET_NAME, config.R2_ACCESS_KEY_ID, config.R2_SECRET_ACCESS_KEY]):
            print("FATAL: Cloudflare R2 environment variables are not fully set. Cannot upload model.")
            return False

        print(f"Uploading model to R2 bucket '{bucket}' with key '{key}'...")
        # Serialize model to an in-memory buffer
        with io.BytesIO() as buffer:
            joblib.dump(model_object, buffer)
            buffer.seek(0)
            
            s3_client = boto3.client(
                's3',
                endpoint_url=config.R2_ENDPOINT_URL,
                aws_access_key_id=config.R2_ACCESS_KEY_ID,
                aws_secret_access_key=config.R2_SECRET_ACCESS_KEY,
                region_name="auto" # R2 is region-less
            )
            s3_client.upload_fileobj(buffer, bucket, key)
        print("Model uploaded successfully to R2.")
        return True
    except Exception as e:
        print(f"FATAL: Failed to upload model to R2. Error: {e}")
        return False

def load_data():
    """Loads and combines all advanced feature CSVs into a single DataFrame."""
    all_dfs = []
    print("Loading feature data for all tickers...")
    for ticker in config.TICKERS:
        filename = f"{ticker}_advanced_features.csv"
        try:
            df = pd.read_csv(filename, index_col='date', parse_dates=True)
            all_dfs.append(df)
            print(f"  Successfully loaded {filename}")
        except FileNotFoundError:
            print(f"  Warning: {filename} not found. Skipping.")
    
    if not all_dfs:
        print("FATAL: No data files found. Cannot proceed with training.")
        return pd.DataFrame()
        
    return pd.concat(all_dfs)

def run_training():
    """Main function to train, evaluate, and save the Ultimate AI Model."""
    print("\n--- Starting Ultimate AI Model Training ---")
    
    dataset = load_data()
    if dataset.empty:
        return

    dataset.dropna(inplace=True)
    
    X = dataset[config.FEATURE_COLUMNS]
    y = dataset[config.TARGET_COLUMN]

    if X.empty:
        print("FATAL: No data available for training after cleaning. Exiting.")
        return

    # Split data chronologically for time-series validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # --- Define the "Committee of AI Experts" ---
    base_models = [
        ('random_forest', RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)),
        ('gradient_boosting', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)),
        ('svc', SVC(gamma='auto', probability=True, random_state=42))
    ]

    # The "Master AI Strategist" that learns from the experts
    meta_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # --- Create and Train the Ultimate Model ---
    stacking_model = StackingEnsemble(base_models=base_models, meta_model=meta_model)
    
    print("\n--- Fitting the Stacking Ensemble (this may take several minutes)... ---")
    # Use .values to pass numpy arrays for compatibility
    stacking_model.fit(X_train.values, y_train.values)

    # --- Evaluate Performance ---
    print("\n--- Evaluating Ultimate Model performance on the test set... ---")
    predictions = stacking_model.predict(X_test.values)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=['Down/Stay (0)', 'Up (1)'], zero_division=0)
    
    print(f"Ultimate Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(report)

    # --- Upload the final model to Cloudflare R2 ---
    if not upload_model_to_r2(stacking_model, config.R2_BUCKET_NAME, config.MODEL_FILENAME):
        print("Setup failed because the model could not be uploaded to cloud storage.")
        return

    print("\n--- Ultimate AI Model Training Complete ---")

if __name__ == '__main__':
    run_training()

