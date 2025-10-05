import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import config

def load_and_combine_data():
    """Loads advanced feature data for all tickers and combines them."""
    all_dfs = []
    for ticker in config.TICKERS:
        filename = f"{ticker}_advanced_features.csv"
        try:
            df = pd.read_csv(filename, index_col='date', parse_dates=True)
            df['ticker'] = ticker
            all_dfs.append(df)
            print(f"Successfully loaded {filename}")
        except FileNotFoundError:
            print(f"Warning: {filename} not found. Skipping.")
    if not all_dfs: return pd.DataFrame()
    return pd.concat(all_dfs, axis=0)

class StackingEnsemble:
    """
    A custom class to manage our "Committee of Experts" model.
    This class orchestrates the training of multiple base models and a final meta-model.
    """
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        self.feature_columns = []

    def fit(self, X, y):
        self.feature_columns = X.columns.tolist()
        print("\n--- Fitting the Stacking Ensemble ---")
        print("Training base expert models...")
        for name, model in self.base_models.items():
            print(f"  Training {name}...")
            model.fit(X, y)
        
        print("Generating meta-features from base model predictions...")
        # Create a new dataset for the meta-model, where features are the predictions of base models
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models.items()):
            # Use predict_proba to get the probability of the positive class
            meta_features[:, i] = model.predict_proba(X)[:, 1]
            
        print("Training the master meta-model...")
        self.meta_model.fit(meta_features, y)
        print("--- Stacking Ensemble Fit Complete ---")
        return self

    def predict(self, X):
        # To make a prediction, first get predictions from all base models
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models.items()):
            meta_features[:, i] = model.predict_proba(X[self.feature_columns])[:, 1]
        
        # Then, use the meta-model to make the final prediction based on the base models' outputs
        return self.meta_model.predict(meta_features)

def run_model_training():
    """Main function to train, evaluate, and save the ultimate ensemble model."""
    print("\n--- Starting Ultimate AI Model Training ---")
    
    full_dataset = load_and_combine_data()
    if full_dataset.empty: 
        print("No data found to train on. Exiting.")
        return

    # These are the high-level signals our AI will learn from
    feature_columns = ['ma_crossover_signal', 'rsi_signal', 'bb_signal', 'macd_signal', 'market_regime']
    full_dataset.dropna(subset=feature_columns + ['target'], inplace=True)
    
    X = full_dataset[feature_columns]
    y = full_dataset['target']
    
    # Split chronologically for time-series data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Define the "Committee of Experts" (base models)
    base_models = {
        'random_forest': RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=150, random_state=42),
        'svc': SVC(probability=True, random_state=42) # Needs probability=True for stacking
    }
    
    # Define the "Master AI Strategist" (meta-model)
    meta_model = LogisticRegression()

    ensemble_model = StackingEnsemble(base_models=base_models, meta_model=meta_model)
    ensemble_model.fit(X_train, y_train)

    print("\nEvaluating Ultimate Model performance on the test set...")
    predictions = ensemble_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=['Down/Stay (0)', 'Up (1)'])
    
    print(f"Ultimate Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(report)

    print(f"\nSaving the trained Ultimate Model to '{config.MODEL_FILENAME}'...")
    joblib.dump(ensemble_model, config.MODEL_FILENAME)
    print("Ultimate Model saved successfully.")
    print("--- Ultimate AI Model Training Complete ---")

if __name__ == "__main__":
    run_model_training()

