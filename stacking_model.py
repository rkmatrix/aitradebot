from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import numpy as np

# This class defines the structure of our "Committee of Experts" AI model.
# By keeping it in a separate file, both the trainer and the live bot can use it
# without loading unnecessary, heavy libraries.

class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    A custom stacking ensemble model that combines predictions from multiple base models
    and uses a meta-model to make a final prediction.
    """
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        self.base_models_fitted = []

    def fit(self, X, y):
        """
        Fits the base models on the training data and then trains the meta-model
        on the predictions of the base models.
        """
        print("  Training base expert models...")
        # Train each base model and store it
        for name, model in self.base_models:
            print(f"    Training {name}...")
            model.fit(X, y)
            self.base_models_fitted.append((name, model))

        print("  Generating meta-features from base model predictions...")
        # Generate predictions from base models to be used as features for the meta-model
        meta_features = self._get_meta_features(X)

        print("  Training the master meta-model...")
        # Train the meta-model on the base model predictions
        self.meta_model.fit(meta_features, y)
        return self

    def predict(self, X):
        """
        Makes a prediction by first getting predictions from the base models
        and then feeding those predictions to the meta-model.
        """
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict(meta_features)

    def _get_meta_features(self, X):
        """
        Generates predictions from the fitted base models.
        """
        predictions = []
        for name, model in self.base_models_fitted:
            # The predictions need to be reshaped to be concatenated as columns
            predictions.append(model.predict(X).reshape(-1, 1))
        
        # Concatenate predictions horizontally to create the meta-feature set
        return np.hstack(predictions)

