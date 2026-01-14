# custom_transformers.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class BehavioralFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, feature_columns=None):
        """
        Initialize the extractor.
        Replace or adjust parameters based on your original class.
        """
        self.feature_columns = feature_columns  # Example param

    def fit(self, X, y=None):
        # Usually no fitting needed for simple feature engineering
        return self

    def transform(self, X):
        """
        Add your custom feature engineering logic here.
        X is expected to be a pandas DataFrame.
        Return a numpy array or DataFrame with the same number of rows.
        """
        X = X.copy()  # Avoid modifying original

        # === Paste your original feature engineering code here ===
        # Example (replace with your actual logic):
        X['hour_of_day'] = pd.to_datetime(X['timestamp']).dt.hour
        X['is_rush_hour'] = X['hour_of_day'].isin([7, 8, 17, 18]).astype(int)
        X['day_of_week'] = pd.to_datetime(X['timestamp']).dt.dayofweek

        # Select final features (adjust to match what your model expects)
        if self.feature_columns:
            return X[self.feature_columns].values
        else:
            return X.values