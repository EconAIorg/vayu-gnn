import pandas as pd
import numpy as np
from feature_engine.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import KNNImputer
from feature_engine.wrappers import SklearnTransformerWrapper

def impute_outliers_iqr(series: pd.Series, multiplier: float = 1.5, strategy=np.nan) -> pd.Series:
    """
    Identify and impute outliers in a Pandas Series based on the interquartile range (IQR).
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outliers = (series < lower_bound) | (series > upper_bound)

    return series.where(~outliers, strategy)

class OutlierImputer(BaseEstimator, TransformerMixin):
    """
    Transformer to apply the IQR-based outlier imputation to each column of a DataFrame.
    """
    def __init__(self, multiplier=1.5, strategy=np.nan):
        self.multiplier = multiplier
        self.strategy = strategy

    def fit(self, X, y=None):
        return self  # No fitting needed

    def transform(self, X):
        return X.apply(lambda col: impute_outliers_iqr(col, self.multiplier, self.strategy))

# Define the pipeline using Feature-engine's SklearnTransformerWrapper
impute_pipeline = Pipeline([
    ("outlier_imputer", OutlierImputer()),
    ("knn_imputer", SklearnTransformerWrapper(KNNImputer(n_neighbors=5, weights='distance')))
], verbose=True)