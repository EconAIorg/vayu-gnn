import pandas as pd
import numpy as np
from feature_engine.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import KNNImputer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import RobustScaler
# from panelsplit.pipeline import SequentialCVPipeline

import numpy as np
import pandas as pd
from feature_engine.imputation.base_imputer import BaseImputer

class ThresholdFilter(BaseImputer):
    """
    A Feature-engine transformer that sets values above specified thresholds to NaN.
    
    Parameters:
    ----------
    thresholds : dict
        Dictionary of column names and their maximum valid values.
    
    Attributes:
    ----------
    variables_ : list
        The list of variables to be transformed.
    """

    def __init__(self, thresholds: dict):
        if not isinstance(thresholds, dict):
            raise ValueError("`thresholds` must be a dictionary mapping column names to max values.")
        self.thresholds = thresholds

    def fit(self, X: pd.DataFrame, y=None):
        """
        The fit method does nothing, as this transformer does not learn any parameters.
        """
        self.variables_ = [col for col in self.thresholds if col in X.columns]
        return self

    def transform(self, X: pd.DataFrame):
        """
        Apply threshold filtering: set values greater than their respective max threshold to NaN.
        """
        X = X.copy()
        for col in self.variables_:
            X.loc[X[col] > self.thresholds[col], col] = np.nan
        return X

thresholds = {
    "pm_25": 1000,    # μg/m³ – Values above 1000 μg/m³ are extremely unusual even in polluted conditions.
    "pm_10": 2000,    # μg/m³ – Values above 2000 μg/m³ are likely due to sensor errors.
    "no2": 400,       # μg/m³ – Although urban NO₂ can spike, readings above 400 μg/m³ are suspect.
    "co": 10,         # μg/m³ – Given that the sensor averages ~1.2 μg/m³, values above 10 μg/m³ likely indicate faults.
    "co2": 2000,      # ppm   – Outdoor CO₂ typically hovers below 1000 ppm; >2000 ppm is likely an error.
    "ch4": 5       # ppb   – Ambient CH₄ is normally around 1800–2000 ppb; above 5000 ppb is highly unusual.
}