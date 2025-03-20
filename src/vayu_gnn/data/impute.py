import pandas as pd
import numpy as np
from feature_engine.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import KNNImputer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import RobustScaler
# from panelsplit.pipeline import SequentialCVPipeline

class OutlierImputer(BaseEstimator, TransformerMixin):
    """
    Transformer to apply the IQR-based outlier imputation to each column of a DataFrame,
    storing the computed lower and upper bounds during fitting.
    """
    def __init__(self, multiplier=1.5, strategy=np.nan):
        self.multiplier = multiplier
        self.strategy = strategy
        self.bounds_ = {}

    def fit(self, X, y=None):
        """
        Compute and store the lower and upper bounds for each column based on IQR.
        """
        self.bounds_ = {
            col: {
                "lower": X[col].quantile(0.25) - self.multiplier * (X[col].quantile(0.75) - X[col].quantile(0.25)),
                "upper": X[col].quantile(0.75) + self.multiplier * (X[col].quantile(0.75) - X[col].quantile(0.25))
            }
            for col in X.select_dtypes(include=[np.number]).columns
        }
        return self

    def transform(self, X):
        """
        Apply the stored bounds to replace outliers with the specified strategy.
        """
        X_transformed = X.copy()
        for col, bounds in self.bounds_.items():
            lower_bound, upper_bound = bounds["lower"], bounds["upper"]
            mask = (X_transformed[col] < lower_bound) | (X_transformed[col] > upper_bound)
            X_transformed.loc[mask, col] = self.strategy
        return X_transformed

from sklearn.neighbors import NearestNeighbors

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.neighbors import NearestNeighbors
class SpatialKNNImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values using spatial KNN based on latitude and longitude with Haversine distance.
    Ensures neighbors have matching 'date' and 'hour'.

    Parameters
    ----------
    reference_data : pd.DataFrame
        DataFrame containing at least ['date', 'hour', 'latitude', 'longitude'] and columns for imputation.
        Rows with missing values are dropped from the reference data.
    k : int, default=5
        Number of neighbors to consider for imputation.
    weights : {'uniform', 'distance'}, default='distance'
        - 'uniform': All neighbors contribute equally.
        - 'distance': Inverse distance weighting.

    Notes
    -----
    - Uses Haversine distance in kilometers.
    - Ensures neighbors have the same 'date' and 'hour' as the missing row.
    - If no valid neighbors exist, the missing value remains NaN.
    - Skips columns in X that are not in reference_data.
    """

    def __init__(self, reference_data: pd.DataFrame, k=5, weights="distance"):
        self.k = k
        self.weights = weights

        # Ensure required columns exist
        required_cols = {"date", "hour", "latitude", "longitude"}
        if not required_cols.issubset(reference_data.columns):
            raise ValueError(f"Reference data must contain columns: {required_cols}")

        # Drop rows with missing values in reference data
        self.reference_data = reference_data.dropna().copy()

        # Convert lat/lon from degrees to radians for Haversine
        self.reference_data["lat_rad"] = np.radians(self.reference_data["latitude"])
        self.reference_data["lon_rad"] = np.radians(self.reference_data["longitude"])

        # Store nearest neighbor models per (date, hour)
        self.nn_models = {}
        grouped = self.reference_data.groupby(["date", "hour"])

        for (date, hour), group in grouped:
            coords = group[["lat_rad", "lon_rad"]].to_numpy()
            if len(group) >= self.k:  # Ensure enough neighbors exist
                nn_model = NearestNeighbors(n_neighbors=self.k, metric="haversine")
                nn_model.fit(coords)
                self.nn_models[(date, hour)] = (nn_model, group)

    def fit(self, X, y=None):
        # Nothing to fit; reference_data is used for lookups.
        return self

    def transform(self, X):
        X_transformed = X.copy()
        missing_idx = X[X.isna().any(axis=1)].index
        earth_radius_km = 6371  # Earth radius in kilometers

        for idx in missing_idx:
            row = X.loc[idx]

            # Extract date & hour
            date, hour = row["date"], row["hour"]

            if (date, hour) not in self.nn_models:
                continue  # No neighbors with matching date & hour, skip this row

            nn_model, neighbors = self.nn_models[(date, hour)]

            # Convert target coordinates to radians
            lat_r, lon_r = np.radians([row["latitude"], row["longitude"]])

            # Find nearest neighbors
            distances, indices = nn_model.kneighbors([[lat_r, lon_r]])
            distances_km = distances[0] * earth_radius_km  # Convert to km

            # Determine weights
            if self.weights == "uniform":
                w = np.ones_like(distances_km)
            else:
                distances_km = np.where(distances_km < 1e-5, 1e-5, distances_km)
                w = 1 / distances_km

            # Retrieve neighbor rows
            neighbor_subset = neighbors.iloc[indices[0]]

            # Impute missing columns that exist in reference_data
            valid_columns = set(X.columns).intersection(set(neighbors.columns))
            valid_columns -= {"date", "hour", "latitude", "longitude"}

            for col in valid_columns:
                if pd.isna(X.at[idx, col]):
                    vals = neighbor_subset[col].values
                    X_transformed.at[idx, col] = np.average(vals, weights=w)

        return X_transformed


scale_pipe = Pipeline([
    ('scaler', SklearnTransformerWrapper(RobustScaler()))
], verbose=True)