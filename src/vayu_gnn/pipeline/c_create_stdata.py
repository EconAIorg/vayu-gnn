from vayu_gnn.dbx import dbx_helper
from vayu_gnn.data import DataLoader
from vayu_gnn.data.impute import ThresholdFilter, thresholds
from tsl.data import SpatioTemporalDataset
from sklearn.impute import KNNImputer
from feature_engine.wrappers import SklearnTransformerWrapper
import pandas as pd
from feature_engine.pipeline import Pipeline
import numpy as np
from tsl.data.datamodule import (SpatioTemporalDataModule,
                                 TemporalSplitter)
from tsl.data.preprocessing import RobustScaler
from math import radians, sin, cos, sqrt, atan2
import torch
from tsl.data import WINDOW


class CreateSpatioTemporalDataset:
    def __init__(self, city : str, params:dict, dbx_helper = dbx_helper, loader = DataLoader):
        """
        Initialize the class for creating a spatio-temporal dataset.

        Parameters
        ----------
        city : str
            Name of the city to process.
        params : dict
            Configuration parameters including window size, fractions, etc.
        dbx_helper : object, optional
            Helper object for file operations, by default uses `vayu_gnn.dbx.dbx_helper`.
        loader : object, optional
            Data loader object, by default uses `vayu_gnn.data.DataLoader`.
        """
        self.city = city
        self.dbx_helper = dbx_helper
        self.loader = DataLoader(dbx_helper=dbx_helper)
        self.__dict__.update(params)

    def _impute(self):
        """
        Imputes missing values in pollution and sensor datasets using a KNN-based pipeline.

        Notes
        -----
        Uses a fraction of the data for fitting and applies transformations to the full dataset.
        Latitude and longitude information is temporarily added for imputation.
        """
        # Determine the number of days in the dataset
        min_date = self.loader.dfs['sensor_data']['date'].min()
        max_date = self.loader.dfs['sensor_data']['date'].max()
        total_days = (max_date - min_date).days

        # Compute the cutoff date for the fitting sample
        fit_days = int(total_days * (1 - (self.validation_frac + self.test_frac)))
        self.train_cutoff = min_date + pd.Timedelta(days=fit_days)

        # Assign latitude & longitude from node locations
        locs = self.loader.node_locations(city=self.city)
        self.loader.dfs['pollution'][['latitude', 'longitude']] = pd.DataFrame(self.loader.dfs['pollution'].node_id.map(locs).tolist())
        self.loader.dfs['sensor_data'][['latitude', 'longitude']] = pd.DataFrame(self.loader.dfs['sensor_data'].node_id.map(locs).tolist())

        ### --- Pollution Data Imputation --- ###
        vars_to_transform = [col for col in self.loader.dfs['pollution'].columns if col not in ["node_id", "date", "hour", "latitude", "longitude"]]

        pollution_fit_sample = self.loader.dfs['pollution'][self.loader.dfs['pollution']['date'] <= self.train_cutoff]

        pollution_impute_pipe = Pipeline([
            ('knn', SklearnTransformerWrapper(KNNImputer(weights='distance'), variables=vars_to_transform))
        ], verbose=True)

        pollution_impute_pipe.fit(pollution_fit_sample)
        self.loader.dfs['pollution'] = pollution_impute_pipe.transform(self.loader.dfs['pollution'])

        ### --- Sensor Data Imputation --- ###
        sensor_train = self.loader.dfs['sensor_data'][self.loader.dfs['sensor_data']['date'] <= self.train_cutoff]

        vars_to_transform = [col for col in self.loader.dfs['sensor_data'].columns if col not in ["node_id", "date", "hour", "latitude", "longitude"]]
        knn = SklearnTransformerWrapper(KNNImputer(weights='distance'), variables=vars_to_transform)

        sensor_impute_pipe = Pipeline([
            ('threshold_filter', ThresholdFilter(thresholds = thresholds)),
            ('knn', knn)
        ], verbose=True)

        sensor_impute_pipe.fit(sensor_train)
        self.loader.dfs['sensor_data'] = sensor_impute_pipe.transform(self.loader.dfs['sensor_data'])
        print(self.loader.dfs['sensor_data'].isna().sum())

        self.loader.dfs['sensor_data'].drop(columns=['latitude', 'longitude'], inplace=True)

    def create_single_df(self):
        """
        Loads, imputes, merges, and sorts multiple sources into a single DataFrame.

        Returns
        -------
        DataFrame
            Merged and cleaned DataFrame containing sensor and pollution data.
        """
        self.loader.multi_load(sources = self.sources, city= self.city)
        self._impute()
        df = self.loader.multi_merge(remove_dfs=True)
        df.sort_values(['date','hour', 'node_id'], inplace=True)

        self.dbx_helper.write_parquet(df, self.dbx_helper.clean_merged_input_path, self.city, 'merged.parquet')

        return df

    def gen_connectivity(self, distance_threshold_km=5.0):
        """
        Generates graph connectivity based on spatial proximity using Haversine distance.

        Parameters
        ----------
        distance_threshold_km : float, optional
            Distance threshold to define edges between nodes, by default 5.0

        Returns
        -------
        tuple of torch.Tensor
            A tuple containing edge indices and edge weights for the graph.
        """
        locs = self.loader.node_locations(self.city)

        node_names = list(sorted(locs.keys()))
        N = len(node_names)

        lats = np.array([locs[node]["lat"] for node in node_names])
        lons = np.array([locs[node]["long"] for node in node_names])

        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371.0  # Earth's radius in kilometers
            lat1_rad, lon1_rad = radians(lat1), radians(lon1)
            lat2_rad, lon2_rad = radians(lat2), radians(lon2)
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            return R * c

        edge_index = [[], []]
        edge_weights = []

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dist = haversine_distance(lats[i], lons[i], lats[j], lons[j])
                if dist <= distance_threshold_km:
                    edge_index[0].append(i)
                    edge_index[1].append(j)
                    edge_weights.append(dist)

        if len(edge_weights) == 0:
            print("Warning: No edges were added. Consider increasing the distance threshold.")

        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)

        connectivity = (edge_index, edge_weight)
        return connectivity

    def gen_spatio_temporal_dataset(self, df):
        """
        Converts a DataFrame into a SpatioTemporalDataset suitable for GNN models.

        Parameters
        ----------
        df : DataFrame
            Merged and imputed dataset.

        Returns
        -------
        SpatioTemporalDataset
            A TSL-formatted dataset containing temporal, spatial, and covariate information.
        """
        index_cols = ['date','hour','node_id']
        datetime_index = pd.DatetimeIndex(pd.to_datetime(df.date) + pd.to_timedelta(df.hour, unit="h"))
        dt_index = datetime_index.unique()
        n_timesteps = datetime_index.nunique()
        n_nodes = df.node_id.nunique()

        target_cols = ['pm_25', 'pm_10', 'no2', 'co', 'co2', 'ch4']
        target = df[target_cols].values.reshape((n_timesteps, n_nodes, len(target_cols) ))

        mask_cols = ['m_' + col for col in target_cols]

        mask_df = df[mask_cols]
        mask = mask_df.astype(bool).values.reshape((n_timesteps, n_nodes, len(target_cols)))

        connectivity = self.gen_connectivity()

        covariate_cols = df.columns.difference(set(target_cols + index_cols))
        covariates = {}
        for i, col in enumerate(covariate_cols):
            col_data = df[col].values.reshape(n_timesteps, n_nodes)
            covariates[col] = col_data

        input_map = {
            "x": (["target"], WINDOW),
            "u": (list(covariates.keys()), WINDOW),
        }

        torch_dataset = SpatioTemporalDataset(
            target=target,
            index=dt_index,
            mask=mask,
            connectivity=connectivity,
            covariates=covariates,
            horizon=self.horizon,
            window=self.window,
            stride=self.stride,
            input_map=input_map
        )

        return torch_dataset

    def gen_data_module(self, torch_dataset):
        """
        Wraps a SpatioTemporalDataset into a DataModule with normalization and sequential splitting.

        Parameters
        ----------
        torch_dataset : SpatioTemporalDataset
            The dataset to wrap and prepare for training.

        Returns
        -------
        SpatioTemporalDataModule
            The data module with scaling and data splitting applied.
        """
        scalers = {'target': RobustScaler(axis=(0, 1))}
        cov_cols = torch_dataset.covariates.keys()
        scalers.update({cov:RobustScaler(axis=(0, 1)) for cov in cov_cols})

        splitter = TemporalSplitter(val_len=self.validation_frac, test_len=self.test_frac)

        dm = SpatioTemporalDataModule(dataset=torch_dataset, scalers=scalers, splitter=splitter, batch_size=64)

        return dm

    def execute(self):
        """
        Executes the full pipeline: load, impute, merge, transform, and save.

        Returns
        -------
        None
        """
        merged_df = self.create_single_df()
        # merged_df = self.dbx_helper.read_parquet(self.dbx_helper.clean_merged_input_path, self.city,'merged.parquet')
        torch_dataset = self.gen_spatio_temporal_dataset(merged_df)
        dm = self.gen_data_module(torch_dataset)
        self.dbx_helper.save_torch(dm, self.dbx_helper.output_path, self.city, 'data_module.pt')
