from vayu_gnn.dbx import dbx_helper
from vayu_gnn.data import DataLoader
from vayu_gnn.data.impute import OutlierImputer, SpatialKNNImputer
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


class CreateSpatioTemporalDataset:
    def __init__(self, city : str, params:dict, dbx_helper = dbx_helper, loader = DataLoader):
        """Initialize for creation of SpatioTemporal Dataset

        Parameters
        ----------
        dbx_helper : _type_
            _description_
        params : dict
            _description_
        """
        self.city = city
        self.dbx_helper = dbx_helper
        self.loader = DataLoader(dbx_helper=dbx_helper)
        self.__dict__.update(params)
    
    def _impute(self):
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
        reference = self.loader.dfs['sensor_data'].copy()
        reference = (
            reference.rename(columns={'pm_25': 'pm2_5', 'pm_10': 'pm10'})
            .drop(columns=[col for col in reference.columns if col.startswith('m_')], errors='ignore')
        )
        reference.rename(
            columns={col: 'ow_' + col for col in reference.columns if col not in ["node_id", "date", "hour", "latitude", "longitude"]},
            inplace=True
        )

        vars_to_transform = [col for col in self.loader.dfs['pollution'].columns if col not in ["node_id", "date", "hour", "latitude", "longitude"]]

        # Split dataset for fitting
        pollution_fit_sample = self.loader.dfs['pollution'][self.loader.dfs['pollution']['date'] <= self.train_cutoff]

        # Define imputation pipeline
        pollution_impute_pipe = Pipeline([
            ('spatial_knn', SpatialKNNImputer(reference_data=reference)),
            ('knn', SklearnTransformerWrapper(KNNImputer(weights='distance'), variables=vars_to_transform))
        ], verbose=True)

        # Fit on the subset, then transform the entire dataset
        pollution_impute_pipe.fit(pollution_fit_sample)
        self.loader.dfs['pollution'] = pollution_impute_pipe.transform(self.loader.dfs['pollution'])
        print(self.loader.dfs['pollution'].isna().sum())

        ### --- Sensor Data Imputation --- ###
        reference = self.loader.dfs['pollution'].copy()
        reference.rename(
            columns={col: col.removeprefix('ow_') for col in reference.columns},
            inplace=True
        )
        reference.rename(columns={'pm2_5': 'pm_25', 'pm10': 'pm_10'}, inplace=True)

        # Select training subset
        sensor_train = self.loader.dfs['sensor_data'][self.loader.dfs['sensor_data']['date'] <= self.train_cutoff]

        # Define sensor pipeline
        vars_to_transform = [col for col in self.loader.dfs['sensor_data'].columns if col not in ["node_id", "date", "hour", "latitude", "longitude"]]
        spatial_knn = SpatialKNNImputer(reference_data=reference)
        knn = SklearnTransformerWrapper(KNNImputer(weights='distance'), variables=vars_to_transform)

        sensor_impute_pipe = Pipeline([
            ('outlier', OutlierImputer()),
            # ('spatial_knn', spatial_knn),
            ('knn', knn)
        ], verbose=True)

        # Fit on training subset, transform entire dataset
        sensor_impute_pipe.fit(sensor_train)
        self.loader.dfs['sensor_data'] = sensor_impute_pipe.transform(self.loader.dfs['sensor_data'])
        print(self.loader.dfs['sensor_data'].isna().sum())

        # Drop latitude and longitude columns from one of the sources
        self.loader.dfs['sensor_data'].drop(columns=['latitude', 'longitude'], inplace=True)
        # self.loader.dfs['pollution'].drop(columns=['latitude', 'longitude'], inplace=True)

    def create_single_df(self):
        self.loader.multi_load(sources = self.sources, city= self.city)
        self._impute()
        df = self.loader.multi_merge(remove_dfs=True)
        df.sort_values(['date','hour', 'node_id'], inplace=True)

        # assert df.index.is_monotonically_increasing, "DataFrame should be monotonically increasing."
        self.dbx_helper.write_parquet(df, self.dbx_helper.clean_merged_input_path, self.city, 'merged.parquet')

        return df
    
    def gen_connectivity(self):

        locs = self.loader.node_locations(self.city)

        # List node names and number of nodes
        node_names = list(sorted(locs.keys()))
        N = len(node_names)

        # Create arrays of latitudes and longitudes
        lats = np.array([locs[node]["lat"] for node in node_names])
        lons = np.array([locs[node]["long"] for node in node_names])

        # Function to compute Haversine distance (in kilometers)
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371.0  # Earth's radius in kilometers
            # Convert degrees to radians
            lat1_rad, lon1_rad = radians(lat1), radians(lon1)
            lat2_rad, lon2_rad = radians(lat2), radians(lon2)
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            return R * c

        # Create lists for edges and weights
        edge_index = [[], []]
        edge_weights = []

        # Build fully connected graph (skip self-loops)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dist = haversine_distance(lats[i], lons[i], lats[j], lons[j])
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_weights.append(dist)

        # Convert lists to torch tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)

        # The connectivity object to be passed to the spatiotemporal graph is:
        connectivity = (edge_index, edge_weight)

        return connectivity
    
    def gen_spatio_temporal_dataset(self, df):
        # Get DateTime index and n_nodes
        index_cols = ['date','hour','node_id']
        datetime_index = pd.DatetimeIndex(pd.to_datetime(df.date) + pd.to_timedelta(df.hour, unit="h"))
        dt_index = datetime_index.unique()
        n_timesteps = datetime_index.nunique()
        n_nodes = df.node_id.nunique()

        # Generate target array
        target_cols = ['pm_25', 'pm_10', 'no2', 'co', 'co2', 'ch4', 'rh']
        target = df[target_cols].values.reshape((n_timesteps, n_nodes, 7 ))

        # Generate mask array
        mask_df = df[[col for col in df.columns if col.startswith('m_')]]
        mask = mask_df.astype(bool).values.reshape((n_timesteps, n_nodes, 7))

        # Connectivity
        connectivity = self.gen_connectivity()

        # Covariates
        covariate_cols = [col for col in df.columns not in target_cols + index_cols]
        covariates = df[covariate_cols].values.reshape((n_timesteps, n_nodes, len(covariate_cols)))

        torch_dataset = SpatioTemporalDataset(target = target  , index = dt_index, mask = mask, connectivity = connectivity, covariates = covariates, 
                        horizon=self.horizon, window=self.window, stride=self.stride)
        
        return torch_dataset
    
    def gen_data_module(self, torch_dataset):
        # Normalize data using mean and std computed over time and node dimensions
        scalers = {'target': RobustScaler(axis=(0, 1))}

        # Split data sequentially:
        #   |------------ dataset -----------|
        #   |--- train ---|- val -|-- test --|
        splitter = TemporalSplitter(val_len=self.validation_frac, test_len=self.test_frac)
        dm = SpatioTemporalDataModule(dataset=torch_dataset, scalers=scalers, splitter=splitter, batch_size=64)

        return dm
    
    def execute(self):
        merged_df = self.create_single_df()
        torch_dataset = self.load_into_spatio_temporal_dataset(merged_df)
        dm = self.gen_data_module(torch_dataset)
        self.dbx_helper.write_pickle(dm, self.dbx_helper.output_path, self.city, 'data_module.pickle')









        
        

