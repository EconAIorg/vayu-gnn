from vayu_gnn.dbx import dbx_helper
from vayu_gnn.data import DataLoader
from vayu_gnn.data.impute import OutlierImputer, SpatialKNNImputer
from tsl.data import SpatioTemporalDataset
from sklearn.impute import KNNImputer
from feature_engine.wrappers import SklearnTransformerWrapper
import pandas as pd
from feature_engine.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


class CreateSpatioTemporalDataset:
    def __init__(self, params:dict, dbx_helper = dbx_helper, loader = DataLoader):
        """Initialize for creation of SpatioTemporal Dataset

        Parameters
        ----------
        dbx_helper : _type_
            _description_
        params : dict
            _description_
        """
        self.dbx_helper = dbx_helper
        self.loader = DataLoader(dbx_helper=dbx_helper)
        self.__dict__.update(params)
    
    def _impute(self, city):
        # Determine the number of days in the dataset
        min_date = self.dfs['sensor_data']['date'].min()
        max_date = self.dfs['sensor_data']['date'].max()
        total_days = (max_date - min_date).days

        # Compute the cutoff date for the fitting sample
        fit_days = int(total_days * (1 - (self.validation_frac + self.test_frac)))
        self.train_cutoff = min_date + pd.Timedelta(days=fit_days)

        # Assign latitude & longitude from node locations
        locs = self.loader.node_locations(city=city)
        self.dfs['pollution'][['latitude', 'longitude']] = pd.DataFrame(self.dfs['pollution'].node_id.map(locs).tolist())
        self.dfs['sensor_data'][['latitude', 'longitude']] = pd.DataFrame(self.dfs['sensor_data'].node_id.map(locs).tolist())

        ### --- Pollution Data Imputation --- ###
        reference = self.loader['sensor_data'].copy()
        reference = (
            reference.rename(columns={'pm_25': 'pm2_5', 'pm_10': 'pm10'})
            .drop(columns=[col for col in reference.columns if col.startswith('m_')], errors='ignore')
        )
        reference.rename(
            columns={col: 'ow_' + col for col in reference.columns if col not in ["node_id", "date", "hour", "latitude", "longitude"]},
            inplace=True
        )

        vars_to_transform = [col for col in self.dfs['pollution'].columns if col not in ["node_id", "date", "hour", "latitude", "longitude"]]

        # Split dataset for fitting
        pollution_fit_sample = self.dfs['pollution'][self.dfs['pollution']['date'] <= self.train_cutoff]

        # Define imputation pipeline
        pollution_impute_pipe = Pipeline([
            ('spatial_knn', SpatialKNNImputer(reference_data=reference)),
            ('knn', SklearnTransformerWrapper(KNNImputer(weights='distance'), variables=vars_to_transform))
        ], verbose=True)

        # Fit on the subset, then transform the entire dataset
        pollution_impute_pipe.fit(pollution_fit_sample)
        self.loader['pollution'] = pollution_impute_pipe.transform(self.loader['pollution'])

        ### --- Sensor Data Imputation --- ###
        reference = self.loader['pollution'].copy()
        reference.rename(
            columns={col: col.removeprefix('ow_') for col in reference.columns},
            inplace=True
        )
        reference.rename(columns={'pm2_5': 'pm_25', 'pm10': 'pm_10'}, inplace=True)

        # Select training subset
        sensor_train = self.loader['sensor_data'][self.loader['sensor_data']['date'] <= self.train_cutoff]

        # Define sensor pipeline
        vars_to_transform = [col for col in self.dfs['sensor_data'].columns if col not in ["node_id", "date", "hour", "latitude", "longitude"]]
        spatial_knn = SpatialKNNImputer(reference_data=reference)
        knn = SklearnTransformerWrapper(KNNImputer(weights='distance'), variables=vars_to_transform)

        sensor_impute_pipe = Pipeline([
            ('outlier', OutlierImputer()),
            ('spatial_knn', spatial_knn),
            ('knn', knn)
        ], verbose=True)

        # Fit on training subset, transform entire dataset
        sensor_impute_pipe.fit(sensor_train)
        self.loader['sensor_data'] = sensor_impute_pipe.transform(self.loader['sensor_data'])

        # Drop latitude and longitude columns from one of the sources
        self.loader['sensor_data'].drop(columns=['latitude', 'longitude'], inplace=True)
        # self.loader['pollution'].drop(columns=['latitude', 'longitude'], inplace=True)

    def create_single_df(self, city):
        self.loader.multi_load(city= city)
        self._impute(city = city)
        df = self.loader.multi_merge(remove_dfs=True)
        df.set_index(['node_id','date','hour'], inplace=True)

        # assert df.index.is_monotonically_increasing, "DataFrame should be monotonically increasing."

        return df
    
    # def load_into_spatio_temporal_dataset(self, city):


        
        

