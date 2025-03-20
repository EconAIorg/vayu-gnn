from vayu_gnn.dbx import dbx_helper

class DataLoader:
    def __init__(self, dbx_helper = dbx_helper):
        """
        Initialize data loader class.

        Parameters
        ----------
        dbx_helper : DropboxHelper
            A helper to download and upload files to Dropbox.
        datasets : list of str, optional
            The list of dataset names that need loader functions.
        """
        self.dbx_helper = dbx_helper
        
        # Dynamically create loader methods for each dataset
        datasets=["pollution", "elevation", "weather", "weather_forecast"]
        for dataset in datasets:
            # Create a loader function using an inner function or lambda
            loader_func = self._make_loader_func(dataset)
            
            # Assign it as a method of this instance with name = dataset
            setattr(self, dataset, loader_func)

    def _make_loader_func(self, dataset_name):
        """
        Create a loader function that reads a parquet file
        from the appropriate subfolder.
        """
        def loader(city):
            return self.dbx_helper.read_parquet(
                self.dbx_helper.clean_input_path,
                f"{dataset_name}/{city}",
                f"{dataset_name}.parquet"
            )
        return loader
    
    def sensor_data(self, city):
        return self.dbx_helper.read_parquet(
                self.dbx_helper.clean_input_path,
                f"sensor_data/{city}",
                f"static_sensor_data_hourly.parquet"
            )
    def node_locations(self, city):
        return self.dbx_helper.read_pickle(
                self.dbx_helper.clean_input_path,
                f"node_locations/{city}",
                f"nodes.pickle"
            )
    
    def settlement(self, city, building_type, dimension, size):
        return self.dbx_helper.read_parquet(
                self.dbx_helper.clean_input_path,
                f"settlement/{city}",
                f"GHS_{building_type}_building_{dimension}_b{size}.parquet"
            )
    
    def settlements(self, city):
        files_to_load = self.dbx_helper.list_files_in_folder(folder_path = f'{self.dbx_helper.clean_input_path}/settlement/{city}' )
        df = self.dbx_helper.read_parquet(self.dbx_helper.clean_input_path, f'settlement/{city}', files_to_load[0])
        for file in files_to_load[1:]:
            _df = self.dbx_helper.read_parquet(self.dbx_helper.clean_input_path, f'settlement/{city}', file)
            df = df.merge(_df, how='left', on='node_id')
        return df

    def multi_load(self, city:str, sources:list):
        """Loads multiple sourches at once.

        Parameters
        ----------
        city : str
            Patna or Gurugram.
        sources : list
            A list of strings, specifying each source.
        """

        self.dfs = {}

        for source in sources:
            self.dfs[source] = getattr(self, source)(city = city)
    
    def multi_merge(self, remove_dfs = False):
        df = self.dfs['sensor_data']
        for source in self.dfs.keys() - {"sensor_data"}:
            if ['node_id','date','hour'] in self.dfs[source]:
                df = df.merge(self.dfs[source], how = 'left', on=['node_id','date','hour'])
            else:
                df = df.merge(self.dfs[source], how = 'left', on=['node_id'])
        if remove_dfs:
            del self.dfs
        return df
        
    

