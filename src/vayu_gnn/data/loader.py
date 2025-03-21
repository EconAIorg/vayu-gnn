from vayu_gnn.dbx import dbx_helper

class DataLoader:
    def __init__(self, dbx_helper = dbx_helper):
        """
        Initialize the DataLoader class.

        Parameters
        ----------
        dbx_helper : DropboxHelper
            A helper object to read/write data from/to Dropbox.
        """
        self.dbx_helper = dbx_helper

        # Dynamically create loader methods for each dataset
        datasets = ["pollution", "elevation", "weather", "weather_forecast"]
        for dataset in datasets:
            # Create a loader function using an inner function
            loader_func = self._make_loader_func(dataset)
            setattr(self, dataset, loader_func)

    def _make_loader_func(self, dataset_name):
        """
        Create a loader function for a given dataset name.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to create a loader for.

        Returns
        -------
        function
            A function that loads the specified dataset for a given city.
        """
        def loader(city):
            return self.dbx_helper.read_parquet(
                self.dbx_helper.clean_input_path,
                f"{dataset_name}/{city}",
                f"{dataset_name}.parquet"
            )
        return loader

    def sensor_data(self, city):
        """
        Load sensor data for a given city.

        Parameters
        ----------
        city : str
            The name of the city.

        Returns
        -------
        DataFrame
            The loaded sensor data.
        """
        return self.dbx_helper.read_parquet(
                self.dbx_helper.clean_input_path,
                f"sensor_data/{city}",
                f"static_sensor_data_hourly.parquet"
            )

    def node_locations(self, city):
        """
        Load node location data for a given city.

        Parameters
        ----------
        city : str
            The name of the city.

        Returns
        -------
        dict
            A dictionary of node locations.
        """
        return self.dbx_helper.read_pickle(
                self.dbx_helper.clean_input_path,
                f"node_locations/{city}",
                f"nodes.pickle"
            )

    def settlement(self, city, building_type, dimension, size):
        """
        Load a specific settlement file based on parameters.

        Parameters
        ----------
        city : str
            The city name.
        building_type : str
            The type of building.
        dimension : str
            The building dimension.
        size : str
            The building size.

        Returns
        -------
        DataFrame
            The requested settlement data.
        """
        return self.dbx_helper.read_parquet(
                self.dbx_helper.clean_input_path,
                f"settlement/{city}",
                f"GHS_{building_type}_building_{dimension}_b{size}.parquet"
            )

    def settlements(self, city):
        """
        Load and merge all available settlement files for a city.

        Parameters
        ----------
        city : str
            The name of the city.

        Returns
        -------
        DataFrame
            A merged DataFrame of all settlement files for the city.
        """
        files_to_load = self.dbx_helper.list_files_in_folder(folder_path = f'{self.dbx_helper.clean_input_path}/settlement/{city}' )
        df = self.dbx_helper.read_parquet(self.dbx_helper.clean_input_path, f'settlement/{city}', files_to_load[0])
        for file in files_to_load[1:]:
            _df = self.dbx_helper.read_parquet(self.dbx_helper.clean_input_path, f'settlement/{city}', file)
            df = df.merge(_df, how='left', on='node_id')
        return df

    def multi_load(self, city:str, sources:list):
        """
        Load multiple data sources at once for a given city.

        Parameters
        ----------
        city : str
            City name, e.g., 'Patna' or 'Gurugram'.
        sources : list of str
            List of source dataset names to load.

        Returns
        -------
        None
        """
        self.dfs = {}

        for source in sources:
            self.dfs[source] = getattr(self, source)(city = city)

    def multi_merge(self, remove_dfs = False):
        """
        Merge multiple loaded data sources into a single DataFrame.

        Parameters
        ----------
        remove_dfs : bool, optional
            Whether to delete the individual source DataFrames after merging, by default False.

        Returns
        -------
        DataFrame
            The merged DataFrame.
        """
        df = self.dfs['sensor_data']
        for source in self.dfs.keys() - {"sensor_data"}:
            if {'node_id','date','hour'}.issubset(self.dfs[source].columns):
                df = df.merge(self.dfs[source], how = 'left', on=['node_id','date','hour'])
            else:
                df = df.merge(self.dfs[source], how = 'left', on=['node_id'])
        if remove_dfs:
            del self.dfs
        return df
