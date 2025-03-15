import pandas as pd
import rasterio
import numpy as np
from rasterio.mask import mask
from shapely.geometry import mapping
from rasterio.merge import merge
from rasterio.io import MemoryFile
from vayu_gnn.dbx.dbx_config import dbx_helper

class Preprocessor():

    def __init__(self, dbx_helper, nodes:dict, cities:list):

        self.dbx_helper = dbx_helper
        self.nodes = nodes
        self.node_ids = {}
        for city, nodes in self.nodes.items():
            self.node_ids[city] = list(nodes.keys())

        self.cities = cities
        self.base_panels = self._create_base_panels()

    def _create_base_panels(self):

        base_panels = {}

        for city in self.cities:

            if city == 'Patna':
                datetime_range = pd.date_range(start='2024-06-01', end='2025-02-27', freq='h')
            if city == 'Gurugram':
                datetime_range = pd.date_range(start='2024-07-01', end='2025-02-27', freq='h')

            complete_index = pd.MultiIndex.from_product(
                [self.node_ids[city], datetime_range],
                names=['node_id', 'datetime']
            )
            
            complete_df = complete_index.to_frame(index=False)
            complete_df['date'] = complete_df['datetime'].dt.date
            complete_df['hour'] = complete_df['datetime'].dt.hour
            complete_df.drop(columns='datetime', inplace=True)

            complete_df = complete_df[complete_df['date'].astype(str) != '2025-02-27']
            base_panels[city] = complete_df

        return base_panels

    def sensor_data(self, device_types):

        for city in self.cities:

            for device_type in device_types:
                print(f"Processing {device_type} sensor data for {city}")
                df = dbx_helper.read_csv(dbx_helper.raw_input_path, f'sensor_data/{city}', f"{device_type}_sensor_data.csv")

                # Create date and hour columns
                df['data_created_time'] = pd.to_datetime(df['data_created_time'])
                df['date'] = df['data_created_time'].dt.date
                df['hour'] = df['data_created_time'].dt.hour
                df = df.drop(columns=['data_created_time', 'id', 'lat', 'long'])

                # Group by date and hour and take the average of the sensor values
                hourly_df = df.groupby(['device_name', 'date', 'hour']).mean().reset_index().copy()

                hourly_df = hourly_df[['device_name', 'date', 'hour'] + [col for col in hourly_df.columns if col not in ['device_name', 'date', 'hour']]]
                hourly_df = hourly_df.rename(columns={'device_name': 'node_id'})
                hourly_df = hourly_df.sort_values(by=['node_id', 'date', 'hour'])

                ## Expanding the number of rows to include all hours in the date range ##
                merged_hourly_df = pd.merge(self.base_panels[city], hourly_df, on=['node_id', 'date', 'hour'], how='left')

                # Add binary missing flags to be used as masks in the GNN loss function
                for col in merged_hourly_df.columns:
                    if col not in ['node_id', 'date', 'hour', 'rh']:
                        merged_hourly_df[f'm_{col}'] = merged_hourly_df[col].notnull().astype(int)

                # because temperature is present for every reading, where it is missing is where the sensor reading is totally missing due to the expanded rows
                merged_hourly_df = merged_hourly_df.rename(columns={'m_temp': 'm_senor_reading'})
                merged_hourly_df = merged_hourly_df.fillna(0)

                merged_hourly_df = merged_hourly_df.sort_values(by=['node_id', 'date', 'hour'])

                self.dbx_helper.write_parquet(merged_hourly_df, self.dbx_helper.clean_input_path, f'sensor_data/{city}', f"{device_type}_sensor_data_hourly.parquet")

    def raster_data(self, raster_files):

        ### First combine raster tiles for Global Human Settlement data ###
        def combine_and_upload_tifs(tif1, tif2, directory: str, filename: str):
            """
            Combines two in-memory TIFF files (side-by-side) and uploads the resulting mosaic to Dropbox.
            
            Args:
                tif1 (BytesIO): The first TIFF file in-memory.
                tif2 (BytesIO): The second TIFF file in-memory.
                directory (str): The directory within the base Dropbox path where the file will be saved.
                filename (str): The name of the output combined TIFF file.
            """
            if tif1 is None or tif2 is None:
                print("Failed to load one of the files.")
                return

            # Open the in-memory files using MemoryFile and merge them
            with MemoryFile(tif1) as memfile1, MemoryFile(tif2) as memfile2:
                with memfile1.open() as src1, memfile2.open() as src2:
                    # Merge the two datasets into one mosaic
                    mosaic, out_trans = merge([src1, src2])
                    
                    # Update the metadata based on the merged raster
                    out_meta = src1.meta.copy()
                    out_meta.update({
                        "height": mosaic.shape[1],
                        "width": mosaic.shape[2],
                        "transform": out_trans,
                        "driver": "GTiff",
                        "compress": "lzw"  # Using LZW compression to reduce file size
                    })

            # Write the mosaic to an in-memory file (BytesIO)
            with MemoryFile() as memfile:
                with memfile.open(**out_meta) as dest:
                    dest.write(mosaic)
                memfile.seek(0)  # Ensure we are at the start of the file
                combined_tif_bytes = memfile.read()

            # Upload the combined TIFF to Dropbox using your existing helper function
            self.dbx_helper.write_bytes(combined_tif_bytes, dbx_helper.raw_input_path, directory, filename)

        # Total (non-residential + residential) building area
        tif_1 = dbx_helper.read_tif(dbx_helper.raw_input_path, 'settlement/total', 'GHS_BUILT_S_E2025_GLOBE_R2023A_54009_100_V1_0_R6_C26.tif')
        tif_2 = dbx_helper.read_tif(dbx_helper.raw_input_path, 'settlement/total', 'GHS_BUILT_S_E2025_GLOBE_R2023A_54009_100_V1_0_R6_C27.tif')
        combine_and_upload_tifs(tif_1, tif_2, 'settlement', 'GHS_total_building_area.tif')

        # Total (non-residential + residential) building volume
        tif_1 = dbx_helper.read_tif(dbx_helper.raw_input_path, 'settlement/total', 'GHS_BUILT_V_E2025_GLOBE_R2023A_54009_100_V1_0_R6_C26.tif')
        tif_2 = dbx_helper.read_tif(dbx_helper.raw_input_path, 'settlement/total', 'GHS_BUILT_V_E2025_GLOBE_R2023A_54009_100_V1_0_R6_C27.tif')
        combine_and_upload_tifs(tif_1, tif_2, 'settlement', 'GHS_total_building_volume.tif')

        # Non-residential building area
        tif_1 = dbx_helper.read_tif(dbx_helper.raw_input_path, 'settlement/non_residential', 'GHS_BUILT_S_NRES_E2025_GLOBE_R2023A_54009_100_V1_0_R6_C26.tif')
        tif_2 = dbx_helper.read_tif(dbx_helper.raw_input_path, 'settlement/non_residential', 'GHS_BUILT_S_NRES_E2025_GLOBE_R2023A_54009_100_V1_0_R6_C27.tif')
        combine_and_upload_tifs(tif_1, tif_2, 'settlement', 'GHS_non_res_building_area.tif')

        # Non-residential building volume
        tif_1 = dbx_helper.read_tif(dbx_helper.raw_input_path, 'settlement/non_residential', 'GHS_BUILT_V_NRES_E2025_GLOBE_R2023A_54009_100_V1_0_R6_C26.tif')
        tif_2 = dbx_helper.read_tif(dbx_helper.raw_input_path, 'settlement/non_residential', 'GHS_BUILT_V_NRES_E2025_GLOBE_R2023A_54009_100_V1_0_R6_C27.tif')
        combine_and_upload_tifs(tif_1, tif_2, 'settlement', 'GHS_non_res_building_volume.tif')

        ### Next, 
        def pixel_sum_around_point(gdf, raster_bytes, stat_col_name, buffer_distance=0):
            """
            Returns a DataFrame with each device's identifier and its corresponding raster summary statistic.
            
            For single-band rasters, when buffer_distance is 0, the raster value is sampled exactly at the point.
            If buffer_distance > 0, a buffer polygon is created around the point, pixels within that area are extracted,
            and their values are summed.
            
            Args:
                gdf (GeoDataFrame): A GeoDataFrame containing device geometries (Points) and an optional
                                    'device_id' column. If missing, the index is used.
                raster_bytes (BytesIO): The in-memory TIFF file (assumed to be single-band).
                buffer_distance (float): The buffer distance (in CRS units). Set to 0 to sample the exact point.
            
            Returns:
                DataFrame: A DataFrame with columns 'device_id' and 'summary_stat', where summary_stat is a scalar.
            """
            results = {}
            
            with rasterio.open(raster_bytes) as src:
                raster_crs = src.crs
                
                # Reproject GeoDataFrame if its CRS doesn't match the raster's CRS.
                if gdf.crs != raster_crs:
                    print(f"Reprojecting GeoDataFrame from {gdf.crs} to {raster_crs}")
                    devices_in_raster_crs = gdf.to_crs(raster_crs)
                else:
                    devices_in_raster_crs = gdf
                
                for idx, row in devices_in_raster_crs.iterrows():
                    device_id = row.get("device_id", idx)
                    geom = row.geometry
                    
                    if buffer_distance == 0:
                        # Sample the raster value at the exact point location.
                        for val in src.sample([(geom.x, geom.y)]):
                            results[device_id] = float(val[0])
                    else:
                        # Create a buffer polygon around the point.
                        buffer_geom = geom.buffer(buffer_distance)
                        geom_mapping = [mapping(buffer_geom)]
                        
                        # Mask the raster to get pixels within the buffer.
                        out_image, _ = mask(src, geom_mapping, crop=True)
                        
                        # Handle nodata values by converting them to NaN.
                        nodata = src.nodata
                        if nodata is not None:
                            out_image = np.where(out_image == nodata, np.nan, out_image)
                        # For a single band, out_image shape is (1, height, width).
                        band_sum = np.nansum(out_image[0])
                        results[device_id] = float(band_sum)
            
            # Convert the results into a DataFrame.
            data = [{"node_id": device_id, stat_col_name: summary} for device_id, summary in results.items()]
            return pd.DataFrame(data)

        for city in self.cities:

            devices = self.dbx_helper.read_shp(self.dbx_helper.clean_input_path, f'node_locations/{city}/gdf')

            for file in raster_files:

                if file == 'elevation':
                    tif = self.dbx_helper.read_tif(self.dbx_helper.raw_input_path, 'elevation', 'elevation.tif')
                
                    print(f'Calculating {file}')
                    result = pixel_sum_around_point(devices, tif, 'elevation', buffer_distance=0)
                    self.dbx_helper.write_parquet(result, dbx_helper.clean_input_path, f'{file}/{city}', f'{file}.parquet')

                else:

                    tif = self.dbx_helper.read_tif(self.dbx_helper.raw_input_path, 'settlement', f'{file}.tif')
                    
                    distances = [0,500]
                    for buffer in distances: 
                        print(f'Calculating {file} buffer {buffer}')
            
                        stat_col_name = 'sum_' + file + '_buffer_' + str(buffer)
                        result = pixel_sum_around_point(devices, tif, stat_col_name, buffer_distance=buffer)
                        self.dbx_helper.write_parquet(result, self.dbx_helper.clean_input_path, f'settlement/{city}', f'{file}_b{buffer}.parquet')
            
    def weather(self):

        for city in self.cities:
            df = self.dbx_helper.read_csv(self.dbx_helper.raw_input_path, f'weather/{city}', f"weather.csv")

            # Create date and hour columns from 'date'
            df['date'] = pd.to_datetime(df['date'])
            df['hour'] = df['date'].dt.hour
            df['date'] = df['date'].dt.date

            df = df[['node', 'date', 'hour', 'temperature', 'wind_speed', 'wind_direction', 'humidity',
                'precipitation', 'soil_temperature', 'soil_moisture', 'pressure', 'cloud_cover']]

            # Add a_ prefix to represent 'actual' weather values
            df.columns = [f'a_{col}' if col not in ['node', 'date', 'hour'] else col for col in df.columns]
            df = df.rename(columns={'node': 'node_id'})

            # Merge on the base panels to ensure all hours are present
            df = pd.merge(self.base_panels[city], df, on=['node_id', 'date', 'hour'], how='left')
            assert df['a_temperature'].isnull().sum() == 0

            self.dbx_helper.write_parquet(df, self.dbx_helper.clean_input_path, f'weather/{city}', f"weather.parquet")

    def weather_forecast(self):
        
        for city in self.cities:
            df = self.dbx_helper.read_csv(self.dbx_helper.raw_input_path, f'weather_forecast/{city}', f"weather_forecast.csv")

            # Create date and hour columns from 'date'
            df['date'] = pd.to_datetime(df['date'])
            df['hour'] = df['date'].dt.hour
            df['date'] = df['date'].dt.date

            df = df[['node', 'date', 'hour', 'temperature', 'wind_speed', 'wind_direction', 'humidity',
                'precipitation', 'soil_temperature', 'soil_moisture', 'pressure', 'cloud_cover']]

            # Add f_ prefix to represent 'forecasted' weather values
            df.columns = [f'f_{col}' if col not in ['node', 'date', 'hour'] else col for col in df.columns]
            df = df.rename(columns={'node': 'node_id'})

            # Merge on the base panels to ensure all hours are present
            df = pd.merge(self.base_panels[city], df, on=['node_id', 'date', 'hour'], how='left')
            assert df['f_temperature'].isnull().sum() == 0

            self.dbx_helper.write_parquet(df, self.dbx_helper.clean_input_path, f'weather_forecast/{city}', f"weather_forecast.parquet")

    def open_weather_pollution(self):
        
      for city in self.cities:  
        df = self.dbx_helper.read_parquet(dbx_helper.raw_input_path, f'pollution/{city}', 'pollution.parquet') 

        # Create date and hour columns from 'date'
        df.rename(columns={'dt': 'date', 'node':'node_id'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df['date'] = df['date'].dt.date

        df = df[pd.to_datetime(df['date']) >= '2024-05-01']

        df = df[['node_id', 'date', 'hour', 'aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']]
        df = df.sort_values(['node_id', 'date', 'hour'])

        # Add ow_ prefix to represent 'open_weather' pollution values
        df.columns = [f'ow_{col}' if col not in ['node_id', 'date', 'hour'] else col for col in df.columns]

        # Merge on the base panels to ensure all hours are present
        df = pd.merge(self.base_panels[city], df, on=['node_id', 'date', 'hour'], how='left')
        assert df['ow_aqi'].isnull().sum() == 0

        self.dbx_helper.write_parquet(df, self.dbx_helper.clean_input_path, f'pollution/{city}', f"pollution.parquet")

