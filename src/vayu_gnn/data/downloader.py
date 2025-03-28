import pandas as pd
from urllib.request import urlopen
import zipfile, io
import requests
import logging
from datetime import datetime
import time
import os
import requests_cache
from openmeteo_requests import Client
from retry_requests import retry
from io import StringIO

from vayu_gnn.dbx.dbx_config import DropboxHelper

"""
Module for downloading sensor, weather, forecast, and pollution data.

This module defines the Downloader class which encapsulates methods to download data from various APIs,
process the data, and write it to files using a DropboxHelper instance.

Classes
-------
Downloader
    Class to handle data downloading operations for sensor data, weather data, weather forecast data, and air pollution data.
"""

class Downloader():
    """
    A class used to download various types of data including sensor data, weather data, weather forecast data, and air pollution data.

    Parameters
    ----------
    dbx_helper : DropboxHelper
        An instance of DropboxHelper used for file operations.
    urls : dict
        Dictionary containing API endpoint URLs.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    nodes : dict
        Dictionary containing node information keyed by city.
    cities : list
        List of cities to process.
    """

    def __init__(self, dbx_helper:DropboxHelper, urls:dict, start_date:str, end_date:str, nodes:dict, cities:list):
        """
        Initialize the Downloader instance.

        Parameters
        ----------
        dbx_helper : DropboxHelper
            An instance of DropboxHelper used for file operations.
        urls : dict
            Dictionary containing API endpoint URLs.
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        end_date : str
            End date in 'YYYY-MM-DD' format.
        nodes : dict
            Dictionary containing node information keyed by city.
        cities : list
            List of cities to process.
        """
        
        self.dbx_helper = dbx_helper
        self.urls = urls
        self.start_date = start_date
        self.end_date = end_date
        self.nodes = nodes
        self.cities = cities

    def sensor_data(self, device_types:list, months_years:list, headers:dict):
        """
        Fetch sensor data for specified device types and time periods.

        Parameters
        ----------
        device_types : list
            List of device types to fetch data for.
        months_years : list
            List of tuples (month, year) representing the time periods for which data is fetched.
        headers : dict
            Dictionary of HTTP headers to include in the API request.

        Notes
        -----
        If the city is 'Gurugram', the first element of months_years is removed.
        Data for each month is concatenated and written to a CSV file via DropboxHelper.
        """
        
        # Loop through each city and device type
        for city in self.cities:
            for device_type in  device_types:
                
                monthly_dfs = []  
                
                if city == "Gurugram":
                    months_years.pop(0)

                for month, year in months_years:
                    print(f"Fetching data for {city}, {device_type} for {month} {year}")
                    
                    payload = {
                        "month": month,
                        "year": year,
                        "city": city,
                        "device_type": device_type
                    }
                    response = requests.post(self.urls['sensor_data'], headers=headers, json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("success"):
                            download_link = data.get("data")
                            csv_response = requests.get(download_link)
                            if csv_response.status_code == 200:
                                csv_content = csv_response.content.decode('utf-8')
                                df = pd.read_csv(StringIO(csv_content))
                                monthly_dfs.append(df)
                            else:
                                print(f"Failed to download CSV for {city}, {device_type} for {month} {year}. Status code: {csv_response.status_code}")
                        else:
                            print(f"API error for {city}, {device_type} for {month} {year}: {data.get('message')}")
                    else:
                        print(f"HTTP error for {city}, {device_type} for {month} {year}: {response.status_code}")
                
                if monthly_dfs:
                    concatenated_df = pd.concat(monthly_dfs, ignore_index=True)
                else:
                    concatenated_df = pd.DataFrame()
                
                self.dbx_helper.write_csv(concatenated_df, self.dbx_helper.raw_input_path, f'sensor_data/{city}',  f"{device_type}_sensor_data.csv")

    def weather(self, api_params:dict):
        """
        Download and process weather data for each city.

        Parameters
        ----------
        api_params : dict
            Dictionary containing parameters for the weather API request.

        Notes
        -----
        Utilizes caching and retry mechanisms for API requests.
        Processes hourly weather data and writes the result to a CSV file via DropboxHelper.
        """
        
        for city in self.cities:
            print(f"Downloading weather data for {city}")
            # Setup the Open-Meteo API client with cache and retry on error
            cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
            retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
            openmeteo = Client(session=retry_session)

            dataframes = []

            for node_name, coords in self.nodes[city].items():
                print(node_name)
                
                api_params["latitude"] = coords["lat"]
                api_params["longitude"] = coords["long"]
                
                responses = openmeteo.weather_api(self.urls['weather'], params=api_params)
                response = responses[0]
                
                api_lat = response.Latitude()
                api_long = response.Longitude()
                
                # Process hourly data from the response
                hourly = response.Hourly()
                hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
                hourly_wind_speed_10m = hourly.Variables(1).ValuesAsNumpy()
                hourly_wind_direction_10m = hourly.Variables(2).ValuesAsNumpy()
                hourly_relative_humidity_2m = hourly.Variables(3).ValuesAsNumpy()
                hourly_precipitation = hourly.Variables(4).ValuesAsNumpy()
                hourly_soil_temperature_0_to_7cm = hourly.Variables(5).ValuesAsNumpy()
                hourly_soil_moisture_0_to_7cm = hourly.Variables(6).ValuesAsNumpy()
                hourly_surface_pressure = hourly.Variables(7).ValuesAsNumpy()
                hourly_cloud_cover = hourly.Variables(8).ValuesAsNumpy()
                
                # Create a date range based on the hourly metadata
                hourly_data = {
                    "date": pd.date_range(
                        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                        freq=pd.Timedelta(seconds=hourly.Interval()),
                        inclusive="left"
                    )
                }
                
                # Add all the requested weather variables to the data dictionary
                hourly_data["temperature"] = hourly_temperature_2m
                hourly_data["wind_speed"] = hourly_wind_speed_10m
                hourly_data["wind_direction"] = hourly_wind_direction_10m
                hourly_data["humidity"] = hourly_relative_humidity_2m
                hourly_data["precipitation"] = hourly_precipitation
                hourly_data["soil_temperature"] = hourly_soil_temperature_0_to_7cm
                hourly_data["soil_moisture"] = hourly_soil_moisture_0_to_7cm
                hourly_data["pressure"] = hourly_surface_pressure
                hourly_data["cloud_cover"] = hourly_cloud_cover
                
                # Create a DataFrame for the current device
                df = pd.DataFrame(data=hourly_data)
                
                # Add metadata columns from both the device dictionary and the API response
                df["node"] = node_name
                df["device_lat"] = coords["lat"]
                df["device_long"] = coords["long"]
                df["om_lat"] = api_lat
                df["om_long"] = api_long

                dataframes.append(df)
                time.sleep(3)

            # Concatenate all DataFrames into a single DataFrame
            df = pd.concat(dataframes)
            self.dbx_helper.write_csv(df, self.dbx_helper.raw_input_path, f'weather/{city}', f"weather.csv")
    
    def weather_forecast(self, api_params:dict):
        """
        Download and process weather forecast data for each city.

        Parameters
        ----------
        api_params : dict
            Dictionary containing parameters for the weather forecast API request.

        Notes
        -----
        Utilizes caching and retry mechanisms for API requests.
        Processes hourly weather forecast data and writes the result to a CSV file via DropboxHelper.
        """
        
        for city in self.cities:
            print(f"Downloading weather forecast data for {city}")
            # Setup the Open-Meteo API client with cache and retry on error
            cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
            retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
            openmeteo = Client(session=retry_session)

            dataframes = []

            for node_name, coords in self.nodes[city].items():
                print(node_name)
                
                api_params["latitude"] = coords["lat"]
                api_params["longitude"] = coords["long"]
                
                responses = openmeteo.weather_api(self.urls['weather_forecast'], params=api_params)
                response = responses[0]
                
                api_lat = response.Latitude()
                api_long = response.Longitude()
                
                # Process hourly data from the response
                hourly = response.Hourly()
                hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
                hourly_wind_speed_10m = hourly.Variables(1).ValuesAsNumpy()
                hourly_wind_direction_10m = hourly.Variables(2).ValuesAsNumpy()
                hourly_relative_humidity_2m = hourly.Variables(3).ValuesAsNumpy()
                hourly_precipitation = hourly.Variables(4).ValuesAsNumpy()
                hourly_soil_temperature_0cm = hourly.Variables(5).ValuesAsNumpy()
                hourly_soil_moisture_0_to_1cm = hourly.Variables(6).ValuesAsNumpy()
                hourly_surface_pressure = hourly.Variables(7).ValuesAsNumpy()
                hourly_cloud_cover = hourly.Variables(8).ValuesAsNumpy()
                
                # Create a date range based on the hourly metadata
                hourly_data = {
                    "date": pd.date_range(
                        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                        freq=pd.Timedelta(seconds=hourly.Interval()),
                        inclusive="left"
                    )
                }
                
                # Add all the requested weather variables to the data dictionary
                hourly_data["temperature"] = hourly_temperature_2m
                hourly_data["wind_speed"] = hourly_wind_speed_10m
                hourly_data["wind_direction"] = hourly_wind_direction_10m
                hourly_data["humidity"] = hourly_relative_humidity_2m
                hourly_data["precipitation"] = hourly_precipitation
                hourly_data["soil_temperature"] = hourly_soil_temperature_0cm
                hourly_data["soil_moisture"] = hourly_soil_moisture_0_to_1cm
                hourly_data["pressure"] = hourly_surface_pressure
                hourly_data["cloud_cover"] = hourly_cloud_cover
                
                # Create a DataFrame for the current device
                df = pd.DataFrame(data=hourly_data)
                
                # Add metadata columns from both the device dictionary and the API response
                df["node"] = node_name
                df["device_lat"] = coords["lat"]
                df["device_long"] = coords["long"]
                df["om_lat"] = api_lat
                df["om_long"] = api_long

                dataframes.append(df)
                time.sleep(3)

            # Concatenate all DataFrames into a single DataFrame
            df = pd.concat(dataframes)
            self.dbx_helper.write_csv(df, self.dbx_helper.raw_input_path, f'weather_forecast/{city}', f"weather_forecast.csv")

    def open_weather_pollution(self, api_key):
        """
        Download and process air pollution data from OpenWeather API for each city.

        Parameters
        ----------
        api_key : str
            API key for accessing the OpenWeather air pollution endpoint.

        Notes
        -----
        Uses helper functions to retrieve and process pollution data,
        convert timestamps, and transform data into a pandas DataFrame.
        The final data is written as a Parquet file via DropboxHelper.
        """

        def get_pollution_data(lat, lon, start_unix, end_unix):
            """
            Retrieve air pollution data for specified coordinates and time range.

            Parameters
            ----------
            lat : float
                Latitude coordinate.
            lon : float
                Longitude coordinate.
            start_unix : int
                Start time as a Unix timestamp.
            end_unix : int
                End time as a Unix timestamp.

            Returns
            -------
            dict
                JSON response parsed as a dictionary containing pollution data.
            """
            
            base_open_weather_url = self.urls['open_weather_pollution']
            complete_url = f"{base_open_weather_url}lat={lat}&lon={lon}&start={start_unix}&end={end_unix}&appid={api_key}"

            response = requests.get(complete_url)
            x = response.json()
            
            return x
        
        def year_month_to_unix(year_month_day: str) -> int:
            """
            Convert a date string in 'YYYY-MM-DD' format to a Unix timestamp (UTC).

            Parameters
            ----------
            year_month_day : str
                Date string in the format 'YYYY-MM-DD'.

            Returns
            -------
            int
                Unix timestamp corresponding to the given date.
            """
            
            dt = datetime.strptime(year_month_day, "%Y-%m-%d")  # Convert to datetime
            return int(time.mktime(dt.timetuple()))  # Convert to Unix timestamp

        def air_quality_to_dataframe(data: list) -> pd.DataFrame:
            """
            Convert a list of air quality data dictionaries to a pandas DataFrame.

            Parameters
            ----------
            data : list
                List of dictionaries containing air quality data.

            Returns
            -------
            pd.DataFrame
                DataFrame with columns ['aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 'dt'].
            """
            
            # Extract relevant fields
            df = pd.DataFrame([
                {
                    'aqi': entry['main']['aqi'],
                    'co': entry['components']['co'],
                    'no': entry['components']['no'],
                    'no2': entry['components']['no2'],
                    'o3': entry['components']['o3'],
                    'so2': entry['components']['so2'],
                    'pm2_5': entry['components']['pm2_5'],
                    'pm10': entry['components']['pm10'],
                    'nh3': entry['components']['nh3'],
                    'dt': entry['dt']
                }
                for entry in data
            ])

            return df
        
        def unix_to_readable(timestamp: int, fmt: str = "%Y-%m-%d %H:%M:%S", tz=None) -> str:
            """
            Convert a Unix timestamp to a human-readable datetime string.

            Parameters
            ----------
            timestamp : int
                Unix timestamp (seconds since epoch).
            fmt : str, optional
                Desired output format (default is "%Y-%m-%d %H:%M:%S").
            tz : timezone, optional
                Timezone information; if None, UTC is assumed.

            Returns
            -------
            str
                Formatted datetime string.
            """
            
            dt = datetime.utcfromtimestamp(timestamp) if tz is None else datetime.fromtimestamp(timestamp, tz)
            return dt.strftime(fmt)
        
        start_unix = year_month_to_unix(self.start_date)
        end_unix = year_month_to_unix(self.end_date)

        for city in self.cities:
            
            list_node_dfs= []

            for node, coords in self.nodes[city].items():
                resulting_response = get_pollution_data(coords['lat'], coords['long'], start_unix, end_unix)
                node_df = air_quality_to_dataframe(resulting_response['list'])
                node_df['dt'] = node_df.dt.apply(unix_to_readable)
                node_df = node_df.assign(node = node, lat = coords['lat'], long = coords['long'])
                list_node_dfs.append(node_df)
            self.dbx_helper.write_parquet(pd.concat(list_node_dfs), self.dbx_helper.raw_input_path, f'pollution/{city}', 'pollution.parquet')