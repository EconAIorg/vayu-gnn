from dotenv import load_dotenv
import os

from vayu_gnn.dbx.dbx_config import dbx_helper, DropboxHelper

download_data = ['sensor_data'] #these correspond to the methods in the Downloader class (by constructions this also means the keys in download_params)
#download_data = ['open_weather_pollution', 'weather', 'weather_forecast'] #these correspond to the methods in the Downloader class (by constructions this also means the keys in download_params)

start_date = '2024-05-01'
end_date = '2025-02-28'

nodes = {}
nodes['Patna'] = dbx_helper.read_pickle(dbx_helper.clean_input_path, f'node_locations/Patna', f'nodes.pickle')
# nodes['Gurugram'] = dbx_helper.read_pickle(dbx_helper.clean_input_path, f'node_locations/Gurugram', f'nodes.pickle')

cities = ['Gurugram']
# cities = ['Patna', 'Gurugram']

load_dotenv(override=True)
OPEN_WEATHER_API_KEY = os.getenv('OPEN_WEATHER_API_KEY')

urls = {
    'open_weather_pollution': 'http://api.openweathermap.org/data/2.5/air_pollution/history?',
    'weather': 'https://archive-api.open-meteo.com/v1/archive',
    'weather_forecast': 'https://historical-forecast-api.open-meteo.com/v1/forecast',
    'sensor_data': 'https://vayuapi.undp.org.in/device/api/v1/sensor-data-download'
}

download_params = {
    'open_weather_pollution': {
        'api_key': OPEN_WEATHER_API_KEY
    },
    'weather':{'api_params': {"start_date": "2024-05-01",
                              "end_date": "2025-02-28",
                              "hourly": [
                                    "temperature_2m",
                                    "wind_speed_10m",
                                    "wind_direction_10m",
                                    "relative_humidity_2m",
                                    "precipitation",
                                    "soil_temperature_0_to_7cm",
                                    "soil_moisture_0_to_7cm",
                                    "surface_pressure",
                                    "cloud_cover"]
                             }
    },
    'weather_forecast': {'api_params': {"start_date": "2024-05-01",
                                        "end_date": "2025-02-28",
                                            "hourly": ["temperature_2m",
                                                       "wind_speed_10m",
                                                       "wind_direction_10m",
                                                       "relative_humidity_2m",
                                                       "precipitation",
                                                       "soil_temperature_0cm",
                                                       "soil_moisture_0_to_1cm",
                                                       "surface_pressure",
                                                       "cloud_cover"]
                                        }
    },
    'sensor_data': {'device_types': ["static"],
                    'months_years': [("June", "2024"),
                                     ("July", "2024"),
                                     ("August", "2024"),
                                     ("September", "2024"),
                                     ("October", "2024"),
                                     ("November", "2024"),
                                     ("December", "2024"),
                                     ("January", "2025"),
                                     ("February", "2025")],
                    'headers':{"accept": "application/json",
                               "Content-Type": "application/json"}
    }
}

