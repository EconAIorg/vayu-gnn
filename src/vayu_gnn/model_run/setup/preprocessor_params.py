from dotenv import load_dotenv
import os


preprocess_data = ['acled', 'ucdp', 'population', 'create_neighbor_dicts', 'static_geo_features', 'time_variant_geo_features', 'connect_articles_to_adm2'] #these correspond to the methods in the Preprocessor class (by constructions this also means the keys in preprocess_params)
preprocess_data = ['connect_articles_to_adm2'] 

# preprocess_data = ['acled', 'ucdp', 'population', 'create_neighbor_dicts', 'geo_features', 'election'] #these correspond to the methods in the Preprocessor class (by constructions this also means the keys in preprocess_params)


id_vars = ['shapeID', 'year', 'month']

# Retrieve the EarthData Login token. A fresh token needs to be created every 60 days: https://urs.earthdata.nasa.gov/documentation/for_users/user_token 
load_dotenv(override=True)
EDL_TOKEN = os.getenv('EDL_TOKEN')

urls = {
    # Source: https://disc.gsfc.nasa.gov/datasets/M2TMNXFLX_5.12.4/summary
    'geo_features': 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2_MONTHLY/M2TMNXFLX.5.12.4/'

}

# key is the method in the Preprocessor class, value is the parameters to pass to the method
preprocess_params = {
    'ucdp': None, 
    'acled': None,
    'population': {
        'years': range(1990, 2023)
    },
    'connect_articles_to_adm2': None,
    'election': None,
    'create_neighbor_dicts': {
        'max_neighbor_order':6, 
        'distances_km': [1, 5, 10, 25, 50, 100, 200, 250, 500]
    },
    'static_geo_features': {
        'static_variables': ['open_water', 'barren', 'elevation']},
    'time_variant_geo_features': {
        'nasa_variables': {'PRECTOT':'precipitation', 
                           'TLML':'temperature'},
        'edl_token': EDL_TOKEN 
    }
}

