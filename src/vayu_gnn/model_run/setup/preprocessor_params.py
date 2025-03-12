from vayu_gnn.dbx.dbx_config import dbx_helper, DropboxHelper

preprocess_data = ['open_weather_pollution']
#preprocess_data = ['sensor_data', 'raster_data', 'weather', 'weather_forecast', 'open_weather_pollution']

nodes = {}
nodes['Patna'] = dbx_helper.read_pickle(dbx_helper.clean_input_path, f'node_locations/Patna', f'nodes.pickle')
# nodes['Gurugram'] = dbx_helper.read_pickle(dbx_helper.clean_input_path, f'node_locations/Gurugram', f'nodes.pickle')

cities = ['Patna']
# cities = ['Patna', 'Gurugram']

# key is the method in the Preprocessor class, value is the parameters to pass to the method
preprocess_params = {
    'sensor_data': {
        'cities': ['Patna'],
        'device_types': ['static']
    },
    'raster_data': {
        'cities': ['Patna'],
        'raster_files':['elevation', 
                        'GHS_total_building_area',
                        'GHS_total_building_volume',
                        'GHS_non_res_building_area',
                        'GHS_non_res_building_volume']
    },
    'weather':  {
        'cities': ['Patna']
    },
    'weather_forecast':  {
        'cities': ['Patna']
    },
    'open_weather_pollution':{
        'cities': ['Patna']
    }
}

