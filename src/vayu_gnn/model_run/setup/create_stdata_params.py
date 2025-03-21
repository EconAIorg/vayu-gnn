
params = {}

# Cities:
params['cities'] = ['Patna', 'Gurugram']

# Sources for creating single df
params['sources'] = ["sensor_data","pollution", "elevation", "weather", "weather_forecast", "settlements"]
params['sources'] = ["sensor_data","pollution", "weather",]


params['skip_length'] = [3, 24]
params['n_skip_connections'] = [2, 2]

# Params for SpatioTemporalDataset
params['horizon'] = 8
params['window'] = 24
params['stride'] = 1
 
# Train test split
params['validation_frac'] = .05
params['test_frac'] = .05

