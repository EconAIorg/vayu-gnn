# EconAI – Pollutant prediction using a spatiotemporal GNN

## Overview

Air pollution poses serious health risks, especially in densely populated regions like Northern India. While monitoring has improved significantly over the past decades, forecasting often remains coarse-grained, obscuring meaningful intra-city variation. The [VAYU project](https://vayu.undp.org.in/hackathon) addresses this by enabling hyperlocal air quality monitoring in Patna and Gurugram.

This repository ([EconAI VAYU-GNN](https://github.com/EconAIorg/vayu-gnn)) showcases a forecasting use case built on VAYU sensor data, designed to predict hyperlocal air pollutant levels up to 8 hours into the future. The goal is to support local government advisory systems with fine-grained, real-time forecasts.

If you would like to read the documentation for the repository, you can visit [this page](https://econaiorg.github.io/vayu-gnn/vayu_gnn.html)

---

## Data pipeline

Our pipeline integrates open-source, interpretable, and predictive features across three categories: environmental, human settlement, and pollution data.

### Meteorological data
We use the [Open Meteo API](https://open-meteo.com/) to gather historical and forecasted weather features for each static sensor location:

- Temperature  
- Humidity  
- Wind direction & speed  
- Precipitation  
- Soil temperature & moisture  
- Air pressure  
- Cloud cover  

Additional geospatial features include:
- Distance to rivers  
- Elevation above sea level  

### Human settlement data
Using the [Global Human Settlement Layer](https://ghsl.jrc.ec.europa.eu/), we include:

- Distance to nearest major road  
- Building area and volume (residential & non-residential)  
- Building density at sensor site and within a 500m radius  

### Pollution monitoring data
The foundation of the pipeline is high-resolution data from VAYU sensors, aggregated to hourly averages. We augment this with broader city-level air quality data via the [OpenWeatherMap Air Pollution API](https://openweathermap.org/api/air-pollution).

All features are collected for both in-city and surrounding nodes to capture spatiotemporal patterns that often influence city-level pollution.

### Preprocessing steps
- Outlier removal  
- Missing value imputation  
- Feature scaling  

Each node-hour instance combines static and dynamic features into an embedding, forming a long-format dataset indexed by node, date, and hour.

---

## Modeling approach

Our forecasting approach utilizes a Graph Neural Network (GNN) model, specifically an RNN encoder followed by a GCN decoder, implemented with the [Torch SpatioTemporal](https://torch-spatiotemporal.readthedocs.io/en/latest/index.html) library.

### High-level architecture:
- **RNN encoder**: Captures temporal patterns and dependencies across time.
- **GCN decoder**: Learns spatial dependencies across nodes, leveraging the graph structure.

Each VAYU sensor is treated as a node in a fully connected graph, with edges weighted by distance. Additional nodes representing neighboring regions are included to capture external influences. Node embeddings initialized from processed data are iteratively refined to optimize prediction accuracy.

We ensure a **train-validation-test split** without data leakage by using only the information known at the time of each prediction.

---

## Results & impact

### Model performance 

Below we display the average Mean Absolute Error (MAE) for each pollutant and city. These performance metrics are averaged across all sensors and forecast horizons (1 to 8 hours ahead), and only include the test set (the last 5% of timesteps).

| City      | PM2.5 | PM10 | NO₂  | CO   | CO₂   | CH₄  |
|-----------|-------|------|------|------|-------|------|
| Patna     | 41.3  | 57.3 | 35.7 | 0.7  | 122.2 | 1.1  |
| Gurugram  | 38.1  | 56.2 | 34.3 | 0.3  | 131.0 | 0.1  |

### Conclusion

By combining GNNs with rich hyperlocal features, we offer a scalable and modular framework that local governments can adapt. Hourly pollutant forecasts are mapped to 7-stage risk levels to enable intuitive decision-making.

This project empowers policymakers, urban planners, and public health officials with actionable insights to mitigate air pollution’s health impacts.

