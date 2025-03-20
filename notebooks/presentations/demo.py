import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from branca.colormap import linear

# ----- Generate Random Data -----

# Nodes around central Patna
np.random.seed(42)
n_nodes = 30
center_lat, center_lon = 25.5941, 85.1376
nodes = pd.DataFrame({
    'node_id': [f'Node {i+1}' for i in range(n_nodes)],
    'lat': center_lat + np.random.uniform(-0.05, 0.05, n_nodes),
    'lon': center_lon + np.random.uniform(-0.05, 0.05, n_nodes)
})

# Dates, hours, prediction horizons, pollutants
dates = pd.date_range('2025-03-01', periods=7)
hours = np.arange(24)
horizons = np.arange(1, 9)
pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'CO2', 'CH4']

# Create random pollution data
multi_idx = pd.MultiIndex.from_product(
    [dates, hours, horizons, pollutants, nodes['node_id']],
    names=['date', 'hour', 'horizon', 'pollutant', 'node_id']
)
pollution_df = pd.DataFrame(index=multi_idx).reset_index()
pollution_df['value'] = np.random.uniform(10, 200, len(pollution_df))

# ----- Streamlit App -----

st.title("Patna Pollution Interactive Demo")
st.sidebar.markdown("""
### How to Use This App

1. **Select Date**: Choose a date from the dropdown to view pollution data for that specific day.
2. **Select Hour**: Pick an hour of the day (0-23) to see the pollution levels at that time.
3. **Prediction Horizon**: Use the slider to select how many hours ahead you want to predict pollution levels (1 to 8 hours).
4. **Select Pollutant**: Choose the type of pollutant you want to analyze (e.g., PM2.5, PM10, NO2, CO, CO2, CH4).

This will filter the data accordingly and display the pollution levels on the map and in the table.
""", unsafe_allow_html=True)

# Create layout for controls
col1, col2, col3, col4 = st.columns(4)

with col1:
    selected_date = st.selectbox("Select Date", dates.strftime('%Y-%m-%d'))

with col2:
    selected_hour = st.selectbox("Select Hour", hours)

with col3:
    selected_horizon = st.slider("Prediction Horizon (hours)", 1, 8, 1)

with col4:
    selected_pollutant = st.selectbox("Select Pollutant", pollutants)

# Filter data
filtered_df = pollution_df[
    (pollution_df['date'] == pd.Timestamp(selected_date)) &
    (pollution_df['hour'] == selected_hour) &
    (pollution_df['horizon'] == selected_horizon) &
    (pollution_df['pollutant'] == selected_pollutant)
]

# Merge data
plot_data = nodes.merge(filtered_df, on='node_id')

# ----- Folium Map -----

# Define color mappings for pollutants based on health guidelines
color_maps = {
    'PM2.5': linear.YlGnBu_09.scale(0, 200),
    'PM10': linear.YlOrRd_09.scale(0, 200),
    'NO2': linear.RdYlGn_09.scale(0, 200),
    'CO': linear.PuBu_09.scale(0, 200),
    'CO2': linear.PuRd_09.scale(0, 200),
    'CH4': linear.Greens_09.scale(0, 200),
}

# Create Folium map
m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB positron')

# Color scale based on pollution values
colormap = color_maps[selected_pollutant]

# Add nodes
for _, row in plot_data.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=12,
        color=colormap(row['value']),
        fill=True,
        fill_opacity=0.8,
        popup=f"{row['node_id']}: {row['value']:.1f}"
    ).add_to(m)

# Display map
st.subheader(f"Pollution Levels in Patna - {selected_pollutant}")
st_folium(m, width=1000, height=500)

# Display table
st.dataframe(plot_data[['node_id', 'lat', 'lon', 'value']].sort_values('value', ascending=False))