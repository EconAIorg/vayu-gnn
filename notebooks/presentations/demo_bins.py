import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from branca.colormap import StepColormap

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
pollution_df['value'] = np.random.uniform(10, 1000, len(pollution_df))

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

# Add sidebar legend
st.sidebar.markdown("### Air Quality Legend")
st.sidebar.markdown("""
- <span style='background-color:#1a9850; display:inline-block; width: 12px; height: 12px; border-radius: 50%;'></span> **Good**  
- <span style='background-color:#66bd63; display:inline-block; width: 12px; height: 12px; border-radius: 50%;'></span> **Fair**  
- <span style='background-color:#fee08b; display:inline-block; width: 12px; height: 12px; border-radius: 50%;'></span> **Moderate**  
- <span style='background-color:#fdae61; display:inline-block; width: 12px; height: 12px; border-radius: 50%;'></span> **Poor**  
- <span style='background-color:#f46d43; display:inline-block; width: 12px; height: 12px; border-radius: 50%;'></span> **Very Poor**  
- <span style='background-color:#d73027; display:inline-block; width: 12px; height: 12px; border-radius: 50%;'></span> **Extremely Poor**  
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

# ----- Discrete Bins Setup -----
# Update these bins and colors as needed for each pollutant
pollutant_bins = {
    'PM2.5': {
        'bins': [0, 10, 20, 25, 50, 75, float('inf')],
        'colors': ['#1a9850', '#66bd63', '#fee08b', '#fdae61', '#f46d43', '#d73027']
    },
    'PM10': {
        'bins': [0, 20, 40, 50, 100, 150, float('inf')],
        'colors': ['#1a9850', '#66bd63', '#fee08b', '#fdae61', '#f46d43', '#d73027']
    },
    'NO2': {
        'bins': [0, 40, 90, 120, 230, 340, float('inf')],
        'colors': ['#1a9850', '#66bd63', '#fee08b', '#fdae61', '#f46d43', '#d73027']
    },
    'CO': {
        'bins': [0, 4, 9, 15, 30, 50, float('inf')],
        'colors': ['#1a9850', '#66bd63', '#fee08b', '#fdae61', '#f46d43', '#d73027']
    },
    'CO2': {
        'bins': [0, 500, 1000, 2000, 5000, 10000, float('inf')],
        'colors': ['#1a9850', '#66bd63', '#fee08b', '#fdae61', '#f46d43', '#d73027']
    },
    'CH4': {
        'bins': [0, 5, 10, 50, 100, 200, float('inf')],
        'colors': ['#1a9850', '#66bd63', '#fee08b', '#fdae61', '#f46d43', '#d73027']
    }
}

# ----- Folium Map -----

# Create Folium map
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='CartoDB positron')

# Build the StepColormap for the selected pollutant
bin_info = pollutant_bins[selected_pollutant]

# Replace infinite bin with a finite maximum value if necessary
if bin_info['bins'][-1] == float('inf'):
    # Use the maximum observed value from the data or a default scaling
    if not plot_data.empty:
        finite_max = max(plot_data['value'].max(), bin_info['bins'][-2] * 1.5)
    else:
        finite_max = bin_info['bins'][-2] * 1.5
    bins_for_colormap = bin_info['bins'][:-1] + [finite_max]
else:
    bins_for_colormap = bin_info['bins']

colormap = StepColormap(
    colors=bin_info['colors'],
    index=bins_for_colormap,
    vmin=bins_for_colormap[0],
    vmax=bins_for_colormap[-1],
    caption=f"{selected_pollutant} Levels"
)

# Add nodes with discrete colors
for _, row in plot_data.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=12,
        color=colormap(row['value']),
        fill=True,
        fill_opacity=0.8,
        popup=f"{row['node_id']}: {row['value']:.1f}"
    ).add_to(m)

# Display map in Streamlit
st.subheader(f"Pollution Levels in Patna - {selected_pollutant}")
st_folium(m, width=1000, height=500)

# Display table
st.dataframe(plot_data[['node_id', 'lat', 'lon', 'value']].sort_values('value', ascending=False))