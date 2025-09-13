# =========================================
# Import necessary libraries - Section 1
# =========================================
import streamlit as st
import pandas as pd
import numpy as np
import os
import pygrib
from datetime import timedelta
import folium
from streamlit.components.v1 import html
from geopy.distance import distance
import plotly.express as px
import requests

# ==============================================================
# Defining constants used throughout the application - Section 2
# ==============================================================
SAVE_DIR = "D:/deployment code/bhutan_app/"
COUNTRY = "Bhutan"

# ===================================================================================
# List of top 10 cities with their coordinates for display in the sidebar - Section 3
# ===================================================================================
TOP_CITIES = [
    {"name": "Thimphu", "lat": 27.4728, "lon": 89.6393},
    {"name": "Phuntsholing", "lat": 26.8574, "lon": 89.3886},
    {"name": "Punakha", "lat": 27.5833, "lon": 89.8667},
    {"name": "Trongsa", "lat": 27.4833, "lon": 90.0000},
    {"name": "Paro", "lat": 27.4305, "lon": 89.4134},
    {"name": "Wangdue Phodrang", "lat": 27.4167, "lon": 89.9000},
    {"name": "Trashigang", "lat": 27.3333, "lon": 91.5000},
    {"name": "Georgetown", "lat": 27.4500, "lon": 89.6500},
    {"name": "Samdrup Jongkhar", "lat": 26.8000, "lon": 91.5000},
    {"name": "Bumthang", "lat": 27.5000, "lon": 90.7500}
]

# ================================================================
# Set streamlit page layout to wide for better display - Section 4
# ================================================================
st.set_page_config(layout="wide")

# =====================================================
# Define functions used in the application - Section 5
# =====================================================

# Function A: To perform bilinear interpolation for smoother weather data extraction
def bilinear_interpolate(lat_target, lon_target, lats, lons, data):
    lats = np.array(lats)
    lons = np.array(lons)
    data = np.array(data)
    if lats[0, 0] > lats[-1, 0]:
        lats = np.flipud(lats)
        data = np.flipud(data)
    if np.any(lons > 180):
        lon_target = (lon_target + 360) % 360
    lat_idx = np.searchsorted(lats[:, 0], lat_target) - 1
    lon_idx = np.searchsorted(lons[0, :], lon_target) - 1
    lat_idx = np.clip(lat_idx, 0, lats.shape[0]-2)
    lon_idx = np.clip(lon_idx, 0, lons.shape[1]-2)
    lat1, lat2 = lats[lat_idx, 0], lats[lat_idx+1, 0]
    lon1, lon2 = lons[0, lon_idx], lons[0, lon_idx+1]
    Q11 = data[lat_idx, lon_idx]
    Q21 = data[lat_idx, lon_idx+1]
    Q12 = data[lat_idx+1, lon_idx]
    Q22 = data[lat_idx+1, lon_idx+1]
    if lon2 == lon1 or lat2 == lat1:
        return Q11
    f_lon = (lon_target - lon1)/(lon2 - lon1)
    f_lat = (lat_target - lat1)/(lat2 - lat1)
    return Q11*(1-f_lon)*(1-f_lat) + Q21*f_lon*(1-f_lat) + Q12*(1-f_lon)*f_lat + Q22*f_lon*f_lat

# Function B: To extract relevant weather variables from a GRIB2 file for a specific location
def get_weather_variables(grib_file, lat_target, lon_target):
    grbs = pygrib.open(grib_file)
    variables = {}
    utc_time = None
    for msg in grbs:
        name = msg.name.lower()
        if name in ['2 metre temperature', 'total precipitation']:
            data_vals, lats, lons = msg.data()
            value = bilinear_interpolate(lat_target, lon_target, lats, lons, data_vals)
            if name == '2 metre temperature':
                units = getattr(msg, "units", "").strip().lower()
                if not units or units.startswith("k"):
                    value -= 273.15
            if name == 'total precipitation':
                value *= 1000
            variables[name] = value
            utc_time = msg.validDate
    grbs.close()
    if utc_time:
        local_time = utc_time + timedelta(hours=6)
    else:
        local_time = None
    return variables, local_time

# Function C: To get geographical coordinates from a location description using geopy
def get_coords_from_location(locality, gewog_thromde, dzongkhag):
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="bhutan_weather_app")
        query = f"{locality}, {gewog_thromde}, {dzongkhag}, {COUNTRY}"
        loc = geolocator.geocode(query, timeout=10)
        if loc:
            return loc.latitude, loc.longitude, loc.address
        else:
            return None, None, None
    except Exception:
        return None, None, None

# Function D: To fetch nearby places (city, town, village) using OpenStreetMap's Overpass API
def fetch_nearby_places(lat, lon, radius_km=10, max_results=10):
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:25];
    (
      node["place"~"city|town|village|hamlet"](around:{radius_km*1000},{lat},{lon});
    );
    out body {max_results};
    """
    try:
        response = requests.post(overpass_url, data=query, timeout=30)
        data = response.json()
        places = []
        for element in data.get('elements', []):
            name = element.get('tags', {}).get('name')
            place_type = element.get('tags', {}).get('place')
            el_lat = element.get('lat')
            el_lon = element.get('lon')
            if name and place_type and el_lat and el_lon:
                dist = distance((lat, lon), (el_lat, el_lon)).km
                places.append({
                    "name": name,
                    "type": place_type,
                    "lat": el_lat,
                    "lon": el_lon,
                    "dist": dist
                })
        places_sorted = sorted(places, key=lambda x: x['dist'])
        return places_sorted[:max_results]
    except Exception as e:
        return []

# =================================
# Streamlit App Layout - Section 6
# =================================
st.set_page_config(layout="wide")

# Header
st.markdown("""
<div style="background-color:white; padding: 10px 5px 5px 5px; border-bottom: 2px solid #e6e6e6;">
    <h1 style="text-align: left; font-size: 60px; color: black; font-style: Calibri;">
        འབྲུག་ཆུ་རུད་ཀྱི་རྐྱེན་ངན་ཉེན་བརྡའི་དྲ་ངོས།<br>
        <span style="font-size: 30px;">Bhutan Weather Portal</span>
    </h1>
</div>
""", unsafe_allow_html=True)

# ==============================================================
# Sidebar section displaying weather with Top Cities - Section 7
# ==============================================================
with st.sidebar:
    st.header("Top 10 Cities Weather")
    grib_files = sorted([f for f in os.listdir(SAVE_DIR) if f.endswith(".grib2")])
    if grib_files:
        latest_file = grib_files[-1]
        file_path = os.path.join(SAVE_DIR, latest_file)
        for city in TOP_CITIES:
            name = city["name"]
            lat = city["lat"]
            lon = city["lon"]
            try:
                variables, local_time = get_weather_variables(file_path, lat, lon)
                temp = variables.get('2 metre temperature', None)
                precip = variables.get('total precipitation', None)
                if temp is not None and precip is not None:
                    st.markdown(f"**{name}**")
                    st.markdown(f"🌡 Temperature: {temp:.1f} °C")
                    st.markdown(f"💧 Precipitation: {precip:.1f} mm")
                    st.markdown("---")
            except Exception:
                st.error(f"Error loading data for {name}")
    else:
        st.error("No GRIB files found in SAVE_DIR.")

# ==================================================================================
# Create the Weather Forecast tab where users can input location details - Section 8
# ==================================================================================
tab_weather_forecast, = st.tabs(["Weather Forecast"])

with tab_weather_forecast:

    # Input fields for locality, gewog/thromde and Dzongkhag
    col1, col2, col3 = st.columns(3)
    with col1:
        locality = st.text_input("Locality", value="Changzamtog")
    with col2:
        gewog_thromde = st.text_input("Gewog or Thromde", value="Thimphu Thromde")
    with col3:
        dzongkhag = st.text_input("Dzongkhag", value="Thimphu")

    # Pricess inputs when "Get Forecast" button is clicked
    if st.button("Get Forecast") or 'df_forecast' not in st.session_state:
        # Load forecast files and extract data
        lat_user, lon_user, address = get_coords_from_location(locality, gewog_thromde, dzongkhag)
        if lat_user is None or lon_user is None:
            st.error("Unable to find coordinates for the provided location.")
            st.stop()

        forecast_files = [
            "20250805000000-0h-oper-fc.grib2",
            "20250805000000-6h-oper-fc.grib2",
            "20250805000000-12h-oper-fc.grib2",
            "20250805000000-24h-oper-fc.grib2",
            "20250805000000-30h-oper-fc.grib2",
            "20250805000000-36h-oper-fc.grib2",
            "20250805000000-42h-oper-fc.grib2",
            "20250805000000-48h-oper-fc.grib2"
        ]

        # Store results in session state
        times_local = []
        all_vars = {'Temperature': []}

        for file_name in forecast_files:
            file_path = os.path.join(SAVE_DIR, file_name)
            if os.path.exists(file_path):
                vars_dict, local_time = get_weather_variables(file_path, lat_user, lon_user)
                if local_time:
                    times_local.append(local_time)
                    all_vars['Temperature'].append(vars_dict.get('2 metre temperature', np.nan))

        if not times_local:
            st.error("No forecast files found for the selected location.")
            st.stop()

        st.session_state.df_forecast = pd.DataFrame({'Local Time': times_local, **all_vars})
        st.session_state.lat_user = lat_user
        st.session_state.lon_user = lon_user
        st.session_state.address = address

    # Retrieve data from session state
    df_forecast = st.session_state.df_forecast
    lat_user = st.session_state.lat_user
    lon_user = st.session_state.lon_user
    address = st.session_state.address

    # Split into two columns: one for map and nearby places, the other for the forecast chart
    col1_map, col2_chart = st.columns(2)
    
    # Map and nearby locations display
    with col1_map:
        # Create map centered on user's location
        st.markdown('<h4 style="font-size:24px; font-weight:bold;">Selected Location</h4>', unsafe_allow_html=True)
        
        m = folium.Map(location=[lat_user, lon_user], zoom_start=10, tiles="OpenStreetMap")
        folium.Circle(
            location=[lat_user, lon_user],
            radius=10000,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.2,
            popup="10 km radius"
        ).add_to(m)
        html(m._repr_html_(), height=400)

        avg_temp = df_forecast['Temperature'].mean()
        st.markdown(f'<h4 style="font-size:20px; font-weight:bold; color:green;">Average Temperature: {avg_temp:.1f} °C</h4>', unsafe_allow_html=True)

        st.markdown('<h4 style="font-size:20px; font-weight:bold; color:black;">Nearby Locations</h4>', unsafe_allow_html=True)

        # Display nearby places with weather data
        places = fetch_nearby_places(lat_user, lon_user, radius_km=10, max_results=10)
        if places:
            file_path = os.path.join(SAVE_DIR, sorted([f for f in os.listdir(SAVE_DIR) if f.endswith(".grib2")])[-1])
            for place in places:
                try:
                    variables, _ = get_weather_variables(file_path, place['lat'], place['lon'])
                    temp = variables.get('2 metre temperature', None)
                    if temp is not None:
                        st.markdown(f"**{place['name']}** ({place['type']}, {place['dist']:.1f} km away): 🌡 {temp:.1f} °C")
                except Exception:
                    continue
        else:
            st.markdown("No nearby geographic locations found within 10 km.")
    
    # Forecast chart display
    with col2_chart:
        # Plot temperature forecast in a line chart using plotly
        st.markdown('<h4 style="font-size:24px; font-weight:bold;">Weather Forecast</h4>', unsafe_allow_html=True)
        if not df_forecast.empty:
            fig = px.line(df_forecast, x='Local Time', y='Temperature', title='Temperature Forecast', markers=True)
            st.plotly_chart(fig, use_container_width=True)
