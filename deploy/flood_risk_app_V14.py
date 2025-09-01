import streamlit as st
import pandas as pd
import numpy as np
import os
import pygrib
from datetime import timedelta
import folium
from streamlit.components.v1 import html
from geopy.geocoders import Nominatim
import plotly.express as px

# ================================
# Constants
# ================================
SAVE_DIR = "D:/deployment code/bhutan_app/"
COUNTRY = "Bhutan"

# ================================
# Page Configuration
# ================================
st.set_page_config(layout="wide")

# ================================
# Header Section
# ================================
st.markdown("""
<div style="background-color:white; padding: 10px 5px 5px 5px; border-bottom: 2px solid #e6e6e6;">
    <h1 style="text-align: left; font-size: 60px; color: black; font-style: Calibri;">
        འབྲུག་ཆུ་རུད་ཀྱི་རྐྱེན་ངན་ཉེན་བརྡའི་དྲ་ངོས།<br>
        <span style="font-size: 30px;">Bhutan Flood Disaster Alert Portal</span>
    </h1>
</div>
""", unsafe_allow_html=True)

# Live Alert Banner
st.markdown("""
<div style="background-color: #2a6aad; border-radius: 6px 6px 0 0; padding: 5px 15px; margin: 5px 0 0 0; color: white;">
    <div style="font-size: 18px; font-weight: bold; animation: blink-text 1.2s infinite;">
        LIVE:
    </div>
</div>
<div style="background-color: #ffcccc; border-left: 5px solid red; padding: 10px; margin: 0 0 10px 0; overflow: hidden; border-radius: 0 0 6px 6px;">
    <div style="display: inline-block; white-space: nowrap; animation: scroll-left 15s linear infinite;
        font-weight: bold; color: red; font-size: 18px;">
        🚨 Orange Alert| Very Heavy Rainfall | Paro | 01 Sep 2025 ------- Stay Alert and follow protocols. 🚨
    </div>
</div>
<style>
    @keyframes scroll-left {0% {transform: translateX(100%);}100% {transform: translateX(-100%);}}
    @keyframes blink-text {0% {opacity: 1;}50% {opacity: 0;}100% {opacity: 1;}}
</style>
""", unsafe_allow_html=True)

# ================================
# Sidebar
# ================================
with st.sidebar:
    st.header("Navigation")
    st.markdown("- 📘 [Guide](#)")
    st.markdown("- 🧠 [GitHub Repo](https://github.com/name_123)")
    st.markdown("---")
    st.markdown("""
        <div>
            <a href="https://facebook.com" target="_blank">📘 Facebook</a><br>
            <a href="https://twitter.com" target="_blank">🐦 X (Twitter)</a><br>
            <a href="https://youtube.com" target="_blank">📺 YouTube</a><br>
            <a href="https://instagram.com" target="_blank">📷 Instagram</a>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("🇧🇹 Powered by Omdena Bhutan<br>Version 1.0<br>Last Updated On: 24 July 2025", unsafe_allow_html=True)

# ================================
# CSS Styling
# ================================
st.markdown("""
<style>
    [data-testid="stHeader"] {background-color: white !important; height: 0px !important;}
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main {
        background-color: white !important; color: black !important;
    }
    [data-testid="stSidebar"] {background-color: #2a6aad !important;}
    [data-testid="stSidebar"] * {color: white !important; font-weight: bold !important;}
    a {text-decoration: none !important;}
    div[data-baseweb="input"] > div:first-child label, 
    div[data-baseweb="textarea"] > div:first-child label {
        color: black !important;
        font-weight: bold !important;
        font-size: 16px !important;
    }
</style>
""", unsafe_allow_html=True)



# ================================
# Tabs
# ================================
tab_weather_forecast, = st.tabs(["Weather Forecast"])

# ================================
# Geocoding function
# ================================
@st.cache_data
def get_coords_from_location(locality, gewog_thromde, dzongkhag):
    try:
        geolocator = Nominatim(user_agent="bhutan_flood_app")
        query = f"{locality}, {gewog_thromde}, {dzongkhag}, {COUNTRY}"
        loc = geolocator.geocode(query, timeout=10)
        if loc:
            return loc.latitude, loc.longitude, loc.address
        else:
            return None, None, None
    except Exception:
        return None, None, None

# ================================
# Bilinear Interpolation (spatial Interpolation)
# ================================
def bilinear_interpolate(lat_target, lon_target, lats, lons, data):
    lats = np.array(lats)
    lons = np.array(lons)
    data = np.array(data)

    if lats[0,0] > lats[-1,0]:
        lats = np.flipud(lats)
        data = np.flipud(data)

    if np.any(lons > 180):
        lon_target = (lon_target + 360) % 360

    lat_idx = np.searchsorted(lats[:,0], lat_target) - 1
    lon_idx = np.searchsorted(lons[0,:], lon_target) - 1

    lat_idx = np.clip(lat_idx, 0, lats.shape[0]-2)
    lon_idx = np.clip(lon_idx, 0, lons.shape[1]-2)

    lat1, lat2 = lats[lat_idx,0], lats[lat_idx+1,0]
    lon1, lon2 = lons[0,lon_idx], lons[0,lon_idx+1]
    Q11 = data[lat_idx, lon_idx]
    Q21 = data[lat_idx, lon_idx+1]
    Q12 = data[lat_idx+1, lon_idx]
    Q22 = data[lat_idx+1, lon_idx+1]

    if lon2 == lon1 or lat2 == lat1:
        return Q11

    f_lon = (lon_target - lon1)/(lon2-lon1)
    f_lat = (lat_target - lat1)/(lat2-lat1)

    return Q11*(1-f_lon)*(1-f_lat) + Q21*f_lon*(1-f_lat) + Q12*(1-f_lon)*f_lat + Q22*f_lon*f_lat

# ================================
# Extract weather variables from GRIB
# ================================
def get_weather_variables(grib_file, lat_target, lon_target):
    grbs = pygrib.open(grib_file)
    variables = {}
    utc_time = None

    for msg in grbs:
        name = msg.name.lower()
        if name in ['2 metre temperature', 'runoff', '2 metre dewpoint temperature', 'total precipitation', '2 metre relative humidity']:
            data_vals, lats, lons = msg.data()
            value = bilinear_interpolate(lat_target, lon_target, lats, lons, data_vals)

            # Unit conversion
            if name == '2 metre temperature' or name == '2 metre dewpoint temperature':
                units = getattr(msg, "units", "").strip().lower()
                if not units or units.startswith("k"):
                    value -= 273.15

            if name == 'total precipitation':
                value *= 1000  # m -> mm

            if name == '2 metre relative humidity':
                value = float(value)

            variables[name] = value
            utc_time = msg.validDate

    grbs.close()
    if utc_time:
        local_time = utc_time + timedelta(hours=6)
    else:
        local_time = None
    return variables, local_time

# ================================
# Weather Forecast Tab
# ================================
with tab_weather_forecast:
    # User Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<span style="color:black; font-weight:bold; font-size:18px;">Specify Locality</span>', unsafe_allow_html=True)
        locality = st.text_input("Locality", value="Changzamtog", label_visibility='collapsed')
    with col2:
        st.markdown('<span style="color:black; font-weight:bold; font-size:18px;">Specify Gewog/Thromde</span>', unsafe_allow_html=True)
        gewog_thromde = st.text_input("Gewog or Thromde", value="Thimphu Thromde", label_visibility='collapsed')
    with col3:
        st.markdown('<span style="color:black; font-weight:bold; font-size:18px;">Specify Dzongkhag</span>', unsafe_allow_html=True)
        dzongkhag = st.text_input("Dzongkhag", value="Thimphu", label_visibility='collapsed')

    if st.button("Get Forecast") or 'df_forecast' not in st.session_state:
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

        times_local = []
        all_vars = {
            'Temperature': [], 'Runoff': [], 'Dewpoint Temp': [],
            'Humidity (%)': [], 'Precipitation (mm)': []
        }

        for file_name in forecast_files:
            file_path = os.path.join(SAVE_DIR, file_name)
            if os.path.exists(file_path):
                vars_dict, local_time = get_weather_variables(file_path, lat_user, lon_user)
                if local_time:
                    times_local.append(local_time)
                    all_vars['Temperature'].append(vars_dict.get('2 metre temperature', np.nan))
                    all_vars['Runoff'].append(vars_dict.get('runoff', np.nan))
                    all_vars['Dewpoint Temp'].append(vars_dict.get('2 metre dewpoint temperature', np.nan))
                    all_vars['Humidity (%)'].append(vars_dict.get('2 metre relative humidity', np.nan))
                    all_vars['Precipitation (mm)'].append(vars_dict.get('total precipitation', np.nan))

        if not times_local:
            st.error("No forecast files found for the selected location.")
            st.stop()

        # Save to session state to avoid reloading
        st.session_state.df_forecast = pd.DataFrame({'Local Time': times_local, **all_vars})
        st.session_state.lat_user = lat_user
        st.session_state.lon_user = lon_user
        st.session_state.address = address

    # Use cached data
    df_forecast = st.session_state.df_forecast
    lat_user = st.session_state.lat_user
    lon_user = st.session_state.lon_user
    address = st.session_state.address

    # Map display (added _repr_html() function so that zoom in/out doesn't make the entire app reload)
    col1_map, col2_chart = st.columns(2)
    with col1_map:
        st.markdown(
            f'<h4 style="font-size:24px; font-weight:bold;">Selected Location</h4>',
            unsafe_allow_html=True
        )
        m = folium.Map(location=[lat_user, lon_user], zoom_start=12, tiles="OpenStreetMap")
        folium.CircleMarker(
            location=[lat_user, lon_user],
            radius=6,
            popup=address,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.8
        ).add_to(m)
        html(m._repr_html_(), height=400)

    # ===============================
    # Chart with variable dropdown (added session_state to prevent the entire app from reloading)
    # ===============================
    with col2_chart:
        #st.subheader(f"📈 Weather Forecast in {locality}, {gewog_thromde}, {dzongkhag}")
        st.markdown(
            f'<h4 style="font-size:24px; font-weight:bold;">Weather Forecast in {locality}, {gewog_thromde}, {dzongkhag}</h4>',
            unsafe_allow_html=True
        )

        # Session state for selected variable
        if 'selected_variable' not in st.session_state:
            st.session_state.selected_variable = 'Temperature'

        variable_options = ['Temperature', 'Runoff', 'Dewpoint Temp', 'Humidity (%)', 'Precipitation (mm)']
        selected_variable = st.selectbox(
            "Select Variable to Plot",
            variable_options,
            index=variable_options.index(st.session_state.selected_variable)
        )
        st.session_state.selected_variable = selected_variable

        # Plot
        fig = px.line(df_forecast, x='Local Time', y=selected_variable, markers=True,
                      title=f"{selected_variable} Forecast")
        fig.update_traces(marker=dict(size=8, color='orange'))
        fig.update_layout(height=300, margin=dict(l=10,r=10,t=40,b=10),
                          plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_title="Time", yaxis_title=selected_variable)
        st.plotly_chart(fig, use_container_width=True)

        # Forecast details
        st.markdown("**Forecast Details:**")
        df_display = df_forecast.copy()
        df_display['Date'] = df_display['Local Time'].dt.strftime('%d %b %Y')
        df_display['Time'] = df_display['Local Time'].dt.strftime('%H:%M')
        df_display = df_display[['Date', 'Time', 'Temperature', 'Runoff', 'Dewpoint Temp', 'Humidity (%)', 'Precipitation (mm)']]

        # Round numeric values
        df_display['Temperature'] = df_display['Temperature'].round(1)
        df_display['Runoff'] = df_display['Runoff'].round(2)
        df_display['Dewpoint Temp'] = df_display['Dewpoint Temp'].round(1)
        df_display['Humidity (%)'] = df_display['Humidity (%)'].round(1)
        df_display['Precipitation (mm)'] = df_display['Precipitation (mm)'].round(1)

        # Display table
        st.dataframe(df_display, use_container_width=True)