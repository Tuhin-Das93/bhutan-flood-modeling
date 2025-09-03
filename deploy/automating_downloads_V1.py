import streamlit as st
import os
import requests
from datetime import datetime, timezone

# === CONFIGURATION ===
if os.name == "nt":  # Windows local
    save_dir = os.path.join(os.getcwd(), "ecmwf_forecasts")
else:  # Render (Linux container)
    save_dir = "/tmp/ecmwf_forecasts"

os.makedirs(save_dir, exist_ok=True)

date = datetime.now(timezone.utc).strftime("%Y%m%d")
run = "00"
base_url = f"https://data.ecmwf.int/forecasts/{date}/{run}z/ifs/0p25/oper"
forecast_hours = [0, 6, 12, 24]  # Example hours

def build_filename(date, run, hour):
    return f"{date}{run}0000-{hour}h-oper-fc.grib2"

def download_file(url, output_path):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    return False

# === STREAMLIT UI ===
st.title("ECMWF Forecast Downloader")

if st.button("Download Now"):
    for hour in forecast_hours:
        grib_filename = build_filename(date, run, hour)
        url = f"{base_url}/{grib_filename}"
        output_path = os.path.join(save_dir, grib_filename)

        if not os.path.exists(output_path):
            success = download_file(url, output_path)
            if success:
                st.success(f"Downloaded {grib_filename}")
            else:
                st.error(f"Failed {grib_filename}")
        else:
            st.warning(f"Already exists: {grib_filename}")

# Show downloaded files
st.subheader("Downloaded Files")
for file in os.listdir(save_dir):
    st.write(file)
