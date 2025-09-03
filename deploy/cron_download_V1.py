import os
import requests
from datetime import datetime, timezone
import shutil

# === CONFIGURATION ===
save_dir = "/tmp/ecmwf_forecasts"
os.makedirs(save_dir, exist_ok=True)

# Clear old files first
shutil.rmtree(save_dir)
os.makedirs(save_dir, exist_ok=True)

date = datetime.now(timezone.utc).strftime("%Y%m%d")
run = "00"
base_url = f"https://data.ecmwf.int/forecasts/{date}/{run}z/ifs/0p25/oper"
forecast_hours = [0, 6, 12, 24, 30]  # Example hours

def build_filename(date, run, hour):
    return f"{date}{run}0000-{hour}h-oper-fc.grib2"

def download_file(url, output_path):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Saved {output_path}")
    else:
        print(f"❌ Failed {url} ({r.status_code})")

# === MAIN LOOP ===
for hour in forecast_hours:
    grib_filename = build_filename(date, run, hour)
    url = f"{base_url}/{grib_filename}"
    output_path = os.path.join(save_dir, grib_filename)
    download_file(url, output_path)
