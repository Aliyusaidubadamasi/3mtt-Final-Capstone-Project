"""
Download NASA POWER daily weather data for multiple Nigerian states
and combine into a single CSV for the Crop Yield Estimator project.

No API key required. Just run: python download_nasa_power_nigeria.py
"""

import time
import requests
import pandas as pd

# ---------------------------------------------------------------------
# 1. CONFIG — edit this section for your project
# ---------------------------------------------------------------------

# All 36 states + FCT, using state capital coordinates
STATES = {
    "Abia":         (5.5320, 7.4860),
    "Adamawa":      (9.2035, 12.4954),
    "Akwa Ibom":    (5.0377, 7.9128),
    "Anambra":      (6.2120, 7.0690),
    "Bauchi":       (10.3103, 9.8439),
    "Bayelsa":      (4.9267, 6.2676),
    "Benue":        (7.7322, 8.5391),
    "Borno":        (11.8333, 13.1500),
    "Cross River":  (4.9757, 8.3417),
    "Delta":        (6.1984, 6.7370),
    "Ebonyi":       (6.3249, 8.1137),
    "Edo":          (6.3350, 5.6037),
    "Ekiti":        (7.6211, 5.2214),
    "Enugu":        (6.5244, 7.5086),
    "FCT Abuja":    (9.0765, 7.3986),
    "Gombe":        (10.2897, 11.1673),
    "Imo":          (5.4836, 7.0333),
    "Jigawa":       (11.7565, 9.3389),
    "Kaduna":       (10.5222, 7.4381),
    "Kano":         (12.0022, 8.5920),
    "Katsina":      (12.9908, 7.6018),
    "Kebbi":        (12.4539, 4.1975),
    "Kogi":         (7.8023, 6.7337),
    "Kwara":        (8.4966, 4.5426),
    "Lagos":        (6.6018, 3.3515),
    "Nasarawa":     (8.4939, 8.5163),
    "Niger":        (9.6139, 6.5569),
    "Ogun":         (7.1475, 3.3619),
    "Ondo":         (7.2571, 5.2058),
    "Osun":         (7.7719, 4.5566),
    "Oyo":          (7.3775, 3.9470),
    "Plateau":      (9.8965, 8.8583),
    "Rivers":       (4.8156, 7.0498),
    "Sokoto":       (13.0059, 5.2476),
    "Taraba":       (8.8833, 11.3667),
    "Yobe":         (11.7470, 11.9608),
    "Zamfara":      (12.1704, 6.6641),
}

START_DATE = "20150101"   # YYYYMMDD
END_DATE   = "20241231"   # YYYYMMDD

# Rainfall, avg/min/max temp, humidity, solar radiation, wind speed
PARAMETERS = "PRECTOTCORR,T2M,T2M_MIN,T2M_MAX,RH2M,ALLSKY_SFC_SW_DWN,WS2M"

COMMUNITY = "AG"  # Agroclimatology — units suited for farming applications
OUTPUT_FILE = "nigeria_states_weather_combined.csv"

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# ---------------------------------------------------------------------
# 2. DOWNLOAD LOOP
# ---------------------------------------------------------------------

def fetch_state_weather(state_name: str, lat: float, lon: float, max_retries: int = 4) -> pd.DataFrame:
    """Fetch daily weather data for one location and return a tidy DataFrame.

    Retries with exponential backoff if the connection is reset/dropped,
    which happens occasionally with the NASA POWER API.
    """
    params = {
        "parameters": PARAMETERS,
        "community": COMMUNITY,
        "longitude": lon,
        "latitude": lat,
        "start": START_DATE,
        "end": END_DATE,
        "format": "JSON",
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(BASE_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            break  # success, exit retry loop
        except (requests.exceptions.RequestException, ValueError) as e:
            last_error = e
            wait = 5 * attempt  # 5s, 10s, 15s, 20s
            print(f"  Attempt {attempt}/{max_retries} failed ({e}). Retrying in {wait}s...")
            time.sleep(wait)
    else:
        # all retries exhausted
        raise last_error

    if "properties" not in data:
        print(f"  WARNING: no data returned for {state_name}: {data}")
        return pd.DataFrame()

    param_data = data["properties"]["parameter"]

    # Each parameter comes back as {date: value, ...} — merge them into one table
    df = pd.DataFrame(param_data)
    df.index.name = "date"
    df = df.reset_index()

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["state"] = state_name
    df["latitude"] = lat
    df["longitude"] = lon

    return df


def main():
    all_frames = []

    # If a previous run already succeeded for some states, don't re-download them
    already_done = set()
    try:
        existing = pd.read_csv(OUTPUT_FILE)
        already_done = set(existing["state"].unique())
        all_frames.append(existing)
        if already_done:
            print(f"Found existing '{OUTPUT_FILE}' with data for: {sorted(already_done)}")
            print("These will be skipped. Delete the file first if you want a full re-download.\n")
    except FileNotFoundError:
        pass

    for state_name, (lat, lon) in STATES.items():
        if state_name in already_done:
            continue

        print(f"Fetching weather data for {state_name} ({lat}, {lon})...")
        try:
            df = fetch_state_weather(state_name, lat, lon)
            if not df.empty:
                all_frames.append(df)
                print(f"  OK: {len(df)} rows")
        except requests.exceptions.RequestException as e:
            print(f"  FAILED after retries for {state_name}: {e}")

        # Be polite to the API between requests
        time.sleep(2)

    if not all_frames:
        print("No data was downloaded. Check your internet connection or parameters.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["state", "date"] if "date" in combined.columns else None)

    # NASA POWER uses -999 as a missing-value flag — convert to NaN
    combined = combined.replace(-999, pd.NA)

    # Friendlier column names
    combined = combined.rename(columns={
        "PRECTOTCORR": "rainfall_mm",
        "T2M": "avg_temp_c",
        "T2M_MIN": "min_temp_c",
        "T2M_MAX": "max_temp_c",
        "RH2M": "humidity_pct",
        "ALLSKY_SFC_SW_DWN": "solar_radiation",
        "WS2M": "wind_speed_ms",
    })

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone. Combined data for {len(STATES)} states saved to '{OUTPUT_FILE}'")
    print(f"Total rows: {len(combined)}")


if __name__ == "__main__":
    main()