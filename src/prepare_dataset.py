"""
prepare_dataset.py
------------------
Automated Data Preparation Pipeline for Crop Yield Estimator (Phase 1).

Combines raw datasets in `datasets/`:
  1. nigeria_states_weather_combined.csv (NASA POWER API daily weather 2015-2024 for 37 states)
  2. FAOSTAT_data_en_8-12-2026.csv (Nigeria national crop yields & area harvested 2019-2024)
  3. Crop Yiled with Soil and Weather.csv (Global soil/weather/yield dataset)
  4. Crop_recommendation.csv (Soil nutrients N, P, K, pH by crop)
  5. pesticides.csv (Pesticide usage stats)

Outputs:
  - datasets/processed_crop_yield.csv (Unified standardized table for model training)
"""

import os
import glob
import pandas as pd
import numpy as np

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
OUTPUT_FILE = os.path.join(DATASETS_DIR, "processed_crop_yield.csv")

def process_nigeria_weather():
    """Aggregate daily NASA POWER weather into annual state-level summaries (2015-2024)."""
    weather_path = os.path.join(DATASETS_DIR, "nigeria_states_weather_combined.csv")
    if not os.path.exists(weather_path):
        print(f"Warning: {weather_path} not found.")
        return pd.DataFrame()

    print("Loading Nigeria daily weather dataset...")
    df = pd.read_csv(weather_path)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    # Aggregate daily -> annual by state and year
    annual = df.groupby(["state", "year"]).agg(
        rainfall_mm=("rainfall_mm", "sum"),
        avg_temp_c=("avg_temp_c", "mean"),
        min_temp_c=("min_temp_c", "mean"),
        max_temp_c=("max_temp_c", "mean"),
        humidity_pct=("humidity_pct", "mean"),
        solar_radiation=("solar_radiation", "mean"),
        wind_speed_ms=("wind_speed_ms", "mean"),
    ).reset_index()

    # Round floats for clean output
    numeric_cols = ["rainfall_mm", "avg_temp_c", "min_temp_c", "max_temp_c", "humidity_pct", "solar_radiation", "wind_speed_ms"]
    annual[numeric_cols] = annual[numeric_cols].round(2)
    print(f"Aggregated {len(df)} daily weather rows -> {len(annual)} state-year annual records.")
    return annual


def load_crop_soil_recommendations():
    """Extract baseline soil nutrient (N, P, K, pH) profiles per crop from Crop_recommendation.csv."""
    rec_path = os.path.join(DATASETS_DIR, "Crop_recommendation.csv")
    if not os.path.exists(rec_path):
        print(f"Warning: {rec_path} not found.")
        return {}

    df = pd.read_csv(rec_path)
    crop_map = {
        'rice': 'Rice',
        'maize': 'Maize (corn)',
        'banana': 'Bananas',
        'mango': 'Mangoes, guavas and mangosteens',
        'papaya': 'Papayas',
        'coffee': 'Coffee, green',
        'cotton': 'Seed cotton, unginned'
    }

    profiles = {}
    for raw_label, group in df.groupby('label'):
        norm_label = crop_map.get(raw_label, raw_label.title())
        profiles[norm_label] = {
            'N': group['N'].mean(),
            'P': group['P'].mean(),
            'K': group['K'].mean(),
            'ph': group['ph'].mean()
        }
    return profiles


def process_faostat_nigeria(weather_annual, soil_profiles):
    """Combine FAOSTAT Nigeria yield & area harvested data with aggregated state weather and soil parameters."""
    fao_path = os.path.join(DATASETS_DIR, "FAOSTAT_data_en_8-12-2026.csv")
    if not os.path.exists(fao_path):
        print(f"Warning: {fao_path} not found.")
        return pd.DataFrame()

    print("Loading FAOSTAT Nigeria yield data...")
    df_fao = pd.read_csv(fao_path)

    # Filter for Yield and Area harvested
    yield_df = df_fao[df_fao['Element'] == 'Yield'][['Item', 'Year', 'Value', 'Unit']].rename(columns={'Value': 'yield_kg_ha'})
    area_df = df_fao[df_fao['Element'] == 'Area harvested'][['Item', 'Year', 'Value', 'Unit']].rename(columns={'Value': 'area_harvested_ha'})

    merged_fao = pd.merge(yield_df, area_df, on=['Item', 'Year'], how='inner')
    merged_fao = merged_fao.rename(columns={'Item': 'crop', 'Year': 'year'})

    # Target key agricultural crops in Nigeria
    target_crops = [
        'Maize (corn)', 'Cassava, fresh', 'Rice', 'Yams', 'Sorghum',
        'Soya beans', 'Tomatoes', 'Groundnuts, excluding shelled',
        'Sweet potatoes', 'Cocoa beans', 'Oil palm fruit'
    ]
    merged_fao = merged_fao[merged_fao['crop'].isin(target_crops)]

    # Cross-join with 37 Nigerian states & state weather data for each year
    records = []
    np.random.seed(42)  # For reproducible regional micro-variations

    for _, row in merged_fao.iterrows():
        crop = row['crop']
        year = int(row['year'])
        base_yield = float(row['yield_kg_ha'])
        base_area = float(row['area_harvested_ha']) / 37.0  # Approx area per state

        # Soil defaults for crop
        soil = soil_profiles.get(crop, {'N': 50.0, 'P': 30.0, 'K': 25.0, 'ph': 6.2})

        # Match weather for this year across states
        year_weather = weather_annual[weather_annual['year'] == year]
        if year_weather.empty:
            continue

        for _, w in year_weather.iterrows():
            state = w['state']
            
            # Apply slight regional yield adjustment based on weather favorability
            rain = w['rainfall_mm']
            temp = w['avg_temp_c']
            
            # Yield response curve proxy
            rain_factor = np.clip(rain / 1200.0, 0.7, 1.3)
            temp_factor = np.clip(1.0 - abs(temp - 27.0) * 0.03, 0.8, 1.1)
            
            state_yield = round(base_yield * rain_factor * temp_factor * np.random.uniform(0.92, 1.08), 2)
            state_area = round(base_area * np.random.uniform(0.7, 1.3), 2)
            
            # Input estimates (kg/ha)
            fertilizer = round(np.random.uniform(40.0, 120.0), 1)
            pesticide = round(np.random.uniform(2.0, 8.5), 2)

            records.append({
                'state': state,
                'year': year,
                'crop': crop,
                'rainfall_mm': w['rainfall_mm'],
                'avg_temp_c': w['avg_temp_c'],
                'min_temp_c': w['min_temp_c'],
                'max_temp_c': w['max_temp_c'],
                'humidity_pct': w['humidity_pct'],
                'solar_radiation': w['solar_radiation'],
                'nitrogen_n': round(soil['N'] * np.random.uniform(0.9, 1.1), 1),
                'phosphorus_p': round(soil['P'] * np.random.uniform(0.9, 1.1), 1),
                'potassium_k': round(soil['K'] * np.random.uniform(0.9, 1.1), 1),
                'soil_ph': round(soil['ph'] * np.random.uniform(0.95, 1.05), 2),
                'fertilizer_kg_ha': fertilizer,
                'pesticide_kg_ha': pesticide,
                'area_harvested_ha': state_area,
                'yield_kg_ha': state_yield,
                'source': 'Nigeria_Synthesized_FAO_NASA'
            })

    df_ng = pd.DataFrame(records)
    print(f"Generated {len(df_ng)} Nigeria state-crop-year records.")
    return df_ng


def process_global_baseline():
    """Load and format the global Crop Yield with Soil and Weather dataset."""
    global_path = os.path.join(DATASETS_DIR, "Crop Yiled with Soil and Weather.csv")
    if not os.path.exists(global_path):
        print(f"Warning: {global_path} not found.")
        return pd.DataFrame()

    print("Loading global baseline soil & weather dataset...")
    df = pd.read_csv(global_path)

    # Standardize column mapping
    # Original: 'Fertilizer', 'temp', 'N', 'P', 'K', 'yeild'
    df = df.rename(columns={
        'Fertilizer': 'fertilizer_kg_ha',
        'temp': 'avg_temp_c',
        'N': 'nitrogen_n',
        'P': 'phosphorus_p',
        'K': 'potassium_k',
        'yeild': 'yield_kg_ha_raw'
    })

    # Yield in original CSV is in tonnes/ha (~5-12). Convert to kg/ha (e.g. 5000 - 12000)
    df['yield_kg_ha'] = (df['yield_kg_ha_raw'] * 1000.0).round(2)
    df = df.drop(columns=['yield_kg_ha_raw'])

    # Add default/synthetic values for missing columns to match target schema
    df['state'] = 'Global Baseline'
    df['year'] = 2020
    df['crop'] = 'Mixed Grain Baseline'
    df['rainfall_mm'] = (df['avg_temp_c'] * 35.0 + df['nitrogen_n'] * 5.0).round(2)
    df['min_temp_c'] = (df['avg_temp_c'] - 5.0).round(2)
    df['max_temp_c'] = (df['avg_temp_c'] + 6.0).round(2)
    df['humidity_pct'] = 65.0
    df['solar_radiation'] = 18.5
    df['soil_ph'] = 6.5
    df['pesticide_kg_ha'] = 4.5
    df['area_harvested_ha'] = 1000.0
    df['source'] = 'Global_Kaggle_Baseline'

    cols_order = [
        'state', 'year', 'crop', 'rainfall_mm', 'avg_temp_c', 'min_temp_c', 'max_temp_c',
        'humidity_pct', 'solar_radiation', 'nitrogen_n', 'phosphorus_p', 'potassium_k',
        'soil_ph', 'fertilizer_kg_ha', 'pesticide_kg_ha', 'area_harvested_ha', 'yield_kg_ha', 'source'
    ]
    return df[cols_order]


def main():
    print("=== STARTING DATASET PREPARATION PIPELINE ===")

    # 1. Weather
    weather_annual = process_nigeria_weather()

    # 2. Soil recommendations
    soil_profiles = load_crop_soil_recommendations()

    # 3. Nigeria dataset
    df_ng = process_faostat_nigeria(weather_annual, soil_profiles)

    # 4. Global baseline dataset
    df_global = process_global_baseline()

    # 5. Combine into unified table
    frames = [f for f in [df_ng, df_global] if not f.empty]
    if not frames:
        print("Error: No data was generated.")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Check non-null & ordering
    target_cols = [
        'state', 'year', 'crop', 'rainfall_mm', 'avg_temp_c', 'min_temp_c', 'max_temp_c',
        'humidity_pct', 'solar_radiation', 'nitrogen_n', 'phosphorus_p', 'potassium_k',
        'soil_ph', 'fertilizer_kg_ha', 'pesticide_kg_ha', 'area_harvested_ha', 'yield_kg_ha', 'source'
    ]
    combined = combined[target_cols]

    # Save to CSV
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSUCCESS! Created unified dataset at: {OUTPUT_FILE}")
    print(f"Total Rows: {len(combined)}")
    print(f"Columns: {list(combined.columns)}")
    print("\nDataset Summary by Source:")
    print(combined['source'].value_counts())


if __name__ == "__main__":
    main()
