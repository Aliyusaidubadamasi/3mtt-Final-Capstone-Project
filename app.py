"""
app.py
------
FastAPI Backend Server & Web Application Router for Crop Yield Estimator (Phase 3).

Serves:
  - Multi-page glassmorphic Web App UI at `/`
  - REST API Endpoints:
      POST `/api/predict`: Runs model yield prediction & feature importances
      GET  `/api/options`: Returns list of states & crops for form dropdowns
      GET  `/api/analytics-data`: Aggregated summary statistics for Analytics dashboard
      GET  `/api/health`: System status healthcheck
"""

import os
import pandas as pd
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.predict import predict_yield, get_feature_importance, FEATURE_COLUMNS, load_model

app = FastAPI(
    title="Crop Yield Estimator API",
    description="ML-powered yield forecasting and explainable feature evaluation for Nigerian Agriculture",
    version="1.0.0"
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "processed_crop_yield.csv")

# Mount Static Assets
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class YieldRequest(BaseModel):
    state: str = Field(default="Kaduna", description="State or territory name")
    year: int = Field(default=2024, ge=2000, le=2030, description="Calendar year")
    crop: str = Field(default="Maize (corn)", description="Crop commodity name")
    rainfall_mm: float = Field(default=1150.0, ge=0.0, le=5000.0, description="Annual rainfall in mm")
    avg_temp_c: float = Field(default=27.0, ge=10.0, le=45.0, description="Mean temperature in Celsius")
    min_temp_c: float = Field(default=20.5, ge=5.0, le=40.0, description="Minimum temperature in Celsius")
    max_temp_c: float = Field(default=33.5, ge=15.0, le=50.0, description="Maximum temperature in Celsius")
    humidity_pct: float = Field(default=65.0, ge=0.0, le=100.0, description="Relative humidity percentage")
    solar_radiation: float = Field(default=18.5, ge=0.0, le=40.0, description="Solar radiation MJ/m2/day")
    nitrogen_n: float = Field(default=80.0, ge=0.0, le=300.0, description="Soil Nitrogen concentration")
    phosphorus_p: float = Field(default=35.0, ge=0.0, le=200.0, description="Soil Phosphorus concentration")
    potassium_k: float = Field(default=25.0, ge=0.0, le=200.0, description="Soil Potassium concentration")
    soil_ph: float = Field(default=6.5, ge=3.0, le=10.0, description="Soil pH level")
    fertilizer_kg_ha: float = Field(default=95.0, ge=0.0, le=500.0, description="Fertilizer rate kg/ha")
    pesticide_kg_ha: float = Field(default=4.2, ge=0.0, le=50.0, description="Pesticide rate kg/ha")
    area_harvested_ha: float = Field(default=1200.0, ge=0.1, le=100000.0, description="Harvested area in ha")


@app.get("/")
def read_root():
    """Serve the main multi-page web app index.html."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Crop Yield Estimator API is active. Web UI index.html under construction."}


@app.get("/api/health")
def healthcheck():
    """System health check endpoint."""
    model_loaded = False
    try:
        load_model()
        model_loaded = True
    except Exception:
        pass
    return {
        "status": "online",
        "model_loaded": model_loaded,
        "dataset_exists": os.path.exists(DATASET_PATH)
    }


@app.get("/api/options")
def get_options():
    """Return available states and crops for frontend select dropdowns."""
    if not os.path.exists(DATASET_PATH):
        raise HTTPException(status_code=404, detail="Processed dataset not found.")

    df = pd.read_csv(DATASET_PATH)
    states = sorted(df['state'].unique().tolist())
    crops = sorted(df['crop'].unique().tolist())

    return {
        "states": states,
        "crops": crops
    }


@app.post("/api/predict")
def predict(request: YieldRequest):
    """Run model inference and extract top feature importances."""
    try:
        input_data = request.model_dump()
        result = predict_yield(input_data)
        importances = get_feature_importance(top_n=5)
        result["feature_importances"] = importances
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/analytics-data")
def get_analytics():
    """Return aggregated stats for the Analytics Dashboard tab."""
    if not os.path.exists(DATASET_PATH):
        raise HTTPException(status_code=404, detail="Dataset file not found.")

    df = pd.read_csv(DATASET_PATH)

    # Yield by Crop
    crop_stats = df.groupby("crop")["yield_kg_ha"].mean().round(2).reset_index()
    crop_summary = crop_stats.sort_values(by="yield_kg_ha", ascending=False).to_dict(orient="records")

    # Yield by Nigerian State (excluding Global Baseline)
    ng_df = df[df["source"] == "Nigeria_Synthesized_FAO_NASA"]
    state_stats = ng_df.groupby("state")["yield_kg_ha"].mean().round(2).reset_index()
    top_states = state_stats.sort_values(by="yield_kg_ha", ascending=False).head(10).to_dict(orient="records")

    # Key dataset metrics
    metrics = {
        "total_records": int(len(df)),
        "nigeria_records": int(len(ng_df)),
        "global_baseline_records": int(len(df) - len(ng_df)),
        "mean_yield_kg_ha": float(round(df["yield_kg_ha"].mean(), 2)),
        "median_yield_kg_ha": float(round(df["yield_kg_ha"].median(), 2)),
        "max_yield_kg_ha": float(round(df["yield_kg_ha"].max(), 2)),
        "min_yield_kg_ha": float(round(df["yield_kg_ha"].min(), 2)),
        "mean_rainfall_mm": float(round(ng_df["rainfall_mm"].mean(), 2)),
        "mean_temp_c": float(round(ng_df["avg_temp_c"].mean(), 2))
    }

    return {
        "metrics": metrics,
        "crop_summary": crop_summary,
        "top_states": top_states
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
