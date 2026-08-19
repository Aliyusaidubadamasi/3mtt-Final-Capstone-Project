"""
predict.py
----------
Inference module for Crop Yield Estimator (Phase 2 & MVP).

Functions:
  - load_model(): Load crop_yield_model.pkl at startup
  - predict_yield(input_dict): Take raw feature input dictionary, format DataFrame, run prediction
  - get_feature_importance(top_n=5): Extract top features driving yield predictions
"""

import os
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "crop_yield_model.pkl")
_MODEL_PIPELINE = None

FEATURE_COLUMNS = [
    'state', 'year', 'crop', 'rainfall_mm', 'avg_temp_c', 'min_temp_c', 'max_temp_c',
    'humidity_pct', 'solar_radiation', 'nitrogen_n', 'phosphorus_p', 'potassium_k',
    'soil_ph', 'fertilizer_kg_ha', 'pesticide_kg_ha', 'area_harvested_ha'
]


def load_model():
    """Load and cache the trained scikit-learn pipeline."""
    global _MODEL_PIPELINE
    if _MODEL_PIPELINE is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run src/train_model.py first.")
        _MODEL_PIPELINE = joblib.load(MODEL_PATH)
    return _MODEL_PIPELINE


def predict_yield(input_dict: dict) -> dict:
    """Predict crop yield (kg/ha) from a raw input dictionary.

    Args:
        input_dict: dictionary containing keys matching FEATURE_COLUMNS.

    Returns:
        dictionary with predicted_yield_kg_ha, predicted_yield_tonnes_ha, and interpretation.
    """
    pipeline = load_model()

    # Validate required fields
    missing_fields = [col for col in FEATURE_COLUMNS if col not in input_dict]
    if missing_fields:
        raise ValueError(f"Missing required input features: {missing_fields}")

    # Build 1-row DataFrame in exact training order
    input_df = pd.DataFrame([input_dict])[FEATURE_COLUMNS]

    # Run prediction
    predicted_kg = float(pipeline.predict(input_df)[0])
    predicted_kg = round(max(0.0, predicted_kg), 2)
    predicted_tonnes = round(predicted_kg / 1000.0, 2)

    # Contextual interpretation
    crop = input_dict.get('crop', 'Crop')
    state = input_dict.get('state', 'Region')
    
    if predicted_kg > 8000:
        interpretation = f"High yield potential for {crop} in {state}. Soil nutrients and climate conditions are highly favorable."
    elif predicted_kg >= 3000:
        interpretation = f"Moderate/Average expected yield for {crop} in {state}. Recommended input levels are maintained."
    else:
        interpretation = f"Below-average expected yield for {crop} in {state}. Check soil nutrient levels (N/P/K) or water availability."

    return {
        'crop': crop,
        'state': state,
        'predicted_yield_kg_ha': predicted_kg,
        'predicted_yield_tonnes_ha': predicted_tonnes,
        'interpretation': interpretation
    }


def get_feature_importance(top_n: int = 5) -> list:
    """Retrieve top N features driving model predictions."""
    pipeline = load_model()
    preprocessor = pipeline.named_steps['preprocessor']
    regressor = pipeline.named_steps['regressor']

    if not hasattr(regressor, 'feature_importances_'):
        return []

    cat_cols = ['state', 'crop']
    num_cols = [c for c in FEATURE_COLUMNS if c not in cat_cols]

    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
    all_feature_names = list(num_cols) + list(cat_feature_names)

    importances = regressor.feature_importances_
    fi_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    top_features = []
    for _, row in fi_df.head(top_n).iterrows():
        top_features.append({
            'feature': row['feature'],
            'importance': round(float(row['importance']), 4)
        })
    return top_features


if __name__ == "__main__":
    # Sanity check test
    print("Testing predict_yield()...")
    sample_input = {
        'state': 'Kaduna',
        'year': 2024,
        'crop': 'Maize (corn)',
        'rainfall_mm': 1180.5,
        'avg_temp_c': 26.8,
        'min_temp_c': 20.2,
        'max_temp_c': 33.1,
        'humidity_pct': 65.0,
        'solar_radiation': 18.2,
        'nitrogen_n': 80.0,
        'phosphorus_p': 35.0,
        'potassium_k': 25.0,
        'soil_ph': 6.5,
        'fertilizer_kg_ha': 95.0,
        'pesticide_kg_ha': 4.2,
        'area_harvested_ha': 1200.0
    }
    result = predict_yield(sample_input)
    print("Prediction Result:", result)
    print("\nTop 5 Feature Importances:")
    for f in get_feature_importance(5):
        print(f"  - {f['feature']}: {f['importance']}")
