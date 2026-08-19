"""
train_model.py
--------------
Automated Model Training, Evaluation, and Persistence Script for Crop Yield Estimator (Phase 2).

Loads:
  - datasets/processed_crop_yield.csv

Trains & Compares:
  1. LinearRegression (Baseline)
  2. RandomForestRegressor (Primary Model)
  3. GradientBoostingRegressor (Advanced Regressor)

Evaluates:
  - RMSE, MAE, R² on 80/20 Test Split
  - 5-Fold Cross-Validation on top model
  - Feature Importance Rankings

Saves:
  - model/crop_yield_model.pkl (Scikit-Learn Pipeline artifact)
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
DATASET_PATH = os.path.join(DATASETS_DIR, "processed_crop_yield.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "crop_yield_model.pkl")


def load_data():
    """Load the harmonized dataset and return X, y."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run src/prepare_dataset.py first.")

    df = pd.read_csv(DATASET_PATH)
    
    # Feature columns
    feature_cols = [
        'state', 'year', 'crop', 'rainfall_mm', 'avg_temp_c', 'min_temp_c', 'max_temp_c',
        'humidity_pct', 'solar_radiation', 'nitrogen_n', 'phosphorus_p', 'potassium_k',
        'soil_ph', 'fertilizer_kg_ha', 'pesticide_kg_ha', 'area_harvested_ha'
    ]
    target_col = 'yield_kg_ha'

    X = df[feature_cols]
    y = df[target_col]
    return X, y


def build_preprocessor(categorical_cols, numerical_cols):
    """Construct ColumnTransformer for numerical and categorical features."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])
    return preprocessor


def evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """Train and evaluate multiple models, returning metrics and best pipeline."""
    models = {
        'Linear Regression (Baseline)': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = []
    trained_pipelines = {}

    print("\n=== TRAINING & EVALUATING MODELS ===")
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Fit ONLY on train data
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Predict on test set
        y_pred = pipeline.predict(X_test)

        # Compute metrics
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Model': name,
            'RMSE (kg/ha)': round(rmse, 2),
            'MAE (kg/ha)': round(mae, 2),
            'R² Score': round(r2, 4)
        })
        print(f"[{name}] RMSE: {rmse:.2f} kg/ha | MAE: {mae:.2f} kg/ha | R²: {r2:.4f}")

    results_df = pd.DataFrame(results).sort_values(by='R² Score', ascending=False)
    best_model_name = results_df.iloc[0]['Model']
    best_pipeline = trained_pipelines[best_model_name]

    print(f"\nTop Performing Model: {best_model_name}")
    return results_df, best_model_name, best_pipeline


def root_mean_squared_error(y_true, y_pred):
    """Calculate RMSE compatible across scikit-learn versions."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def perform_cross_validation(pipeline, X_train, y_train):
    """Run 5-Fold Cross Validation on the training set."""
    print("\n=== RUNNING 5-FOLD CROSS-VALIDATION ON BEST MODEL ===")
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    print(f"5-Fold R² Scores: {[round(s, 4) for s in scores]}")
    print(f"Mean R²: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    return scores


def extract_feature_importances(pipeline, categorical_cols, numerical_cols):
    """Extract and print top feature importances from tree-based model in pipeline."""
    preprocessor = pipeline.named_steps['preprocessor']
    regressor = pipeline.named_steps['regressor']

    if not hasattr(regressor, 'feature_importances_'):
        print("Selected model does not support native feature_importances_.")
        return None

    # Retrieve feature names from OneHotEncoder
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
    all_feature_names = list(numerical_cols) + list(cat_feature_names)

    importances = regressor.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\n=== TOP 10 FEATURE IMPORTANCES ===")
    print(fi_df.head(10).to_string(index=False))
    return fi_df


def save_model(pipeline):
    """Save trained pipeline artifact with joblib."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved best model pipeline to: {MODEL_PATH}")


def main():
    print("=== STARTING MODEL TRAINING PIPELINE ===")
    X, y = load_data()

    categorical_cols = ['state', 'crop']
    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    # Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Dataset split into Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # Preprocessor
    preprocessor = build_preprocessor(categorical_cols, numerical_cols)

    # Train & Compare Models
    results_df, best_name, best_pipeline = evaluate_models(X_train, X_test, y_train, y_test, preprocessor)

    # Cross Validation
    perform_cross_validation(best_pipeline, X_train, y_train)

    # Feature Importance
    extract_feature_importances(best_pipeline, categorical_cols, numerical_cols)

    # Save Pipeline
    save_model(best_pipeline)
    print("\nMODEL TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
