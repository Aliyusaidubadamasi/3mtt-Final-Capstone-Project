# Crop Yield Estimator — Nigerian Agricultural AI (3MTT Capstone Project)

> **Data-driven crop yield forecasting and explainable feature evaluation for Nigerian agriculture.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.0-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141.1-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 1. Problem Statement (Nigerian Agriculture Context)

Smallholder farmers across Nigeria frequently experience unpredictable crop yields and financial instability due to climate variability, lack of localized weather insights, and uncertain yield expectations prior to planting. Without data-driven yield benchmarks, farmers struggle to plan fertilizer investments, storage logistics, and crop insurance effectively.

The **Crop Yield Estimator** provides a machine learning solution that estimates crop yield per hectare ($kg/ha$) from localized satellite weather telemetry (rainfall, temperature, solar radiation), soil nutrient profiles ($N, P, K, pH$), and farming input rates.

---

## 2. What it Does (MVP Scope)

### In Scope (Current MVP)
- **Predictive Yield Forecasts**: Instant crop yield prediction ($kg/ha$ and $tonnes/ha$) based on selected Nigerian state, crop commodity, climate factors, and soil inputs.
- **Explainable Feature Importance**: Native tree ensemble feature importance rankings displaying the top 5 environmental and soil factors driving each prediction.
- **Interactive Multi-Page Web Product**: Sleek glassmorphic web application featuring a Yield Estimator Tool, Agricultural Analytics Dashboard, Model Benchmarks tab, and Data Provenance view.
- **Complete Jupyter Notebook Suite**: 3 step-by-step notebooks covering dataset preparation (ETL), exploratory data analysis (EDA), and machine learning model benchmarks.

### Out of Scope (Future Roadmap)
- Real-time IoT soil sensor hardware integrations.
- Satellite multispectral imagery (NDVI/EVI) computation.
- Low-connectivity offline USSD / SMS gateway interface for basic feature phones.

---

## 3. Demo

> **[Demo Video Link Placeholder]** — *A 2-minute walkthrough demonstrating parameter adjustments, prediction generation, and feature importance visual analysis.*

---

## 4. Dataset Lineage & Sources

No single raw dataset contains Nigeria + state + weather + soil nutrients + yield in one CSV. We combined four authoritative and baseline sources into a unified dataset (`datasets/processed_crop_yield.csv` — **5,038 total rows**):

| Source Name | Data Provider | Description & Coverage |
|---|---|---|
| **NASA POWER Agroclimatology API** | NASA Langley Research Center | Daily rainfall, air temperatures, relative humidity, and solar radiation for 37 Nigerian states (2015–2024). |
| **FAOSTAT** | UN Food and Agriculture Organization (FAO) | Ground-truth national crop yield ($kg/ha$) and area harvested data for Nigeria (2019–2024). |
| **Crop Recommendation Dataset** | Kaggle (`atharvaingle`) | Representative soil nutrient requirement profiles ($N, P, K, pH$) across major staple crops. |
| **Global Crop Yield with Soil & Weather** | Kaggle (`gurudathg`) | Baseline global soil nutrient, weather, fertilizer, and crop yield observation dataset (2,596 rows). |

---

## 5. Model Benchmarks & Comparison

We evaluated three scikit-learn regressor algorithms on an 80/20 train/test split. Preprocessing is encapsulated inside a `ColumnTransformer` (`OneHotEncoder` for state/crop categoricals, `StandardScaler` and `SimpleImputer` for numerical features).

| Model Algorithm | RMSE ($kg/ha$) | MAE ($kg/ha$) | $R^2$ Score | Status |
|---|---|---|---|---|
| **Random Forest Regressor** | **215.85** | **137.08** | **0.9963** | **Selected Model** |
| Gradient Boosting Regressor | 425.50 | 307.45 | 0.9856 | Secondary Benchmark |
| Linear Regression (Baseline) | 749.61 | 595.80 | 0.9553 | Baseline |

### 5-Fold Cross-Validation Metrics
- **Mean $R^2$ Score**: **0.9964 ($\pm 0.0002$)**, validating strong model stability and generalization across folds without overfitting.

### Top Feature Importance Drivers
1. **State & Crop Categorical Baselines** ($37.9\%$)
2. **Soil Phosphorus ($P$)** ($13.6\%$)
3. **Crop Commodity Type** ($12.0\%$)
4. **Harvested Area ($ha$)** ($10.4\%$)
5. **Temperature & Rainfall Telemetry** ($14.0\%$)

---

## 6. How to Run Locally

### Prerequisites
- Python 3.10+
- `pip`

### Step 1: Clone Repository & Set Up Virtual Environment
```bash
git clone https://github.com/Aliyusaidubadamasi/3mtt-Final-Capstone-Project.git
cd 3mtt-Final-Capstone-Project

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Multi-Page Web Product
```bash
python -m uvicorn app:app --port 8000 --reload
```
Open your browser at `http://localhost:8000`.

### Step 4: Run Jupyter Notebooks
```bash
jupyter notebook
```
Explore the notebooks in order inside `/notebooks`:
- `00_dataset_preparation_pipeline.ipynb`: Interactive ETL pipeline.
- `01_dataset_preparation_and_eda.ipynb`: Exploratory Data Analysis & visual plots.
- `02_model_training_and_evaluation.ipynb`: Model training, comparison, residual plots & artifact export.

---

## 7. Repository Structure

```
3mtt-Final-Capstone-Project/
├── app.py                          # FastAPI backend server & static router
├── requirements.txt                # Pinned project dependencies
├── README.md                       # Project documentation
├── Procfile                        # Cloud deployment web process configuration
├── render.yaml                     # Render automated cloud deployment blueprint
├── download_nasa_power_nigeria.py  # NASA POWER API daily weather downloader
│
├── datasets/                       # Raw and processed datasets
│   ├── processed_crop_yield.csv    # Final 5,038-row unified dataset
│   ├── nigeria_states_weather_combined.csv # NASA POWER daily weather
│   ├── FAOSTAT_data_en_8-12-2026.csv # FAOSTAT Nigeria yield & area
│   ├── Crop Yiled with Soil and Weather.csv # Global Kaggle dataset
│   └── Crop_recommendation.csv     # Soil nutrient requirement profiles
│
├── docs/                           # Documentation
│   └── DATA_DICTIONARY.md          # Full data dictionary & lineage
│
├── model/                          # Persisted Model Artifacts
│   └── crop_yield_model.pkl        # Trained scikit-learn Pipeline artifact
│
├── notebooks/                      # Jupyter Evaluation Notebooks
│   ├── 00_dataset_preparation_pipeline.ipynb
│   ├── 01_dataset_preparation_and_eda.ipynb
│   └── 02_model_training_and_evaluation.ipynb
│
├── src/                            # Core Python Modules
│   ├── prepare_dataset.py          # Data ingestion & ETL pipeline
│   ├── train_model.py              # ML model comparison & training
│   └── predict.py                  # Inference & feature importance API
│
└── static/                         # Web Application Frontend Assets
    ├── index.html                  # Multi-page glassmorphic HTML5 UI
    ├── style.css                   # Vanilla CSS design system & styling
    └── script.js                   # Client-side tab router & API fetch logic
```

---

## 8. Tech Stack

- **Core**: Python 3.10+, Pandas, NumPy
- **Machine Learning**: Scikit-Learn (RandomForestRegressor, GradientBoosting, ColumnTransformer, Joblib)
- **Backend API**: FastAPI, Uvicorn, Pydantic
- **Frontend UI**: HTML5, Vanilla CSS3 (Glassmorphism, Dark/Emerald Palette), Vanilla JavaScript
- **Visualizations**: Matplotlib, Seaborn

---

## 9. Limitations & Future Work

- **State-Level Climate Granularity**: Weather data is currently aggregated at state level; LGA or sub-county weather resolution will enhance micro-climate accuracy.
- **USSD/SMS Access**: Future development will introduce an offline USSD gateway for smallholder farmers with feature phones in low-connectivity rural zones.
- **Satellite Vegetation Indices**: Incorporating real-time Sentinel-2 / Landsat NDVI telemetry during the growing season.

---

## 10. License

This project is licensed under the **MIT License**.
