# Data Dictionary — Crop Yield Estimator Dataset

This document defines the schema, data types, sources, units, and transformation lineage for `processed_crop_yield.csv`, the unified training dataset for the **Crop Yield Estimator** project.

---

## 1. Overview & Data Lineage

`processed_crop_yield.csv` combines multiple authoritative and baseline data sources into a single tabular dataset suitable for supervised machine learning models (e.g. `RandomForestRegressor`, `GradientBoostingRegressor`).

| Source Name | Data Provider | Details & Coverage |
|---|---|---|
| **NASA POWER Agroclimatology API** | NASA Langley Research Center | Daily weather metrics for all 36 Nigerian states + FCT (2015–2024), aggregated into annual totals/averages by state. |
| **FAOSTAT** | UN Food and Agriculture Organization (FAO) | Ground truth national crop yield (`kg/ha`) and area harvested (`ha`) for Nigeria (2019–2024). |
| **Crop Recommendation Dataset** | Kaggle (atharvaingle) | Optimal soil nutrient parameters ($N, P, K$, $pH$) for major staple crops (Maize, Rice, Cassava, Yam, Sorghum, Soya, etc.). |
| **Global Crop Yield with Soil & Weather** | Kaggle (gurudathg) | Baseline global soil nutrient, weather, fertilizer, and crop yield observation dataset (2,596 rows). |

---

## 2. Table Schema Definition

The table below describes every feature present in `datasets/processed_crop_yield.csv`.

| Feature Column | Data Type | Description | Unit / Scale | Example Value | Primary Source |
|---|---|---|---|---|---|
| `state` | `string` (categorical) | State or region name (36 Nigerian states + FCT Abuja, or `Global Baseline`) | N/A | `Kaduna` | NASA POWER / FAOSTAT |
| `year` | `int64` | Calendar year of observation | YYYY (2015–2024) | `2022` | NASA POWER / FAOSTAT |
| `crop` | `string` (categorical) | Agricultural crop commodity name | Categorical | `Maize (corn)` | FAOSTAT / Kaggle |
| `rainfall_mm` | `float64` | Total annual cumulative precipitation | Millimeters ($mm$) | `1145.80` | NASA POWER |
| `avg_temp_c` | `float64` | Mean annual air temperature at 2 meters | Degrees Celsius ($\degree C$) | `26.85` | NASA POWER |
| `min_temp_c` | `float64` | Mean annual minimum daily temperature | Degrees Celsius ($\degree C$) | `20.40` | NASA POWER |
| `max_temp_c` | `float64` | Mean annual maximum daily temperature | Degrees Celsius ($\degree C$) | `33.10` | NASA POWER |
| `humidity_pct` | `float64` | Mean annual relative humidity at 2 meters | Percentage ($\%$) | `64.30` | NASA POWER |
| `solar_radiation` | `float64` | Mean annual surface all-sky shortwave solar radiation | $MJ/m^2/day$ or $kW/m^2/day$ | `18.20` | NASA POWER |
| `nitrogen_n` | `float64` | Soil nitrogen concentration ratio/index | Ratio / $ppm$ | `78.50` | Kaggle Soil / Crop Rec |
| `phosphorus_p` | `float64` | Soil phosphorus concentration ratio/index | Ratio / $ppm$ | `28.40` | Kaggle Soil / Crop Rec |
| `potassium_k` | `float64` | Soil potassium concentration ratio/index | Ratio / $ppm$ | `22.10` | Kaggle Soil / Crop Rec |
| `soil_ph` | `float64` | Soil acidity/alkalinity level | $pH$ scale ($0.0 - 14.0$) | `6.45` | Kaggle Soil / Crop Rec |
| `fertilizer_kg_ha` | `float64` | Chemical fertilizer application rate | Kilograms per hectare ($kg/ha$) | `85.00` | Kaggle / Estimated |
| `pesticide_kg_ha` | `float64` | Pesticide application rate | Kilograms per hectare ($kg/ha$) | `4.20` | Kaggle / Estimated |
| `area_harvested_ha` | `float64` | Total land area harvested for crop | Hectares ($ha$) | `1250.00` | FAOSTAT |
| **`yield_kg_ha`** | `float64` (**Target**) | Crop production yield | Kilograms per hectare ($kg/ha$) | `2150.00` | FAOSTAT / Kaggle |
| `source` | `string` | Provenance identifier (`Nigeria_Synthesized_FAO_NASA` vs `Global_Kaggle_Baseline`) | Metadata | `Nigeria_Synthesized_FAO_NASA` | Data Pipeline |

---

## 3. Transformation & Harmonization Rules

1. **Weather Aggregation**: Daily records from NASA POWER API were grouped by state and year. `PRECTOTCORR` was summed to yield total annual rainfall ($mm$), while temperatures, humidity, and solar radiation were averaged across the 365 daily observations.
2. **Yield Unit Standardization**: FAOSTAT yield entries were verified in $kg/ha$. Global Kaggle baseline yield values were converted from tonnes/ha to $kg/ha$ ($\text{yield}_{kg} = \text{yield}_{tonnes} \times 1000$).
3. **Soil Profile Matching**: Soil nutrients ($N, P, K, pH$) were mapped per crop using empirical agronomic averages from `Crop_recommendation.csv`.
4. **Missing Value Policy**: All numerical features in `processed_crop_yield.csv` are complete (zero nulls). `SimpleImputer(strategy='mean')` will be wrapped inside scikit-learn `Pipeline` objects in Phase 2 for robustness during live inference.

---

## 4. Known Limitations & Caveats

- **State-Level Weather Aggregation**: Annual averages smooth out intra-seasonal climate shocks (e.g. mid-season drought during flowering).
- **Proxy Soil Data**: Soil nutrient levels ($N, P, K$) reflect crop-specific agronomic baselines and regional variation rather than high-density physical soil tests across every LGA.
- **Directional Scope**: Results are intended as an engineering MVP for crop yield estimation and decision support, rather than definitive agronomic guarantees.
