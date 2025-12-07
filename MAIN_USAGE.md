# Main Pipeline Usage Guide

## Overview
`main.py` is the complete end-to-end ML pipeline for Bangkok Traffy complaint prediction. It handles data scraping, ETL, preprocessing, feature engineering, model training, and visualization.

## Environment Setup

### 1. Download Required Data
**Bangkok Traffy Dataset** - Download and place in `data/raw/` directory:
```bash
# Create directory if not exists
mkdir -p data/raw

# Download bangkok_traffy.csv and place it in data/raw/
# Expected file: data/raw/bangkok_traffy.csv
```

**Required file:** `data/raw/bangkok_traffy.csv` (Bangkok complaint dataset from Traffy Fondue)

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

### Key Dependencies
- pandas, numpy - Data manipulation
- scikit-learn - ML models
- imbalanced-learn - SMOTE resampling
- streamlit - Visualization dashboard
- plotly, pydeck - Interactive charts and maps

## Quick Start

### Basic Usage (Full Pipeline)
```bash
python main.py
```
Runs complete pipeline: ETL → Preprocessing → Feature Engineering → Model Training

### Common Workflows

#### 1. First Time Setup (With Data Scraping)
```bash
python main.py --scrape-weather
```
- Scrapes weather data from Open-Meteo API
- Scrapes air quality data (PM2.5, PM10, O3, NO2)
- Runs full ETL and training pipeline

#### 2. Quick Training (Skip ETL, Use Existing Data)
```bash
python main.py --skip-etl
```
- Loads existing `traffy_weather_merged.csv`
- Skips ETL phase (saves ~5-10 minutes)
- Runs preprocessing and training

#### 3. Data Processing Only (No Training)
```bash
python main.py --skip-training
```
- Runs ETL and preprocessing
- Creates `traffy_weather_final.csv` with all features
- Skips model training (for data exploration)

#### 4. Fast Prototyping (Sampled Data)
```bash
python main.py --sample 200000
```
- Randomly samples 200,000 records
- Faster training for testing (~30-40 minutes)
- Useful for parameter tuning

#### 5. Full Pipeline + Visualization
```bash
python main.py --visualize
```
- Runs complete pipeline
- Launches Streamlit dashboard at http://localhost:8501
- Press Ctrl+C to stop server

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--scrape-weather` | flag | False | Scrape weather and air quality data before pipeline |
| `--skip-etl` | flag | False | Skip ETL phase, load existing merged data |
| `--skip-training` | flag | False | Skip model training phase |
| `--visualize` | flag | False | Launch Streamlit dashboard after pipeline |
| `--sample` | int | None | Sample N records for faster training (e.g., 200000) |
| `--n-iter` | int | 5 | RandomizedSearchCV iterations per model |
| `--min-samples` | int | 50 | Minimum positive samples to train a type |

## Pipeline Phases

### Phase 1: Data Scraping (Optional)
**When to use:** `--scrape-weather` flag
- Downloads weather data (temperature, rain, wind, etc.)
- Downloads air quality data (PM2.5, PM10, O3, NO2)
- Saves to `data/raw/` directory

### Phase 2: ETL (Extract, Transform, Load)
**When to skip:** `--skip-etl` flag if data already merged
- Loads raw Traffy complaints (`data/raw/bangkok_traffy.csv`)
- Loads weather data (`data/raw/open-meteo-13.74N100.50.csv`)
- Loads air quality data (`data/processed/bangkok-air-quality.csv`)
- Merges all datasets on date
- Saves to `data/processed/traffy_weather_merged.csv`

**Output:** `traffy_weather_merged.csv` (~700K records)

### Phase 3: Preprocessing
- Parses complaint types from string to list
- Filters empty types
- Drops rows with missing weather data
- Optional: Samples data if `--sample` specified

**Output:** Cleaned dataset ready for feature engineering

### Phase 4: Feature Engineering
- Extracts temporal features (hour, day, month)
- Encodes cyclical features (sin/cos transformations)
- One-hot encodes districts (50 Bangkok districts)
- Creates binary target columns (26 complaint types)

**Output:** `traffy_weather_final.csv` with 70+ features

### Phase 5: Model Training
**When to skip:** `--skip-training` flag
- Trains 26 Random Forest models (one per complaint type)
- Uses adaptive resampling strategy:
  - **High frequency (≥15%):** Class weights only
  - **Medium frequency (5-15%):** Undersampling to 1:2 ratio
  - **Low frequency (<5%):** SMOTE + undersampling
- Hyperparameter tuning with RandomizedSearchCV
- Saves models to `data/models/`

**Output:** 
- 26 model pickle files (`data/models/*_model.pkl`)
- Feature names (`data/models/feature_names.pkl`)
- Training summary (`data/models/training_summary.csv`)

### Phase 6: Visualization (Optional)
**When to use:** `--visualize` flag
- Launches Streamlit dashboard
- Interactive maps, charts, and model performance views
- Accessible at http://localhost:8501

## Example Workflows

### Development Workflow
```bash
# 1. First run: scrape fresh data
python main.py --scrape-weather --sample 200000

# 2. Iterate on features: skip ETL, use sampled data
python main.py --skip-etl --sample 200000

# 3. Final training: full data, optimized parameters
python main.py --skip-etl --n-iter 10
```

### Production Workflow
```bash
# 1. Scrape latest data
python main.py --scrape-weather --skip-training

# 2. Train all models with full data
python main.py --skip-etl --n-iter 10

# 3. Launch dashboard
python main.py --skip-etl --skip-training --visualize
```

### Quick Testing
```bash
# Test pipeline without training
python main.py --skip-training

# Test with small sample
python main.py --sample 50000 --n-iter 3
```

## Output Files

### Data Files
- `data/processed/traffy_weather_merged.csv` - Merged raw data
- `data/processed/traffy_weather_final.csv` - Engineered features + targets

### Model Files
- `data/models/*_model.pkl` - 26 trained Random Forest models
- `data/models/feature_names.pkl` - Feature column names
- `data/models/training_summary.csv` - Performance metrics for all models

### Training Summary Columns
- `type` - Complaint type name
- `accuracy`, `precision`, `recall`, `f1` - Evaluation metrics
- `n_estimators`, `max_depth` - Best hyperparameters
- `model_file` - Path to saved model

## Performance Tips

### Speed Optimization
1. **Use `--skip-etl`** after first run (saves 5-10 minutes)
2. **Use `--sample`** for prototyping (e.g., 200000 records = ~30 min)
3. **Reduce `--n-iter`** for faster tuning (default: 5, max: 10)
4. **Increase `--min-samples`** to train fewer types (default: 50)

### Memory Management
- Full dataset: ~700K records, 70+ features (~2-3 GB RAM)
- Sampled dataset (200K): ~500 MB RAM
- Training all models: ~4-6 GB RAM peak

### Time Estimates
- ETL phase: 5-10 minutes
- Preprocessing: 2-3 minutes
- Feature engineering: 1-2 minutes
- Model training (full data, 26 models): 2-4 hours
- Model training (200K sample): 30-40 minutes

## Help & Documentation

### Show Available Arguments
```bash
python main.py --help
```


