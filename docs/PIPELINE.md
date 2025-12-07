# Data Pipeline Overview

## High-Level Process

The pipeline transforms raw complaint and environmental data into model-ready features through four main phases:

---

## 1. Data Extraction (ETL)

### Input Sources
- **Traffy Complaints**: Bangkok citizen complaints (`bangkok_traffy.csv`)
- **Weather Data**: Hourly weather measurements (`open-meteo-*.csv`)
- **Air Quality**: Daily PM2.5, PM10, O3, NO2 levels (`bangkok-air-quality.csv`)

### Processing
1. Load raw CSV files
2. Clean and validate data types
3. Parse timestamps to datetime format
4. Extract coordinates from complaint locations

**Output**: Three cleaned dataframes ready for merging

---

## 2. Data Merging & Aggregation

### Steps
1. **Extract date** from timestamps (YYYY-MM-DD)
2. **Aggregate weather** from hourly → daily averages
3. **Merge on date**:
   - Traffy complaints + Weather (left join)
   - Result + Air quality (left join)

### Result
- Single unified dataset
- Each complaint row enriched with weather + air quality for that day
- ~540K complaint records with environmental context

**Output**: `traffy_weather_merged.csv`

---

## 3. Preprocessing & Cleaning

### Data Quality Steps
1. **Parse complaint types**: Convert `{ถนน,ไฟฟ้า}` → list of types
2. **Filter empty types**: Remove complaints without categorization
3. **Drop missing weather**: Remove rows without weather data
4. **Drop missing air quality**: Remove rows without PM2.5/PM10/O3/NO2

### Result
- Clean dataset with complete information
- Ready for feature engineering

**Output**: Cleaned dataframe (~500K records)

---

## 4. Feature Engineering

### Temporal Features
- Extract: `hour`, `day_of_week`, `month`
- Encode cyclically: `hour_sin/cos`, `day_sin/cos`, `month_sin/cos`
- Add: `is_weekend` flag

### Location Features
- Keep: `latitude`, `longitude`
- One-hot encode: `district_*` (if available)

### Environmental Features
- Weather: `temperature`, `rain`, `humidity`, `wind_speed`, etc.
- Air quality: `pm25`, `pm10`, `o3`, `no2`

### Target Creation
- Create binary columns: `type_ถนน`, `type_ไฟฟ้า`, etc.
- One column per complaint type (multi-label classification)

**Output**: `traffy_weather_final.csv` (~20-30 features)

---

## Pipeline Execution

```bash
# Full pipeline with training
python main.py

# Skip ETL (use existing merged data)
python main.py --skip-etl

# Skip training (preprocessing only)
python main.py --skip-training

# Fast training with sampling
python main.py --sample 200000 --n-iter 5
```

---

## Data Flow Summary

```
Raw Data
  ↓
[Extract] → Load & Clean
  ↓
[Merge] → Combine by Date
  ↓
[Preprocess] → Filter & Validate
  ↓
[Engineer] → Create Features
  ↓
Final Dataset → Ready for Training
```
