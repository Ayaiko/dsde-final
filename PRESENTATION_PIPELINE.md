# Pipeline Presentation Guide

## 1. Introduction & Problem Statement (1-2 min)

### Key Points
- **Bangkok Traffy System**: 700K+ citizen complaint records
- **Challenge**: Predict complaint types from environmental patterns
- **Goal**: Enable data-driven insights for city planning and resource allocation

### What to Say
> "Bangkok's Traffy Fondue system collects over 700,000 citizen complaints. Our challenge was to understand if environmental factors like weather and air quality could predict what types of complaints citizens would file. This could help the city proactively allocate resources."

---

## 2. Data Sources (2 min)

### Three Main Data Sources

#### Complaint Data (Bangkok Traffy)
- Timestamp of complaint
- Location (latitude, longitude, district)
- Complaint type(s) - can be multi-label
- Citizen descriptions

#### Weather Data (Open-Meteo API)
- Temperature, dew point
- Humidity, rain
- Wind speed/direction
- Cloud cover, pressure
- **Originally hourly** → aggregated to daily averages

#### Air Quality Data
- PM2.5, PM10 (particulate matter)
- O3 (ozone)
- NO2 (nitrogen dioxide)
- **Daily measurements**

### Why These Sources?
> "We hypothesized that environmental factors influence citizen behavior. For example, flooding complaints spike after heavy rain, or air quality complaints increase with high PM2.5 levels."

---

## 3. Pipeline Architecture (3-4 min)

**Visual Aid**: Show `draw_high_level.xml`

### Phase 1: Data Collection
- **Scraping**: Automated weather and air quality data collection
- **Loading**: Read 700K+ complaint records from CSV
- **Storage**: Organized in `data/raw/` directory

### Phase 2: Data Processing
- **Cleaning**: 
  - Remove duplicates
  - Handle missing values
  - Replace space characters with NaN
  - Standardize formats
- **Aggregation**: Hourly weather → daily averages
- **Merging**: Left join on date column
  - Preserve all complaints even if missing weather data
  - Key decision: Don't lose complaint data

### Phase 3: Feature Engineering
- **Temporal features**: Extract hour, day_of_week, month
- **Cyclical encoding**: Convert to sin/cos pairs
- **Location features**: Lat/lon + one-hot encode 50 districts
- **Result**: 70 total features

### Phase 4: Model Training
- Multi-label classification approach
- One Random Forest model per complaint type
- Adaptive class imbalance handling

### Why Modular Design?
> "We built the pipeline with reusable modules in the `pipeline/` directory. This makes it easy to add new data sources, debug issues, and maintain code quality."

---

## 4. Data Challenges & Solutions (2-3 min)

### Challenge 1: Missing Air Quality Data
**Problem**: Not all dates have air quality measurements  
**Solution**: 
- Identified null values in PM columns
- Dropped rows with missing air quality data
- Trade-off: Lose some records but maintain data quality

```python
# Dynamic null detection
air_quality_cols = [col for col in df.columns 
                    if any(x in col.lower() for x in ['pm2.5', 'pm10', 'o3', 'no2'])]
df = df.dropna(subset=air_quality_cols)
```

### Challenge 2: Space Characters as Values
**Problem**: Some cells contained single space `' '` instead of actual values  
**Solution**: 
- Replace spaces with `pd.NA`
- Then use `dropna()` to remove affected rows

```python
df.replace(' ', pd.NA, inplace=True)
df.dropna(how='any', inplace=True)
```

### Challenge 3: Temporal Mismatch
**Problem**: Weather is hourly, air quality is daily, complaints are any time  
**Solution**:
- Aggregate weather to daily averages using `groupby('date').mean()`
- Merge all datasets on normalized date (YYYY-MM-DD)
- Left join to preserve all complaints

### Challenge 4: Data Standardization
**Problem**: Different formats, encoding issues  
**Solution**:
- Consistent timestamp parsing
- Coordinate validation and splitting
- Type field parsing (comma-separated → list)

---

## 5. Feature Engineering Details (2 min)

### The 70 Features Breakdown

#### Environmental Features (11)
- `temperature_2m`, `dew_point_2m`
- `relative_humidity_2m`, `rain`
- `wind_speed_10m`, `wind_direction_10m`
- `cloud_cover`, `surface_pressure`
- `vapour_pressure_deficit`

#### Temporal Features (9)
**Original (3):**
- `hour` (0-23)
- `day_of_week` (0-6)
- `month` (1-12)

**Cyclical Encoded (6):**
- `hour_sin`, `hour_cos`
- `day_sin`, `day_cos`
- `month_sin`, `month_cos`

#### Why Cyclical Encoding?
> "Hour 23 and hour 0 are one hour apart, but numerically they're 23 units apart. Cyclical encoding using sine and cosine preserves the circular nature of time, helping the model understand that midnight wraps around."

**Formula:**
```python
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

#### Location Features (50)
- `latitude`, `longitude` (continuous)
- `district_*` (one-hot encoded for 50 Bangkok districts)

---

## 6. Pipeline Validation (1 min)

### Quality Checks Implemented
- ✅ Data type validation
- ✅ Range checks (lat/lon bounds)
- ✅ Duplicate detection
- ✅ Null value reporting
- ✅ Sample size thresholds

### Output Files
- `data/processed/traffy_weather_final.csv` - Ready for ML
- `training_summary.csv` - Model performance metrics
- Modular functions in `pipeline/` directory

---

## 7. Key Takeaways (30 sec)

### Pipeline Highlights
1. **Integrated 3 diverse data sources** into unified dataset
2. **70 engineered features** from raw environmental and temporal data
3. **Modular architecture** for maintainability and scalability
4. **Robust data cleaning** handling real-world quality issues
5. **Smart aggregation** balancing granularity with availability

### Transition to ML
> "This pipeline transforms raw, messy data from multiple sources into a clean, feature-rich dataset. Now let's see how we use machine learning to extract insights from these 70 features."

---

## Speaking Tips

### Do's
✅ Use the high-level diagram to guide your narrative  
✅ Emphasize real-world challenges you solved  
✅ Explain technical decisions (left join, cyclical encoding)  
✅ Show confidence in your data quality measures  

### Don'ts
❌ Don't read code line by line  
❌ Don't assume audience knows pandas operations  
❌ Don't skip over the "why" behind decisions  
❌ Don't forget to connect pipeline to ML results  

### Anticipate Questions
- "Why left join instead of inner join?" → Preserve all complaints
- "How did you handle missing data?" → Multiple strategies by context
- "Why cyclical encoding?" → Temporal continuity
- "How long does the pipeline take to run?" → ~5-10 minutes with scraping
