# AI/ML Presentation Guide

## 1. Problem Formulation (1-2 min)

**Visual Aid**: Show `draw_ai_pipeline.xml` - Data Preparation section

### Machine Learning Setup

#### Multi-Label Classification
> "Unlike typical classification where each sample has one label, Bangkok complaints can have multiple types. For example, a single complaint might be tagged as both 'sidewalk' and 'flooding'. This requires a different approach."

**Strategy**: One-vs-Rest (Binary classifier per type)
- Train 26 separate models (one per complaint type)
- Each model answers: "Does this complaint have type X?"
- Independent predictions allow multiple positive labels

#### Dataset Split
- **80/20 split**: 80% training, 20% testing
- **Stratified sampling**: Maintain class distribution in both sets
- **Why stratified?** Prevents test set bias, especially for rare types


## 2. Class Imbalance Handling (2 min)

**Visual Aid**: Point to Training Loop section in diagram

### The Imbalance Problem


> "Class imbalance is a major challenge. If 95% of data is negative, a naive model can achieve 95% accuracy by always predicting 'no'. But that's useless for prediction."

### Adaptive Solutions

#### Strategy 1: Sample Threshold
**Rule**: Skip types with fewer than 50 positive samples
```python
if positive_count < 50:
    print(f"Skipping {type_name}: insufficient samples")
    continue
```
**Why 50?** Minimum for meaningful train/test split and cross-validation

#### Strategy 2: SMOTE (High Imbalance)
**When**: Positive class < 10% of total
**Method**: Synthetic Minority Oversampling Technique
- Creates synthetic samples in feature space
- Interpolates between existing minority samples
- More sophisticated than simple duplication

```python
from imblearn.over_sampling import SMOTE
if positive_ratio < 0.1:
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
```

#### Strategy 3: Random Oversampling (Moderate Imbalance)
**When**: Positive class 10-30% of total
**Method**: Randomly duplicate minority samples with slight noise
- Simpler than SMOTE
- Faster computation
- Effective for moderate imbalance

### Why Adaptive?
> "Different types have different imbalance levels. A one-size-fits-all approach would either oversample common types unnecessarily or undersample rare types dangerously. Our adaptive strategy applies the right technique based on the specific class ratio."

---

## 3. Model Selection & Training (2-3 min)

### Why Random Forest?

#### Advantages for Our Problem

**1. Handles Non-Linearity**
- Relationships between features aren't linear
- Example: Temperature effect on complaints isn't constant
- Decision trees naturally capture complex interactions

**2. Robust to Outliers**
- Extreme weather events won't break the model
- Each tree uses bootstrap sampling
- Voting mechanism reduces outlier impact

**3. No Feature Scaling Required**
- Temperature (0-40°C) and PM2.5 (0-200 μg/m³) on different scales
- Random Forest doesn't care about scale
- Saves preprocessing time

**4. Feature Importance**
- Built-in Gini importance scores
- Helps us understand which factors matter
- Valuable for actionable insights

**5. Proven Performance**
- Industry standard for tabular data
- Handles mixed feature types (continuous + categorical)
- Reliable baseline before trying complex models

### Random Forest Architecture

```
Ensemble of Decision Trees
├── Tree 1: Bootstrap sample 1 → Vote
├── Tree 2: Bootstrap sample 2 → Vote  
├── Tree 3: Bootstrap sample 3 → Vote
└── ...
    └── Final Prediction: Majority Vote
```

**Key Hyperparameters:**
- `n_estimators`: Number of trees (100-500)
- `max_depth`: Maximum tree depth (10-50)
- `min_samples_split`: Minimum samples to split node (2-20)
- `min_samples_leaf`: Minimum samples in leaf (1-10)

### Hyperparameter Tuning: RandomizedSearchCV

#### Why Randomized (not Grid)?
> "Grid search tries every combination - could take days with our parameter space. Randomized search samples combinations randomly, finding good parameters faster with 5 iterations."

**Configuration:**
```python
param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 10]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions=param_distributions,
    n_iter=5,           # Try 5 random combinations
    cv=3,               # 3-fold cross validation
    scoring='f1',       # Optimize for F1 score
    random_state=42
)
```

#### 3-Fold Cross Validation
- Split training data into 3 parts
- Train on 2 parts, validate on 1
- Rotate and average results
- Prevents overfitting to single validation set

---

## 4. Evaluation Metrics (2 min)

**Visual Aid**: Show Evaluation section of diagram

### Understanding the Metrics

#### Confusion Matrix Basics
```
                Predicted
              Positive | Negative
Actual ────┼──────────┼──────────
Positive   │    TP    │    FN
Negative   │    FP    │    TN
```

#### Accuracy
**Formula**: (TP + TN) / Total  
**Meaning**: Overall correctness  
**Limitation**: Misleading with imbalanced classes

#### Precision
**Formula**: TP / (TP + FP)  
**Meaning**: Of predicted positives, how many are correct?  
**Use Case**: When false positives are costly  
> "If we predict a flooding complaint, how confident should the response team be?"

#### Recall
**Formula**: TP / (TP + FN)  
**Meaning**: Of actual positives, how many did we catch?  
**Use Case**: When missing positives is costly  
> "Are we catching all the actual flooding complaints?"

#### F1 Score
**Formula**: 2 × (Precision × Recall) / (Precision + Recall)  
**Meaning**: Harmonic mean balancing precision and recall  
**Why Harmonic**: Punishes extreme imbalance between the two  
> "F1 is our primary metric because it balances both false positives and false negatives."

#### ROC-AUC
**Range**: 0.5 (random) to 1.0 (perfect)  
**Meaning**: Model's ability to distinguish classes  
**Advantage**: Threshold-independent  

### What Good Scores Look Like

| Metric | Poor | Fair | Good | Excellent |
|--------|------|------|------|-----------|
| Accuracy | < 60% | 60-70% | 70-85% | > 85% |
| Precision | < 50% | 50-70% | 70-85% | > 85% |
| Recall | < 50% | 50-70% | 70-85% | > 85% |
| F1 Score | < 0.5 | 0.5-0.7 | 0.7-0.85 | > 0.85 |
| ROC-AUC | < 0.6 | 0.6-0.75 | 0.75-0.9 | > 0.9 |

---

## 5. Results Overview (2-3 min)

**Visual Aid**: Reference `training_summary.csv`

### Overall Performance

#### Models Trained
- **26 complaint types** successfully trained
- Each type has independent model
- Stored as pickle files in `models/` directory

#### Top Performing Types (Example)
```
Traffic (จราจร): 
  Accuracy: 89%, Precision: 87%, Recall: 85%, F1: 0.86

Sidewalk (ทางเท้า):
  Accuracy: 86%, Precision: 84%, Recall: 82%, F1: 0.83

Flooding (น้ำท่วม):
  Accuracy: 82%, Precision: 79%, Recall: 81%, F1: 0.80
```

#### Model Performance Patterns
> "Common complaint types generally perform better because we have more training data. Types with clear environmental triggers (flooding after rain) show higher accuracy."

### Feature Importance Insights

#### Top Contributing Features

**1. Air Quality Matters**
- PM2.5 strongly correlates with air quality complaints
- O3 levels predict health-related complaints
- Particulate matter affects outdoor activity complaints

**2. Temporal Patterns**
- **Hour**: Traffic complaints peak 7-9 AM, 5-7 PM
- **Day of week**: Noise complaints higher on weekends
- **Month**: Flooding complaints seasonal (rainy season)

**3. Location Insights**
- Certain districts have characteristic complaint types
- Central districts: Traffic, noise
- Flood-prone areas: Drainage, flooding
- Residential areas: Street lighting, sidewalk

#### Actionable Insights
> "These patterns tell us when and where to expect certain complaints. For example, the city could pre-deploy flood response teams to high-risk districts when rain is forecasted."

### Real-World Application

**Scenario**: Heavy rain predicted tomorrow
```
Model predicts:
├── Flooding complaints: ↑ 300%
├── Traffic complaints: ↑ 150%
├── Drainage complaints: ↑ 200%
└── High-risk districts: [District A, B, C]

City Response:
├── Pre-position water pumps
├── Alert traffic management
└── Dispatch teams to predicted hotspots
```

---

## 6. Key Technical Decisions (1 min)

### Decision 1: Multi-Label vs Multi-Class
**Chose Multi-Label** because:
- Complaints naturally have multiple categories
- No artificial choice between overlapping types
- More realistic model of citizen complaints

### Decision 2: Per-Type Models vs Single Multi-Output
**Chose Per-Type Models** because:
- Each type has different class balance
- Can tune hyperparameters independently
- Easier to debug and interpret
- Can train/deploy types separately

### Decision 3: Left Join in Pipeline
**Chose Left Join** because:
- Preserves all complaint records
- Missing weather data shouldn't delete complaints
- Can handle partial data gracefully

### Decision 4: Cyclical Encoding
**Chose Sin/Cos Encoding** because:
- Captures temporal continuity (23:00 → 00:00)
- Maintains both phase and magnitude
- Proven technique in time series ML

### Decision 5: Feature Engineering Over Deep Learning
**Chose Engineered Features** because:
- Interpretable results matter for city planning
- Limited data for deep learning
- Random Forest performs well on tabular data
- Feature importance provides insights

---

## 7. Limitations & Future Work (1-2 min)

### Current Limitations

#### Data Limitations
- **Insufficient samples**: Some rare types skipped (< 50 samples)
- **Temporal range**: Limited to available data period
- **Missing features**: No holiday, events, or policy changes

#### Model Limitations
- **Linear feature engineering**: Could miss complex interactions
- **Static models**: Don't adapt to changing patterns
- **No text analysis**: Ignoring complaint descriptions

### Future Enhancements

#### Short-Term Improvements
1. **More data sources**:
   - Public events calendar
   - Public holidays
   - Construction schedules
   - Social media sentiment

2. **Advanced features**:
   - Weather forecasts (prediction mode)
   - Historical trends
   - Spatial autocorrelation

3. **Model improvements**:
   - Ensemble methods (XGBoost, LightGBM)
   - Neural networks for text descriptions
   - Time series modeling

#### Long-Term Vision
1. **Real-time prediction**:
   - Deploy as API endpoint
   - Live dashboard for city officials
   - Automated alert system

2. **Prescriptive analytics**:
   - Recommend resource allocation
   - Optimize response routes
   - Predict maintenance needs

3. **Causal inference**:
   - Understand cause-effect relationships
   - Test policy interventions
   - Measure impact of city improvements

---

## 8. Wrap-Up & Key Messages (1 min)

### Three Main Achievements

**1. Successful Multi-Label Classification**
> "We built 26 independent models that can predict multiple complaint types simultaneously with 80%+ accuracy."

**2. Handled Real-World ML Challenges**
- Class imbalance: Adaptive SMOTE/oversampling
- Limited data: Smart threshold and validation
- Model selection: Random Forest for interpretability

**3. Actionable Insights**
- Environmental triggers identified
- Temporal and spatial patterns discovered
- Foundation for predictive city management

### The Big Picture
> "This project demonstrates that machine learning can transform citizen complaint data into predictive insights. By understanding when and where complaints are likely to occur, Bangkok can shift from reactive to proactive city management."

---

## Speaking Tips

### Do's
✅ Use the compact AI pipeline diagram to show flow  
✅ Explain metrics in plain language (avoid jargon)  
✅ Connect results to real-world impact  
✅ Show enthusiasm for the insights discovered  
✅ Be honest about limitations  

### Don'ts
❌ Don't assume audience knows SMOTE, F1 score  
❌ Don't just list metrics - explain what they mean  
❌ Don't skip the "why" behind model choices  
❌ Don't oversell accuracy - acknowledge limitations  

### Anticipate Questions

**Q: "Why Random Forest and not neural networks?"**  
A: "Random Forest is interpretable, handles tabular data well, and provides feature importance. Deep learning would need much more data and lose interpretability."

**Q: "How do you handle new complaint types?"**  
A: "We'd need to retrain. Future work includes online learning for continuous model updates."

**Q: "Can this predict individual complaints?"**  
A: "No, it predicts aggregate patterns. For example, 'expect 50 flooding complaints tomorrow in District X' not 'this specific person will complain.'"

**Q: "What's the ROI for the city?"**  
A: "Better resource allocation, faster response times, and proactive prevention could save significant costs. Exact ROI needs cost-benefit analysis."

**Q: "How often should models be retrained?"**  
A: "Monthly or quarterly, depending on data drift. We should monitor prediction performance continuously."

---

## Transition from Pipeline

> "Now that you've seen how we built a robust data pipeline, let's dive into the machine learning models that extract predictive insights from those 70 engineered features."

## Transition to Q&A

> "These 26 models form the foundation of a predictive system that can help Bangkok move from reactive complaint handling to proactive city management. I'm happy to answer any questions about our methodology, results, or future directions."
