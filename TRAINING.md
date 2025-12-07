# Model Training Overview

## High-Level Process

Multi-label classification using Random Forest models to predict complaint types based on environmental and temporal features.

---

## 1. Training Strategy

### Multi-Label Approach
- **One model per complaint type** (e.g., separate models for "ถนน", "ไฟฟ้า", "น้ำท่วม")
- Each model answers: "Will this type of complaint occur given these conditions?"
- Binary classification: Yes (1) or No (0)

### Why Multi-Label?
- Complaints can have multiple types simultaneously
- Allows independent prediction for each type
- More flexible than single multi-class model

---

## 2. Model Selection & Hyperparameters

### Algorithm
**Random Forest Classifier**
- Ensemble of decision trees
- Handles non-linear relationships
- Provides feature importance scores
- Robust to outliers

### Hyperparameter Tuning
**RandomizedSearchCV** with cross-validation:
```python
param_distributions = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
```

**Search Configuration**:
- `n_iter=5`: Tests 5 random combinations
- `cv=3`: 3-fold cross-validation
- Scores on F1 metric (balances precision & recall)

---

## 3. Class Imbalance Handling

### Problem
- Some complaint types are rare (e.g., 1% positive samples)
- Model biased toward predicting "No complaint" always

### Solution: Adaptive Resampling
```python
if imbalance_ratio > 10:
    # Severe imbalance → SMOTE (synthetic oversampling)
elif imbalance_ratio > 3:
    # Moderate imbalance → Random oversampling
else:
    # Balanced → No resampling
```

**SMOTE**: Creates synthetic examples of minority class  
**Oversampling**: Duplicates minority class examples

---

## 4. Training Process

### For Each Complaint Type:

1. **Check minimum samples**
   - Skip if <50 positive examples
   - Not enough data to train reliably

2. **Prepare data**
   - X: All features (temporal, weather, air quality, location)
   - y: Binary target (1 = this type occurred, 0 = didn't)

3. **Handle imbalance**
   - Apply SMOTE or oversampling if needed

4. **Train with hyperparameter search**
   - RandomizedSearchCV finds best parameters
   - 3-fold cross-validation for robustness

5. **Evaluate on test set**
   - Metrics: Accuracy, Precision, Recall, F1
   - F1 prioritized (balances false positives & negatives)

6. **Save model**
   - Pickle file: `type_ถนน_model.pkl`
   - Feature names: `feature_names.pkl`

---

## 5. Model Outputs

### Per Model
```
data/models/
├── type_ถนน_model.pkl          # Trained model
├── type_ไฟฟ้า_model.pkl
├── type_น้ำท่วม_model.pkl
└── ...
```

### Training Summary
```csv
type,accuracy,precision,recall,f1,n_estimators,max_depth,model_file
ถนน,0.87,0.82,0.79,0.80,200,30,type_ถนน_model.pkl
ไฟฟ้า,0.91,0.88,0.85,0.86,300,20,type_ไฟฟ้า_model.pkl
```

**Saved to**: `data/models/training_summary.csv`

---

## 6. Feature Importance

### What It Shows
- Each feature's contribution to predictions (0-1 scale)
- Higher score = more important for decisions

### Example
```
pm25:           0.15  (15% of prediction power)
hour:           0.12  (time of day matters)
temperature:    0.08
latitude:       0.05
```

### Interpretation
- PM2.5 levels strongly predict certain complaint types
- Time of day is significant (rush hour, evening, etc.)
- Weather conditions influence complaint patterns

---

## 7. Training Execution

```bash
# Full training (all types, full hyperparameter search)
python main.py

# Fast training with sampling
python main.py --sample 200000 --n-iter 5

# Adjust minimum samples threshold
python main.py --min-samples 100

# Only preprocessing (skip training)
python main.py --skip-training
```

### Training Time
- **Full dataset**: ~30-60 minutes per type
- **Sampled (200K)**: ~5-10 minutes per type
- **Total**: Depends on number of viable complaint types

---

## Training Flow Summary

```
Preprocessed Data
  ↓
For Each Complaint Type:
  ├─ Check sample size (≥50)
  ├─ Split train/test (80/20)
  ├─ Handle class imbalance (SMOTE/oversample)
  ├─ Hyperparameter search (RandomizedSearchCV)
  ├─ Train best model (Random Forest)
  ├─ Evaluate (Accuracy, F1, etc.)
  └─ Save model & metrics
  ↓
Training Summary Report
```

---

## Model Performance Interpretation

### Good Model
- **F1 > 0.7**: Reliable predictions
- **Precision > 0.8**: Few false alarms
- **Recall > 0.7**: Catches most actual cases

### Poor Model
- **F1 < 0.5**: Not better than random guessing
- Usually due to insufficient data or weak signal

### Use Case
Models help understand:
- **When** complaints are likely (temporal patterns)
- **Where** complaints concentrate (location patterns)
- **Why** complaints occur (environmental triggers)
