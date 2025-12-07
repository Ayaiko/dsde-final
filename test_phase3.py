"""Test Phase 3: Multiple Binary Classifiers"""
import pandas as pd
import numpy as np
from pipeline.train import (
    prepare_features_for_training,
    get_trainable_types,
    train_single_type_model,
    train_all_types
)
from pipeline.preprocess import create_binary_targets

print("="*80)
print("Testing Phase 3: Multiple Binary Classifiers")
print("="*80)

# Create sample data
np.random.seed(42)
n_samples = 500

df = pd.DataFrame({
    'feature_1': np.random.randn(n_samples),
    'feature_2': np.random.randn(n_samples),
    'feature_3': np.random.randn(n_samples),
    'district_A': np.random.randint(0, 2, n_samples),
    'district_B': np.random.randint(0, 2, n_samples),
    'hour_sin': np.random.randn(n_samples),
    'hour_cos': np.random.randn(n_samples),
    'temperature_2m (°C)': np.random.uniform(20, 35, n_samples),
    'rain (mm)': np.random.uniform(0, 50, n_samples),
    'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
    'type': [['ถนน'] if np.random.rand() > 0.7 else 
             ['ทางเท้า'] if np.random.rand() > 0.85 else 
             ['น้ำท่วม'] for _ in range(n_samples)]
})

# Create binary targets
df, binary_cols = create_binary_targets(df)
print(f"\nCreated {len(binary_cols)} binary target columns")

# Test 1: prepare_features_for_training
print("\n" + "-"*80)
print("Test 1: prepare_features_for_training()")
print("-"*80)
X, feature_names = prepare_features_for_training(df)
print(f"Features extracted: {len(feature_names)}")
print(f"Feature matrix shape: {X.shape}")
print(f"Sample features: {feature_names[:5]}")

# Test 2: get_trainable_types
print("\n" + "-"*80)
print("Test 2: get_trainable_types()")
print("-"*80)
trainable = get_trainable_types(df, min_samples=20)
print(f"Trainable types (min 20 samples): {len(trainable)}")
for col in trainable:
    count = (df[col] == 1).sum()
    print(f"  {col}: {count} positive samples")

# Test 3: train_single_type_model
print("\n" + "-"*80)
print("Test 3: train_single_type_model() - Single Type")
print("-"*80)
if len(trainable) > 0:
    test_type = trainable[0]
    print(f"Training model for: {test_type}")
    
    result = train_single_type_model(
        df, test_type, feature_names, n_iter=3, 
        adaptive_resampling=True, random_state=42
    )
    
    print(f"\n✓ Model trained successfully")
    print(f"  Minority ratio: {result['minority_ratio']:.2%}")
    print(f"  F1 Score: {result['metrics']['f1']:.4f}")
    print(f"  Precision: {result['metrics']['precision']:.4f}")
    print(f"  Recall: {result['metrics']['recall']:.4f}")
    
    if result['resampling_stats']:
        print(f"  Resampling strategy: {result['resampling_stats']['strategy']}")

# Test 4: train_all_types (with very small n_iter for speed)
print("\n" + "-"*80)
print("Test 4: train_all_types() - All Types")
print("-"*80)
print("Training all viable types (n_iter=2 for speed)...")

results = train_all_types(
    df, n_iter=2, min_samples=20, 
    output_dir='data/models/test', 
    adaptive_resampling=True
)

print(f"\n✓ Successfully trained {len(results)} models")

# Verify files were created
import os
print("\nVerifying saved files:")
if os.path.exists('data/models/test/feature_names.pkl'):
    print("  ✓ feature_names.pkl created")
else:
    print("  ✗ feature_names.pkl NOT found")

for type_name in results.keys():
    filepath = f'data/models/test/{type_name}_model.pkl'
    if os.path.exists(filepath):
        print(f"  ✓ {type_name}_model.pkl created")
    else:
        print(f"  ✗ {type_name}_model.pkl NOT found")

print("\n" + "="*80)
print("✓ Phase 3 tests completed!")
print("="*80)
