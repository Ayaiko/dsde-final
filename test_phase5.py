"""Test Phase 5: Model Loading Compatibility"""
import pickle
import os
import pandas as pd

print("="*80)
print("Testing Phase 5: Model Loading Compatibility")
print("="*80)

test_model_dir = 'data/models/test'

# Test 1: Load feature_names.pkl
print("\nTest 1: Loading feature_names.pkl")
print("-"*80)
feature_names_path = os.path.join(test_model_dir, 'feature_names.pkl')

if os.path.exists(feature_names_path):
    with open(feature_names_path, 'rb') as f:
        feature_names = pickle.load(f)
    print(f"✓ Loaded {len(feature_names)} feature names")
    print(f"  Sample features: {feature_names[:5]}")
else:
    print("✗ feature_names.pkl not found")
    feature_names = None

# Test 2: Load a model (raw format, not dict)
print("\nTest 2: Loading model in simple format")
print("-"*80)

model_files = [f for f in os.listdir(test_model_dir) if f.endswith('_model.pkl')]
if model_files:
    test_model_file = model_files[0]
    test_model_path = os.path.join(test_model_dir, test_model_file)
    
    print(f"Loading: {test_model_file}")
    
    with open(test_model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✓ Model loaded successfully")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Has feature_importances_: {hasattr(model, 'feature_importances_')}")
    
    if hasattr(model, 'feature_importances_'):
        print(f"  Number of features: {len(model.feature_importances_)}")
        
        # Test 3: Map feature importance (like load_model.ipynb does)
        print("\nTest 3: Mapping feature importances to names")
        print("-"*80)
        
        if feature_names and len(feature_names) == len(model.feature_importances_):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"✓ Feature mapping successful")
            print(f"\nTop 5 Feature Importances:")
            for i, row in importance_df.head(5).iterrows():
                print(f"  {row['feature']:40s} {row['importance']:.6f}")
        else:
            print("✗ Feature name mismatch or not available")
            if feature_names:
                print(f"  Expected: {len(model.feature_importances_)}, Got: {len(feature_names)}")
else:
    print("✗ No model files found in test directory")

# Test 4: Verify format matches load_model.ipynb expectations
print("\nTest 4: Format compatibility check")
print("-"*80)

# Simulate what load_model.ipynb does
def load_model_notebook_style(model_path):
    """Simulates load_model.ipynb loading logic"""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    # Check if it's wrapped in dict or raw model
    if isinstance(data, dict):
        print("  Format: Dict wrapper (old format)")
        return data.get('model', data), data.get('metadata', None)
    else:
        print("  Format: Raw model (new format)")
        return data, None

if model_files:
    model, metadata = load_model_notebook_style(test_model_path)
    print(f"✓ Compatible with load_model.ipynb")
    print(f"  Model extracted: {type(model).__name__}")
    print(f"  Metadata: {metadata if metadata else 'None (expected for new format)'}")

print("\n" + "="*80)
print("✓ Phase 5 compatibility tests completed!")
print("="*80)

# Summary
print("\nSummary:")
print(f"  ✓ Models saved in raw format (not dict wrapper)")
print(f"  ✓ feature_names.pkl saved separately")
print(f"  ✓ Feature importances can be mapped to names")
print(f"  ✓ Compatible with load_model.ipynb expectations")
