"""
Adaptive Resampling Module
Handles class imbalance with different strategies based on data distribution
"""
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


def determine_resampling_strategy(minority_ratio, 
                                   high_threshold=0.15, 
                                   low_threshold=0.05):
    """
    Determine appropriate resampling strategy based on minority class ratio.
    
    Parameters:
    -----------
    minority_ratio : float
        Ratio of positive samples to total samples (e.g., 0.08 for 8%)
    high_threshold : float
        Threshold for high frequency types (default: 0.15 = 15%)
    low_threshold : float
        Threshold for low frequency types (default: 0.05 = 5%)
    
    Returns:
    --------
    str
        Strategy name: 'class_weight' / 'undersampling' / 'smote_under'
    
    Strategy Rules:
    --------------
    - ratio >= 15%: 'class_weight' - No resampling, use balanced class weights
    - 5% <= ratio < 15%: 'undersampling' - Reduce majority class only
    - ratio < 5%: 'smote_under' - Generate synthetic minority + reduce majority
    """
    if minority_ratio >= high_threshold:
        return 'class_weight'
    elif minority_ratio >= low_threshold:
        return 'undersampling'
    else:
        return 'smote_under'


def apply_adaptive_resampling(X, y, strategy='auto', 
                              minority_ratio=None,
                              random_state=42,
                              smote_k_neighbors=5,
                              smote_sampling_strategy=0.2,
                              under_sampling_strategy=0.5):
    """
    Apply adaptive resampling strategy to handle class imbalance.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix
    y : pandas.Series or numpy.ndarray
        Target vector (binary: 0 or 1)
    strategy : str
        Resampling strategy: 'auto', 'class_weight', 'undersampling', 'smote_under'
        If 'auto', will determine strategy based on minority_ratio
    minority_ratio : float, optional
        Ratio of positive samples. Required if strategy='auto'
    random_state : int
        Random seed for reproducibility
    smote_k_neighbors : int
        Number of neighbors for SMOTE (default: 5)
    smote_sampling_strategy : float
        Target ratio after SMOTE oversampling (default: 0.2 = 20%)
    under_sampling_strategy : float
        Target ratio after undersampling (default: 0.5 = 50% for medium, 0.3 for SMOTE+under)
    
    Returns:
    --------
    tuple
        (X_resampled, y_resampled, stats_dict)
        stats_dict contains: strategy used, original counts, resampled counts
    
    Raises:
    -------
    ValueError
        If strategy='auto' but minority_ratio not provided
    """
    # Auto-determine strategy if needed
    if strategy == 'auto':
        if minority_ratio is None:
            raise ValueError("minority_ratio must be provided when strategy='auto'")
        strategy = determine_resampling_strategy(minority_ratio)
    
    # Get original counts
    original_counts = {
        'positive': int(np.sum(y)),
        'negative': int(len(y) - np.sum(y)),
        'total': int(len(y))
    }
    
    stats = {
        'strategy': strategy,
        'original': original_counts,
        'resampled': None,
        'error': None
    }
    
    try:
        if strategy == 'class_weight':
            # No resampling, just return original data
            X_res, y_res = X, y
            stats['resampled'] = original_counts.copy()
            
        elif strategy == 'undersampling':
            # Pure undersampling to 1:2 ratio (1 positive : 2 negative)
            under = RandomUnderSampler(
                random_state=random_state,
                sampling_strategy=under_sampling_strategy
            )
            X_res, y_res = under.fit_resample(X, y)
            
            stats['resampled'] = {
                'positive': int(np.sum(y_res)),
                'negative': int(len(y_res) - np.sum(y_res)),
                'total': int(len(y_res))
            }
            
        elif strategy == 'smote_under':
            # SMOTE oversampling + RandomUnderSampling
            # First SMOTE to 20%, then undersample to 30% (final ~1:2.3 ratio)
            smote = SMOTE(
                random_state=random_state,
                k_neighbors=smote_k_neighbors,
                sampling_strategy=smote_sampling_strategy
            )
            under = RandomUnderSampler(
                random_state=random_state,
                sampling_strategy=0.3  # Fixed for SMOTE+under combination
            )
            
            pipeline = ImbPipeline([('smote', smote), ('under', under)])
            X_res, y_res = pipeline.fit_resample(X, y)
            
            stats['resampled'] = {
                'positive': int(np.sum(y_res)),
                'negative': int(len(y_res) - np.sum(y_res)),
                'total': int(len(y_res))
            }
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
    except Exception as e:
        # If resampling fails, return original data
        stats['error'] = str(e)
        X_res, y_res = X, y
        stats['resampled'] = original_counts.copy()
    
    return X_res, y_res, stats


def print_resampling_stats(stats):
    """
    Print resampling statistics in a readable format.
    
    Parameters:
    -----------
    stats : dict
        Statistics dictionary from apply_adaptive_resampling()
    """
    print(f"  Strategy: {stats['strategy'].upper().replace('_', ' ')}")
    
    orig = stats['original']
    print(f"  Original: {orig['positive']:,} positive, {orig['negative']:,} negative ({orig['total']:,} total)")
    
    if stats['resampled'] and stats['strategy'] != 'class_weight':
        res = stats['resampled']
        change = ((res['total'] - orig['total']) / orig['total']) * 100
        print(f"  Resampled: {res['positive']:,} positive, {res['negative']:,} negative ({res['total']:,} total)")
        print(f"  Change: {change:+.1f}%")
    
    if stats['error']:
        print(f"  ⚠ Warning: {stats['error']}")
        print(f"  → Using original data")


# Standalone test
if __name__ == "__main__":
    import pandas as pd
    from sklearn.datasets import make_classification
    
    print("=" * 60)
    print("Testing Adaptive Resampling Module")
    print("=" * 60)
    
    # Test 1: High frequency (≥15%) - should use class_weight
    print("\nTest 1: High Frequency Type (22% positive)")
    print("-" * 60)
    X1, y1 = make_classification(n_samples=1000, n_features=10, 
                                  weights=[0.78, 0.22], random_state=42)
    ratio1 = np.sum(y1) / len(y1)
    strategy1 = determine_resampling_strategy(ratio1)
    print(f"Minority ratio: {ratio1:.2%}")
    print(f"Determined strategy: {strategy1}")
    
    X1_res, y1_res, stats1 = apply_adaptive_resampling(X1, y1, strategy='auto', 
                                                        minority_ratio=ratio1)
    print_resampling_stats(stats1)
    
    # Test 2: Medium frequency (5-15%) - should use undersampling
    print("\n\nTest 2: Medium Frequency Type (8% positive)")
    print("-" * 60)
    X2, y2 = make_classification(n_samples=1000, n_features=10, 
                                  weights=[0.92, 0.08], random_state=42)
    ratio2 = np.sum(y2) / len(y2)
    strategy2 = determine_resampling_strategy(ratio2)
    print(f"Minority ratio: {ratio2:.2%}")
    print(f"Determined strategy: {strategy2}")
    
    X2_res, y2_res, stats2 = apply_adaptive_resampling(X2, y2, strategy='auto', 
                                                        minority_ratio=ratio2)
    print_resampling_stats(stats2)
    
    # Test 3: Low frequency (<5%) - should use SMOTE+under
    print("\n\nTest 3: Low Frequency Type (3% positive)")
    print("-" * 60)
    X3, y3 = make_classification(n_samples=1000, n_features=10, 
                                  weights=[0.97, 0.03], random_state=42)
    ratio3 = np.sum(y3) / len(y3)
    strategy3 = determine_resampling_strategy(ratio3)
    print(f"Minority ratio: {ratio3:.2%}")
    print(f"Determined strategy: {strategy3}")
    
    X3_res, y3_res, stats3 = apply_adaptive_resampling(X3, y3, strategy='auto', 
                                                        minority_ratio=ratio3)
    print_resampling_stats(stats3)
    
    print("\n" + "=" * 60)
    print("✓ All resampling tests completed!")
    print("=" * 60)
