"""
Model Training Module
Train, evaluate, and save machine learning models
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as sklearn_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import pickle
import os
from datetime import datetime


def split_data(df, target_col, feature_cols=None, exclude_cols=None, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Full dataset
    target_col : str or list
        Name of target column(s)
    feature_cols : list, optional
        List of feature column names. If None, uses all columns except target and exclude_cols
    exclude_cols : list, optional
        List of column names to exclude from features (e.g., IDs, timestamps)
    test_size : float
        Proportion of data for testing (default 0.2)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    if feature_cols is None:
        # Use all columns except target and excluded columns
        if isinstance(target_col, list):
            exclude_list = target_col.copy()
        else:
            exclude_list = [target_col]
        
        if exclude_cols:
            exclude_list.extend(exclude_cols)
        
        feature_cols = [col for col in df.columns if col not in exclude_list]
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = sklearn_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    print(f"Features: {len(feature_cols)}")
    
    return X_train, X_test, y_train, y_test


def train_classifier(X_train, y_train, n_estimators=100, max_depth=None, 
                     random_state=42, multi_output=False):
    """
    Train Random Forest classifier
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series or DataFrame
        Training target(s)
    n_estimators : int
        Number of trees in the forest
    max_depth : int, optional
        Maximum depth of trees
    random_state : int
        Random seed
    multi_output : bool
        If True, wraps classifier for multi-label classification
    
    Returns:
    --------
    model
        Trained classifier
    """
    print(f"\nTraining Random Forest Classifier...")
    print(f"  Estimators: {n_estimators}")
    print(f"  Max depth: {max_depth if max_depth else 'None (full depth)'}")
    
    base_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1  # Use all CPU cores
    )
    
    if multi_output:
        print("  Multi-output mode: Enabled")
        model = MultiOutputClassifier(base_model, n_jobs=-1)
    else:
        model = base_model
    
    model.fit(X_train, y_train)
    print("✓ Training complete")
    
    return model


def evaluate_classifier(model, X_test, y_test, target_names=None):
    """
    Evaluate classifier performance
    
    Parameters:
    -----------
    model : trained classifier
        Model to evaluate
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series or DataFrame
        True labels
    target_names : list, optional
        Names of target classes for classification report
    
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("CLASSIFICATION METRICS")
    print("="*60)
    
    y_pred = model.predict(X_test)
    
    # Handle multi-output case
    if isinstance(y_test, pd.DataFrame) or (isinstance(y_test, np.ndarray) and y_test.ndim > 1):
        # Multi-label classification
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_micro': precision_score(y_test, y_pred, average='micro', zero_division=0),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(y_test, y_pred, average='micro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_test, y_pred, average='micro', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        }
    else:
        # Single-output classification
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
    
    # Print metrics
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    
    # Classification report
    try:
        print("\n" + "-"*60)
        print("CLASSIFICATION REPORT")
        print("-"*60)
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    except Exception as e:
        print(f"Could not generate classification report: {e}")
    
    metrics['predictions'] = y_pred
    
    return metrics


def cross_validate_model(model, X, y, cv=5, scoring='accuracy'):
    """
    Perform cross-validation
    
    Parameters:
    -----------
    model : sklearn model
        Model to validate
    X : pandas.DataFrame
        Features
    y : pandas.Series
        Target
    cv : int
        Number of folds
    scoring : str
        Scoring metric
    
    Returns:
    --------
    dict
        Cross-validation results
    """
    print(f"\nPerforming {cv}-fold cross-validation...")
    print(f"Scoring metric: {scoring}")
    
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    results = {
        'scores': scores,
        'mean': scores.mean(),
        'std': scores.std()
    }
    
    print(f"Scores: {scores}")
    print(f"Mean: {results['mean']:.4f} (+/- {results['std']*2:.4f})")
    
    return results


def get_feature_importance(model, feature_names):
    """
    Extract feature importance from trained model
    
    Parameters:
    -----------
    model : trained model
        Model with feature_importances_ attribute
    feature_names : list
        List of feature names
    
    Returns:
    --------
    pandas.DataFrame
        Feature importance sorted by importance
    """
    # Handle MultiOutputClassifier
    if hasattr(model, 'estimators_'):
        # Average importance across all estimators
        importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print("Model does not have feature_importances_ attribute")
        return None
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*60)
    print("TOP 10 FEATURE IMPORTANCE")
    print("="*60)
    print(importance_df.head(10).to_string(index=False))
    
    return importance_df


def save_model(model, filepath, metadata=None):
    """
    Save trained model to disk
    
    Parameters:
    -----------
    model : trained model
        Model to save
    filepath : str
        Path to save model (including filename)
    metadata : dict, optional
        Additional metadata to save with model
    
    Returns:
    --------
    str
        Path to saved model
    """
    # Create directory if doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Prepare save package
    save_data = {
        'model': model,
        'saved_at': datetime.now().isoformat(),
        'metadata': metadata or {}
    }
    
    # Save with pickle
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"\n✓ Model saved to: {filepath}")
    print(f"  File size: {os.path.getsize(filepath) / 1024:.2f} KB")
    
    return filepath


def load_model(filepath):
    """
    Load trained model from disk
    
    Parameters:
    -----------
    filepath : str
        Path to saved model
    
    Returns:
    --------
    dict
        Dictionary with model and metadata
    """
    with open(filepath, 'rb') as f:
        save_data = pickle.load(f)
    
    print(f"✓ Model loaded from: {filepath}")
    print(f"  Saved at: {save_data.get('saved_at', 'Unknown')}")
    
    return save_data


# ============================================================================
# NEW: Multiple Binary Classifiers Architecture (Phase 3)
# ============================================================================

def prepare_features_for_training(df):
    """
    Extract and prepare features for model training.
    Ensures consistent feature extraction for both training and prediction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with all columns including features and targets
    
    Returns:
    --------
    tuple
        (X, feature_names)
        X : pandas.DataFrame - Feature matrix
        feature_names : list - List of feature column names
    
    Notes:
    ------
    Excludes: object columns, timestamp columns, type_* target columns
    Handles missing values with fillna(0)
    """
    # Identify columns to exclude
    exclude_cols = (
        df.select_dtypes(include=['object', 'datetime', 'datetime64']).columns.tolist() +
        ['timestamp', 'timestamp_col', 'timestamp_hour', 'date', 'time'] +
        [c for c in df.columns if c.startswith('type_')]
    )
    
    # Select feature columns
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Extract features and handle missing values
    X = df[feature_cols].fillna(0)
    feature_names = feature_cols
    
    return X, feature_names


def get_trainable_types(df, min_samples=50):
    """
    Get list of trainable complaint types based on minimum sample threshold.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with binary type_* columns
    min_samples : int
        Minimum number of positive samples required (default: 50)
    
    Returns:
    --------
    list
        List of type column names sorted by frequency (most common first)
    
    Notes:
    ------
    Filters out types with insufficient positive samples for SMOTE (k_neighbors=5)
    """
    type_cols = [c for c in df.columns if c.startswith('type_')]
    
    # Count positive samples for each type
    type_counts = {}
    for col in type_cols:
        count = (df[col] == 1).sum()
        if count >= min_samples:
            type_counts[col] = count
    
    # Sort by frequency (descending)
    sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    trainable_types = [col for col, count in sorted_types]
    
    return trainable_types


def train_single_type_model(df, type_col, feature_names, n_iter=5, 
                            adaptive_resampling=True, random_state=42):
    """
    Train a single binary classifier for one complaint type.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Full dataframe with features and target
    type_col : str
        Name of binary target column (e.g., 'type_ถนน')
    feature_names : list
        List of feature column names to use
    n_iter : int
        Number of RandomizedSearchCV iterations (default: 5)
    adaptive_resampling : bool
        Whether to use adaptive resampling strategy (default: True)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'model': trained RandomForestClassifier
        - 'metrics': dict with accuracy, precision, recall, f1, best_params
        - 'minority_ratio': original class distribution
        - 'resampling_stats': statistics from resampling (if applied)
    """
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint
    from pipeline.resampling import apply_adaptive_resampling
    
    # Prepare features and target
    X = df[feature_names].fillna(0)
    y = df[type_col]
    
    minority_ratio = y.sum() / len(y)
    
    # Apply adaptive resampling if enabled
    resampling_stats = None
    if adaptive_resampling:
        X_res, y_res, resampling_stats = apply_adaptive_resampling(
            X, y, strategy='auto', minority_ratio=minority_ratio, random_state=random_state
        )
    else:
        X_res, y_res = X, y
    
    # Train/test split with stratification
    X_train, X_test, y_train, y_test = sklearn_split(
        X_res, y_res, test_size=0.2, random_state=random_state, stratify=y_res
    )
    
    # Optimized parameter distribution (from notebook)
    param_dist = {
        'n_estimators': randint(100, 301),
        'max_depth': randint(10, 31),
        'min_samples_split': randint(2, 6),
        'min_samples_leaf': randint(1, 4),
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    # Train with RandomizedSearchCV
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    random_search = RandomizedSearchCV(
        rf, param_dist, n_iter=n_iter, cv=2, scoring='f1', 
        random_state=random_state, n_jobs=-1, verbose=0
    )
    random_search.fit(X_train, y_train)
    
    # Evaluate on test set
    model = random_search.best_estimator_
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'best_params': random_search.best_params_
    }
    
    return {
        'model': model,
        'metrics': metrics,
        'minority_ratio': minority_ratio,
        'resampling_stats': resampling_stats
    }


def save_model_simple(model, type_name, output_dir='data/models'):
    """
    Save model in simple format (raw model, not wrapped in dict).
    Compatible with load_model.ipynb expectations.
    
    Parameters:
    -----------
    model : trained model
        Model to save
    type_name : str
        Type name (e.g., 'ถนน', 'ทางเท้า')
    output_dir : str
        Directory to save models
    
    Returns:
    --------
    str
        Path to saved model file
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{type_name}_model.pkl"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    return filepath


def save_feature_names(feature_names, output_dir='data/models'):
    """
    Save shared feature names list for all models.
    Used by load_model.ipynb to map feature_importances_.
    
    Parameters:
    -----------
    feature_names : list
        List of feature column names
    output_dir : str
        Directory to save file
    
    Returns:
    --------
    str
        Path to saved feature_names.pkl
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'feature_names.pkl')
    
    with open(filepath, 'wb') as f:
        pickle.dump(feature_names, f)
    
    return filepath


def train_all_types(df, n_iter=5, min_samples=50, output_dir='data/models', 
                    adaptive_resampling=True, random_state=42):
    """
    Train individual binary classifiers for all viable complaint types.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed dataframe with binary type_* columns and features
    n_iter : int
        Number of RandomizedSearchCV iterations per model (default: 5)
    min_samples : int
        Minimum positive samples required to train a type (default: 50)
    output_dir : str
        Directory to save models (default: 'data/models')
    adaptive_resampling : bool
        Whether to use adaptive resampling (default: True)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Dictionary with type names as keys and result dicts as values.
        Each result contains: model, metrics, filepath, feature_importance
    """
    import time
    from pipeline.resampling import print_resampling_stats
    
    print("\n" + "="*80)
    print("TRAINING MULTIPLE BINARY CLASSIFIERS")
    print("="*80)
    
    # Get trainable types
    trainable_types = get_trainable_types(df, min_samples=min_samples)
    print(f"\nFound {len(trainable_types)} trainable types (≥{min_samples} samples)")
    
    # Prepare features once
    X, feature_names = prepare_features_for_training(df)
    print(f"Features prepared: {len(feature_names)} features")
    
    # Save feature names once (shared across all models)
    feature_names_path = save_feature_names(feature_names, output_dir)
    print(f"✓ Feature names saved: {feature_names_path}\n")
    
    results = {}
    start_time = time.time()
    
    # Train each type
    for idx, type_col in enumerate(trainable_types):
        type_name = type_col.replace('type_', '')
        
        print("="*80)
        print(f"[{idx+1}/{len(trainable_types)}] Training: {type_name}")
        print("="*80)
        
        # Show class distribution
        positive_count = (df[type_col] == 1).sum()
        minority_ratio = positive_count / len(df)
        print(f"Positive samples: {positive_count:,} ({minority_ratio:.2%})")
        
        # Train model
        result = train_single_type_model(
            df, type_col, feature_names, n_iter=n_iter,
            adaptive_resampling=adaptive_resampling, random_state=random_state
        )
        
        # Print resampling stats if available
        if result['resampling_stats']:
            print_resampling_stats(result['resampling_stats'])
        
        # Save model
        filepath = save_model_simple(result['model'], type_name, output_dir)
        
        # Print metrics
        metrics = result['metrics']
        print(f"\n✓ Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Model saved: {filepath}")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': result['model'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 Features:")
        for i, row in importance_df.head(5).iterrows():
            print(f"  {row['feature']:40s} {row['importance']:.6f}")
        
        # Store results
        results[type_name] = {
            'model': result['model'],
            'metrics': metrics,
            'filepath': filepath,
            'feature_importance': importance_df
        }
        
        print()
    
    elapsed = time.time() - start_time
    
    # Print summary
    print("="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Models trained: {len(results)}/{len(trainable_types)}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Average per model: {elapsed/len(results):.1f} seconds")
    print(f"\nModels sorted by F1 score:")
    print("-"*80)
    
    for type_name, result in sorted(results.items(), key=lambda x: x[1]['metrics']['f1'], reverse=True):
        f1 = result['metrics']['f1']
        print(f"  {type_name:30s} F1={f1:.4f}")
    
    print("="*80)
    
    return results


# Standalone test
if __name__ == "__main__":
    print("Testing train.py module...")
    
    # Create sample data
    from sklearn.datasets import make_classification
    
    # Test classification
    print("\n" + "="*60)
    print("TESTING CLASSIFICATION")
    print("="*60)
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                                n_classes=3, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_df = pd.Series(y, name='target')
    
    X_train, X_test, y_train, y_test = split_data(
        pd.concat([X_df, y_df], axis=1), 
        target_col='target'
    )
    
    clf = train_classifier(X_train, y_train, n_estimators=50, max_depth=10)
    clf_metrics = evaluate_classifier(clf, X_test, y_test)
    importance = get_feature_importance(clf, X_train.columns.tolist())
    
    # Test save/load
    print("\n" + "="*60)
    print("TESTING SAVE/LOAD")
    print("="*60)
    save_model(clf, 'data/models/test_model.pkl', metadata={'type': 'classifier', 'test': True})
    loaded = load_model('data/models/test_model.pkl')
    
    print("\n✓ All tests passed!")
