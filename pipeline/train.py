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
