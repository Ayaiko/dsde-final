from pipeline.extract import load_traffy_data, load_all_weather_data
from pipeline.transform import parse_timestamps, assign_grid_cells, merge_datasets
from pipeline.utils import split_coordinates, clean_traffy_data
from pipeline.load import save_merged_data
from pipeline.preprocess import (
    parse_type_column, filter_empty_types, drop_missing_weather,
    create_binary_targets, sample_data
)
from pipeline.features import prepare_features
from pipeline.train import train_all_types
import pandas as pd
import argparse


def main(skip_etl=False, skip_training=False, sample_size=None, train_n_iter=5, train_min_samples=50):
    """
    Complete ML pipeline from data extraction to model training.
    
    Parameters:
    -----------
    skip_etl : bool
        If True, loads existing traffy_weather_merged.csv instead of running ETL
    skip_training : bool
        If True, skips model training phase
    sample_size : int, optional
        Number of records to sample for faster training (e.g., 200000)
    train_n_iter : int
        Number of RandomizedSearchCV iterations per model (default: 5)
    train_min_samples : int
        Minimum positive samples required to train a type (default: 50)
    """
    
    if skip_etl:
        print("\n" + "="*80)
        print("LOADING EXISTING DATA")
        print("="*80)
        df_merged = pd.read_csv('data/processed/traffy_weather_merged.csv')
        print(f"✓ Loaded: {len(df_merged):,} records from traffy_weather_merged.csv")
    else:
        print("\n" + "="*80)
        print("ETL PHASE")
        print("="*80)
        # ETL
        df_traffy = load_traffy_data()
        df_traffy = clean_traffy_data(df_traffy)
        df_traffy = split_coordinates(df_traffy)
        
        df_weather = load_all_weather_data()
        
        df_traffy, df_weather = parse_timestamps(df_traffy, df_weather)
        df_traffy, df_weather = assign_grid_cells(df_traffy, df_weather, delta=0.1)
        
        df_merged = merge_datasets(df_traffy, df_weather)
        save_merged_data(df_merged)
    
    print("\n" + "="*80)
    print("PREPROCESSING PHASE")
    print("="*80)
    
    # Preprocessing
    df_merged = parse_type_column(df_merged)
    df_merged = filter_empty_types(df_merged)
    df_merged = drop_missing_weather(df_merged)
    print(f"After preprocessing: {len(df_merged):,} records")
    
    # Optional sampling
    if sample_size and sample_size < len(df_merged):
        print(f"\nSampling data for faster training...")
        df_merged = sample_data(df_merged, n=sample_size)
    
    print("\n" + "="*80)
    print("FEATURE ENGINEERING PHASE")
    print("="*80)
    
    # Feature Engineering
    df_merged = prepare_features(df_merged)
    
    # Create binary target columns for training
    print("\nCreating binary target columns...")
    df_merged, binary_cols = create_binary_targets(df_merged)
    print(f"✓ Created {len(binary_cols)} binary target columns")
    
    # Save final dataset
    output_path = 'data/processed/traffy_weather_final.csv'
    df_merged.to_csv(output_path, index=False)
    
    temp_col = 'temperature_2m (°C)' if 'temperature_2m (°C)' in df_merged.columns else 'temperature_2m'
    match_rate = (~df_merged[temp_col].isna()).sum() / len(df_merged) * 100 if temp_col in df_merged.columns else 0
    
    print(f"\n✓ Pipeline preprocessing complete: {len(df_merged):,} records, {len(df_merged.columns)} features, {match_rate:.1f}% matched")
    print(f"✓ Saved to: {output_path}")
    
    # Model Training Phase
    if not skip_training:
        print("\n" + "="*80)
        print("MODEL TRAINING PHASE")
        print("="*80)
        print(f"Configuration:")
        print(f"  - RandomizedSearchCV iterations: {train_n_iter}")
        print(f"  - Minimum samples per type: {train_min_samples}")
        print(f"  - Adaptive resampling: Enabled")
        print(f"  - Output directory: data/models/")
        
        # Train all viable types
        trained_results = train_all_types(
            df_merged, 
            n_iter=train_n_iter, 
            min_samples=train_min_samples,
            output_dir='data/models',
            adaptive_resampling=True
        )
        
        # Save training summary report
        summary_data = []
        for type_name, result in trained_results.items():
            metrics = result['metrics']
            summary_data.append({
                'type': type_name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'n_estimators': metrics['best_params'].get('n_estimators', None),
                'max_depth': metrics['best_params'].get('max_depth', None),
                'model_file': result['filepath']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('f1', ascending=False)
        summary_path = 'data/models/training_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n✓ Training summary saved to: {summary_path}")
        print(f"✓ Total models trained: {len(trained_results)}")
    else:
        print("\n⏭️  Skipping model training phase (--skip-training flag set)")
    
    print("\n" + "="*80)
    print("✓ COMPLETE PIPELINE FINISHED")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bangkok Traffy Complaint ML Pipeline')
    parser.add_argument('--skip-etl', action='store_true', 
                        help='Skip ETL phase, load existing traffy_weather_merged.csv')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip model training phase')
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample size for faster training (e.g., 200000)')
    parser.add_argument('--n-iter', type=int, default=5,
                        help='RandomizedSearchCV iterations per model (default: 5)')
    parser.add_argument('--min-samples', type=int, default=50,
                        help='Minimum positive samples to train a type (default: 50)')
    
    args = parser.parse_args()
    
    main(
        skip_etl=args.skip_etl,
        skip_training=args.skip_training,
        sample_size=args.sample,
        train_n_iter=args.n_iter,
        train_min_samples=args.min_samples
    )
