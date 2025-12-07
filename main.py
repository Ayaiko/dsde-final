from pipeline.extract import load_traffy_data, load_all_weather_data, load_single_location_weather
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
import asyncio
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'scrapers'))
from weather_scrape import download_api_weather_bangkok
from pm_scrape import download_aqi_bangkok


def main(skip_etl=False, skip_training=False, skip_scraping=True, sample_size=None, train_n_iter=5, train_min_samples=50):
    """
    Complete ML pipeline from data extraction to model training.
    
    Parameters:
    -----------
    skip_etl : bool
        If True, loads existing traffy_weather_merged.csv instead of running ETL
    skip_training : bool
        If True, skips model training phase
    skip_scraping : bool
        If True, skips weather and PM2.5/AQI data scraping (default: True)
    sample_size : int, optional
        Number of records to sample for faster training (e.g., 200000)
    train_n_iter : int
        Number of RandomizedSearchCV iterations per model (default: 5)
    train_min_samples : int
        Minimum positive samples required to train a type (default: 50)
    """
    
    if not skip_scraping:
        print("\n" + "="*80)
        print("WEATHER SCRAPING PHASE")
        print("="*80)
        asyncio.run(download_api_weather_bangkok("13.74", "100.50"))
        print(f"✓ Weather data scraped for coordinates: 13.74N, 100.50E")
        
        print("\n" + "="*80)
        print("PM2.5/AQI SCRAPING PHASE")
        print("="*80)
        download_aqi_bangkok()
        print(f"✓ PM2.5/AQI data scraped for Bangkok")
    
    if skip_etl:
        print("\n" + "="*80)
        print("LOADING EXISTING DATA")
        print("="*80)
        df_merged = pd.read_csv('data/processed/traffy_weather_merged.csv')
        print(f"✓ Loaded: {len(df_merged):,} records from traffy_weather_merged.csv")
    else:
        print("\n" + "="*80)
        print("ETL PHASE (Single Location + Daily Aggregation)")
        print("="*80)
        # ETL
        df_traffy = load_traffy_data()
        df_traffy = clean_traffy_data(df_traffy)
        df_traffy = split_coordinates(df_traffy)
        
        # Load single location weather
        df_weather = load_single_location_weather('data/raw/open-meteo-13.74N100.50.csv')
        
        # Parse timestamps to date only
        df_traffy['timestamp'] = pd.to_datetime(df_traffy['timestamp'], format='mixed', utc=True)
        df_traffy['date'] = df_traffy['timestamp'].dt.date
        df_traffy['date'] = pd.to_datetime(df_traffy['date'])
        
        df_weather['date'] = pd.to_datetime(df_weather['time']).dt.date
        df_weather['date'] = pd.to_datetime(df_weather['date'])
        
        # Aggregate weather to daily average
        df_weather_daily = df_weather.groupby('date').mean(numeric_only=True).reset_index()
        
        # Merge on date only (100% match rate)
        df_merged = df_traffy.merge(df_weather_daily, on='date', how='left')
        
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


def run_visualization():
    """
    Launch Streamlit visualization dashboard.
    """
    import subprocess
    
    print("\n" + "="*80)
    print("LAUNCHING VISUALIZATION DASHBOARD")
    print("="*80)
    print("Starting Streamlit dashboard...")
    print("Dashboard will open in your browser at http://localhost:8501")
    print("\nPress Ctrl+C in the terminal to stop the server\n")
    
    subprocess.run(['streamlit', 'run', 'visualization/visualization.py'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bangkok Traffy Complaint ML Pipeline')
    parser.add_argument('--scrape-weather', action='store_true',
                        help='Scrape weather and PM2.5/AQI data')
    parser.add_argument('--skip-etl', action='store_true', 
                        help='Skip ETL phase, load existing traffy_weather_merged.csv')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip model training phase')
    parser.add_argument('--visualize', action='store_true',
                        help='Launch Streamlit visualization dashboard after pipeline')
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample size for faster training (e.g., 200000)')
    parser.add_argument('--n-iter', type=int, default=5,
                        help='RandomizedSearchCV iterations per model (default: 5)')
    parser.add_argument('--min-samples', type=int, default=50,
                        help='Minimum positive samples to train a type (default: 50)')
    
    args = parser.parse_args()
    
    main(
        skip_scraping=not args.scrape_weather,
        skip_etl=args.skip_etl,
        skip_training=args.skip_training,
        sample_size=args.sample,
        train_n_iter=args.n_iter,
        train_min_samples=args.min_samples
    )
    
    if args.visualize:
        run_visualization()
