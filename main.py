from pipeline.extract import load_traffy_data, load_all_weather_data
from pipeline.transform import parse_timestamps, assign_grid_cells, merge_datasets
from pipeline.utils import split_coordinates, clean_traffy_data
from pipeline.load import save_merged_data
from pipeline.preprocess import parse_type_column, filter_empty_types, drop_missing_weather
from pipeline.features import prepare_features
import pandas as pd


def main():
    # ETL
    df_traffy = load_traffy_data()
    df_traffy = clean_traffy_data(df_traffy)
    df_traffy = split_coordinates(df_traffy)
    
    df_weather = load_all_weather_data()
    
    df_traffy, df_weather = parse_timestamps(df_traffy, df_weather)
    df_traffy, df_weather = assign_grid_cells(df_traffy, df_weather, delta=0.1)
    
    df_merged = merge_datasets(df_traffy, df_weather)
    save_merged_data(df_merged)
    
    # Preprocessing
    df_merged = parse_type_column(df_merged)
    df_merged = filter_empty_types(df_merged)
    df_merged = drop_missing_weather(df_merged)
    
    # Feature Engineering
    df_merged = prepare_features(df_merged)
    
    # Save final dataset
    output_path = 'data/processed/traffy_weather_final.csv'
    df_merged.to_csv(output_path, index=False)
    
    temp_col = 'temperature_2m (°C)' if 'temperature_2m (°C)' in df_merged.columns else 'temperature_2m'
    match_rate = (~df_merged[temp_col].isna()).sum() / len(df_merged) * 100 if temp_col in df_merged.columns else 0
    
    print(f"\n✓ Pipeline complete: {len(df_merged):,} records, {len(df_merged.columns)} features, {match_rate:.1f}% matched")
    print(f"✓ Saved to: {output_path}")


if __name__ == "__main__":
    main()
