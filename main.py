from pipeline.extract import load_traffy_data, load_all_weather_data
from pipeline.transform import parse_timestamps, assign_grid_cells, merge_datasets
from pipeline.utils import split_coordinates, clean_traffy_data
from pipeline.load import save_merged_data


def main():
    df_traffy = load_traffy_data()
    df_traffy = clean_traffy_data(df_traffy)
    df_traffy = split_coordinates(df_traffy)
    
    df_weather = load_all_weather_data()
    
    df_traffy, df_weather = parse_timestamps(df_traffy, df_weather)
    df_traffy, df_weather = assign_grid_cells(df_traffy, df_weather, delta=0.1)
    
    df_merged = merge_datasets(df_traffy, df_weather)
    
    save_merged_data(df_merged)
    
    temp_col = 'temperature_2m (°C)' if 'temperature_2m (°C)' in df_merged.columns else 'temperature_2m'
    match_rate = (~df_merged[temp_col].isna()).sum() / len(df_merged) * 100 if temp_col in df_merged.columns else 0
    print(f"\n✓ Done. {len(df_merged):,} records, {match_rate:.1f}% matched")


if __name__ == "__main__":
    main()
