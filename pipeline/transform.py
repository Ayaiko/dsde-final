"""
Transform module - Clean, parse, and merge data
"""
import pandas as pd
import numpy as np
from .utils import split_coordinates, calculate_distance


def parse_timestamps(df_traffy, df_weather):
    """
    Parse and standardize timestamps for both datasets
    
    Parameters:
    -----------
    df_traffy : pandas.DataFrame
        Traffy data with 'timestamp' column
    df_weather : pandas.DataFrame
        Weather data with 'time' column
    
    Returns:
    --------
    tuple
        (df_traffy, df_weather) with parsed timestamps
    """
    # Parse Traffy timestamps (mixed format, UTC-aware)
    df_traffy['timestamp'] = pd.to_datetime(df_traffy['timestamp'], format='mixed', utc=True)
    df_traffy['timestamp_hour'] = df_traffy['timestamp'].dt.floor('h')
    
    # Parse weather timestamps (make UTC-aware)
    df_weather['time'] = pd.to_datetime(df_weather['time']).dt.tz_localize('UTC')
    
    return df_traffy, df_weather


def calculate_distances(df, weather_lat=13.74, weather_lon=100.50):
    """
    Calculate distance from each point to weather station
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'latitude' and 'longitude' columns
    weather_lat : float
        Weather station latitude
    weather_lon : float
        Weather station longitude
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with 'distance_to_weather_station_km' column
    """
    df['distance_to_weather_station_km'] = df.apply(
        lambda row: calculate_distance(row['latitude'], row['longitude'], weather_lat, weather_lon),
        axis=1
    )
    
    return df


def assign_grid_cells(df_traffy, df_weather, delta=0.1):
    df_traffy['grid_lat'] = ((df_traffy['latitude'] / delta).round() * delta).round(1)
    df_traffy['grid_lon'] = ((df_traffy['longitude'] / delta).round() * delta).round(1)
    
    df_weather['grid_lat'] = ((df_weather['latitude'] / delta).round() * delta).round(1)
    df_weather['grid_lon'] = ((df_weather['longitude'] / delta).round() * delta).round(1)
    
    return df_traffy, df_weather


def merge_datasets(df_traffy, df_weather):
    """
    Merge Traffy and weather data on hourly timestamps
    
    Parameters:
    -----------
    df_traffy : pandas.DataFrame
        Traffy data with 'timestamp_hour'
    df_weather : pandas.DataFrame
        Weather data with 'time'
    
    Returns:
    --------
    pandas.DataFrame
        Merged dataset
    """
    df_merged = df_traffy.merge(
        df_weather,
        left_on=['grid_lat', 'grid_lon', 'timestamp_hour'],
        right_on=['grid_lat', 'grid_lon', 'time'],
        how='left',
        suffixes=('', '_weather')
    )
    
    return df_merged


def transform(df_traffy, df_weather):
    """
    Main transform function - Apply all transformations
    
    Parameters:
    -----------
    df_traffy : pandas.DataFrame
        Raw Traffy data
    df_weather : pandas.DataFrame
        Raw weather data
    
    Returns:
    --------
    pandas.DataFrame
        Transformed and merged dataset
    """
    print("\n" + "=" * 60)
    print("TRANSFORM: Processing Data")
    print("=" * 60)
    
    # Split coordinates
    print("Splitting coordinates...")
    df_traffy = split_coordinates(df_traffy)
    
    # Parse timestamps
    print("Parsing timestamps...")
    df_traffy, df_weather = parse_timestamps(df_traffy, df_weather)
    
    # Calculate distances
    print("Calculating distances to weather station...")
    df_traffy = calculate_distances(df_traffy)
    
    # Merge datasets
    print("Merging datasets...")
    df_merged = merge_datasets(df_traffy, df_weather)
    
    # Statistics
    missing_weather = df_merged['temperature_2m (°C)'].isna().sum() if 'temperature_2m (°C)' in df_merged.columns else 0
    
    print(f"\nTransform complete:")
    print(f"  - Final shape: {df_merged.shape}")
    print(f"  - Missing weather data: {missing_weather:,} ({missing_weather/len(df_merged)*100:.2f}%)")
    print(f"  - Avg distance to station: {df_merged['distance_to_weather_station_km'].mean():.2f} km")
    
    return df_merged


if __name__ == "__main__":
    # Test the transform module
    from extract import extract
    
    df_traffy, df_weather = extract()
    df_merged = transform(df_traffy, df_weather)
    
    print("\n✓ Transform module test successful")
    print(f"Sample data:\n{df_merged.head()}")
