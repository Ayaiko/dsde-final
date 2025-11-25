"""
Extract module - Load data from CSV files
"""
import pandas as pd
import os


def load_traffy_data(filepath='data/raw/bangkok_traffy.csv'):
    """
    Load Bangkok Traffy complaint data
    
    Parameters:
    -----------
    filepath : str
        Path to the Traffy CSV file
    
    Returns:
    --------
    pandas.DataFrame
        Raw Traffy data
    """
    print(f"Loading Traffy data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df):,} records")
    return df


def load_weather_data(filepath='data/raw/open-meteo-13.74N100.50E7m.csv'):
    """
    Load weather data from Open-Meteo
    
    Parameters:
    -----------
    filepath : str
        Path to the weather CSV file
    
    Returns:
    --------
    pandas.DataFrame
        Raw weather data
    """
    print(f"Loading weather data from: {filepath}")
    
    # Check if file needs skiprows (has metadata)
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        # If first line starts with 'Latitude' or similar, skip metadata rows
        skiprows = 2 if 'Latitude' in first_line or 'latitude' in first_line else 0
    
    df = pd.read_csv(filepath, skiprows=skiprows)
    print(f"✓ Loaded {len(df):,} records")
    return df


def load_all_weather_data(weather_dir='data/weather_scraped'):
    import glob
    
    pattern = os.path.join(weather_dir, '*.csv')
    weather_files = glob.glob(pattern)
    
    print(f"Found {len(weather_files)} weather files")
    
    dfs = []
    for file in weather_files:
        df = pd.read_csv(file)
        
        basename = os.path.basename(file).replace('open-meteo-', '').replace('.csv', '')
        parts = basename.split('N')
        df['latitude'] = float(parts[0])
        lon_str = parts[1].split('E')[0]
        df['longitude'] = float(lon_str)
        
        dfs.append(df)
    
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=['time', 'latitude', 'longitude'])
    
    print(f"✓ Loaded {len(df_all):,} weather records from {len(weather_files)} files")
    return df_all


def extract():
    """
    Main extract function - Load all required data
    
    Returns:
    --------
    tuple
        (df_traffy, df_weather)
    """
    print("\n" + "=" * 60)
    print("EXTRACT: Loading Data")
    print("=" * 60)
    
    df_traffy = load_traffy_data()
    df_weather = load_weather_data()
    
    print(f"\nExtract complete:")
    print(f"  - Traffy: {df_traffy.shape}")
    print(f"  - Weather: {df_weather.shape}")
    
    return df_traffy, df_weather


if __name__ == "__main__":
    # Test the extract module
    df_traffy, df_weather = extract()
    print("\n✓ Extract module test successful")
