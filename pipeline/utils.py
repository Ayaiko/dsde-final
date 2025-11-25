"""
Utility functions for coordinate operations
"""
import pandas as pd


def split_coordinates(df, coord_column='coords'):
    """
    Split coordinates column into separate latitude and longitude columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing coordinates
    coord_column : str
        Name of the column containing coordinates in format 'lon,lat'
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added 'longitude' and 'latitude' columns
    """
    df[['longitude', 'latitude']] = df[coord_column].str.split(',', expand=True).astype(float)
    return df


def get_grid_coordinates(df, coord_column='coords', delta=0.1):
    """
    Extract unique grid coordinates from complaint data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with coordinates (either split or in 'lon,lat' format)
    coord_column : str
        Column name containing coordinates in 'lon,lat' format (if not already split)
    delta : float
        Grid size in degrees (default: 0.1 ≈ 11km)
    
    Returns:
    --------
    list of tuples
        List of (longitude, latitude) grid center coordinates
    """
    df_temp = df.copy()
    
    # Split coordinates if not already done
    if 'longitude' not in df_temp.columns or 'latitude' not in df_temp.columns:
        df_temp[['longitude', 'latitude']] = df_temp[coord_column].str.split(',', expand=True).astype(float)
    
    # Create grid bins
    df_temp['lon_bin'] = ((df_temp['longitude'] // delta) * delta).round(5)
    df_temp['lat_bin'] = ((df_temp['latitude'] // delta) * delta).round(5)
    
    # Get unique grid coordinates
    grid_df = df_temp[['lon_bin', 'lat_bin']].drop_duplicates()
    coords_list = list(grid_df.itertuples(index=False, name=None))
    
    return coords_list


def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate approximate distance between two coordinates
    
    Parameters:
    -----------
    lat1, lon1 : float
        First point coordinates
    lat2, lon2 : float
        Second point coordinates
    
    Returns:
    --------
    float
        Distance in kilometers (Euclidean approximation)
    """
    # Simple Euclidean approximation (1 degree ≈ 111 km)
    distance_km = (((lat1 - lat2)**2 + (lon1 - lon2)**2)**0.5) * 111
    return distance_km


def clean_traffy_data(df, drop_columns=None, filter_bangkok=True):
    """
    Clean Bangkok Traffy data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw Traffy data
    drop_columns : list, optional
        Columns to drop (default: ticket_id, organization, photo, etc.)
    filter_bangkok : bool
        Filter for Bangkok only (default: True)
    
    Returns:
    --------
    pandas.DataFrame
        Cleaned Traffy data
    """
    if drop_columns is None:
        drop_columns = ['ticket_id', 'organization', 'photo', 'photo_after',
                        'address', 'state', 'star', 'count_reopen', 'last_activity']
    
    df_clean = df.drop(drop_columns, axis=1, errors='ignore')
    df_clean = df_clean.dropna()
    
    if filter_bangkok:
        df_clean = df_clean[df_clean['province'] == 'กรุงเทพมหานคร'].reset_index(drop=True)
    
    return df_clean


def clean_pipeline(df_traffy_path='data/raw/bangkok_traffy.csv', save_path='data/processed/traffy_clean.csv'):
    """
    Wrapper: Load, clean, and save Traffy data
    
    Parameters:
    -----------
    df_traffy_path : str
        Path to raw Traffy CSV
    save_path : str
        Path to save cleaned CSV
    
    Returns:
    --------
    pandas.DataFrame
        Cleaned Traffy data
    """
    import os
    
    df = pd.read_csv(df_traffy_path)
    df_clean = clean_traffy_data(df)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_clean.to_csv(save_path, index=False)
    
    print(f"Cleaned: {len(df):,} → {len(df_clean):,} records")
    print(f"Saved: {save_path}")
    
    return df_clean
