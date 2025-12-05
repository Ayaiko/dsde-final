"""
Feature Engineering Module
Extract and encode features for modeling
"""
import pandas as pd
import numpy as np


def extract_time_features(df, timestamp_col='timestamp'):
    df = df.copy()

    # แปลง column timestamp ให้เป็น datetime (utc)
    df['timestamp_col'] = pd.to_datetime(df[timestamp_col], format='mixed',utc=True)

    # สกัด feature เวลา
    df['hour'] = df['timestamp_col'].dt.hour
    df['day_of_week'] = df['timestamp_col'].dt.dayofweek  # 0 = Monday, 6 = Sunday
    df['month'] = df['timestamp_col'].dt.month
    print('hello')

    return df



def encode_cyclical_features(df):
    """
    Encode cyclical time features using sin/cos transformation
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with time features (hour, day_of_week, month)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with sin/cos encoded features
    """
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

#fix as 0 1
def encode_districts(df, district_col='district'):
    """
    One-hot encode district/location features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with district column
    district_col : str
        Name of district column
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with one-hot encoded districts
    """
    if district_col in df.columns:
        district_encoded = pd.get_dummies(df[district_col], prefix='district')
        df = pd.concat([df, district_encoded], axis=1)
        print(f"  ✓ Encoded {len(district_encoded.columns)} districts")
    else:
        print(f"  ⚠ Column '{district_col}' not found, skipping district encoding")
    
    return df


def prepare_features(df, timestamp_col='timestamp', district_col='district'):
    """
    Main function: apply all feature engineering
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw or preprocessed DataFrame
    timestamp_col : str
        Name of timestamp column
    district_col : str
        Name of district column
    
    Returns:
    --------
    pandas.DataFrame
        Feature-engineered DataFrame ready for modeling
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    print("\nExtracting time features...")
    df = extract_time_features(df, timestamp_col)
    
    print("Encoding cyclical features...")
    df = encode_cyclical_features(df)
    
    print("Encoding districts...")
    df = encode_districts(df, district_col)
    
    print(f"\n✓ Feature engineering complete")
    print(f"  Final shape: {df.shape}")
    print(f"  New features: hour, day_of_week, month, season, cyclical encodings, districts")
    
    return df


if __name__ == "__main__":
    print("Testing feature engineering module...")
    
    df = pd.read_csv('data/processed/traffy_weather_merged.csv')
    print(f"Loaded {len(df):,} records")
    
    df = prepare_features(df)
    
    print("\nSample features:")
    feature_cols = ['timestamp', 'hour', 'day_of_week', 'season', 'is_weekend', 
                    'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    print(df[feature_cols].head(10))
    
    print("\n✓ Feature engineering test successful")
