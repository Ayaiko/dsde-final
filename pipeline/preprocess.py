import pandas as pd

def parse_type_column(df):
    """
    แปลง column 'type' จาก string representation ของ set "{น้ำท่วม,ถนน}"
    ให้เป็น list ['น้ำท่วม','ถนน']
    """
    def parse_cell(cell):
        # ถ้าเป็น string
        if isinstance(cell, str):
            # ลบ { } รอบ ๆ
            s = cell.strip('{}').strip()
            # ถ้าไม่ว่าง ให้ split ด้วย comma
            if s:
                # แยกด้วย ',' แล้ว strip space
                return [x.strip() for x in s.split(',')]
            else:
                return []  # empty set -> empty list
        # ถ้าเป็น list อยู่แล้ว หรือ NaN
        elif isinstance(cell, list):
            return cell
        else:
            return []

    # แปลง column type
    df['type'] = df['type'].apply(parse_cell)
    return df

def filter_empty_types(df, column='type'):
    """
    Remove rows where 'type' column is empty list.
    """
    return df[df[column].map(lambda x: len(x) > 0)].reset_index(drop=True)

def drop_missing_weather(df, weather_columns=None):
    """
    Drop rows with null values in weather columns.
    Default weather columns: ['temperature_2m (°C)', 'rain (mm)', 'wind_speed_10m (km/h)']
    """
    if weather_columns is None:
        weather_columns = ['temperature_2m (°C)', 'rain (mm)', 'wind_speed_10m (km/h)']
    return df.dropna(subset=weather_columns).reset_index(drop=True)


def create_binary_targets(df, type_column='type'):
    """
    Create binary target columns for each complaint type.
    Converts parsed type list into individual binary columns (type_ถนน, type_ทางเท้า, etc.)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with parsed 'type' column (list of types)
    type_column : str
        Name of column containing type lists
    
    Returns:
    --------
    tuple
        (df with new binary columns, list of binary column names)
    
    Example:
    --------
    Input: type = ['ถนน', 'ทางเท้า']
    Output: type_ถนน = 1, type_ทางเท้า = 1, type_น้ำท่วม = 0, ...
    """
    df = df.copy()
    
    # Collect all unique types across all rows
    all_types = set()
    for type_list in df[type_column]:
        if isinstance(type_list, list):
            all_types.update(type_list)
    
    print(f"  Found {len(all_types)} unique complaint types")
    
    # Create binary column for each type
    binary_columns = []
    for complaint_type in all_types:
        col_name = f'type_{complaint_type}'
        df[col_name] = df[type_column].apply(
            lambda x: 1 if (isinstance(x, list) and complaint_type in x) else 0
        )
        binary_columns.append(col_name)
    
    print(f"  Created {len(binary_columns)} binary target columns")
    
    return df, binary_columns


def sample_data(df, n=200000, random_state=42):
    """
    Randomly sample n records from dataframe for faster training.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Full dataset
    n : int
        Number of samples to extract (default: 200,000)
    random_state : int
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    pandas.DataFrame
        Sampled dataframe with reset index
    """
    print(f"  Original: {len(df):,} records")
    
    if len(df) <= n:
        print(f"  ⚠ Dataset smaller than {n:,}, returning full dataset")
        return df.reset_index(drop=True)
    
    df_sampled = df.sample(n=n, random_state=random_state).reset_index(drop=True)
    print(f"  Sampled: {len(df_sampled):,} records ({len(df_sampled)/len(df)*100:.1f}%)")
    
    return df_sampled