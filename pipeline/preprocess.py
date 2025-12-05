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