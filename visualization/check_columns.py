import pandas as pd

df = pd.read_csv('../data/processed/traffy_weather_merged.csv')
print("Original columns:")
print(df.columns.tolist()[:15])

df.columns = df.columns.str.strip()
print("\nAfter stripping:")
print(df.columns.tolist()[:15])

# Check for pm columns specifically
pm_cols = [col for col in df.columns if 'pm' in col.lower()]
print("\nPM columns:")
print(pm_cols)
