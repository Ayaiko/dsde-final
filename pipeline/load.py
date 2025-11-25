"""
Load module - Save processed data to files
"""
import pandas as pd
import os


def save_merged_data(df_merged, output_dir='data/processed'):
    """
    Save the merged dataset in multiple formats
    
    Parameters:
    -----------
    df_merged : pandas.DataFrame
        Merged dataset to save
    output_dir : str
        Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'traffy_weather_merged.csv')
    df_merged.to_csv(csv_path, index=False)
    print(f"  ✓ Saved CSV: {csv_path} ({len(df_merged):,} records)")


def save_aggregations(aggregations, output_dir='data/processed'):
    """
    Save all aggregated views
    
    Parameters:
    -----------
    aggregations : dict
        Dictionary of aggregated DataFrames
    output_dir : str
        Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for name, df in aggregations.items():
        csv_path = os.path.join(output_dir, f'{name}_summary.csv')
        df.to_csv(csv_path)
        print(f"  ✓ Saved {name}: {csv_path} ({len(df)} records)")


def load(df_merged, aggregations, output_dir='data/processed'):
    """
    Main load function - Save all processed data
    
    Parameters:
    -----------
    df_merged : pandas.DataFrame
        Merged dataset
    aggregations : dict
        Dictionary of aggregated DataFrames
    output_dir : str
        Output directory path
    """
    print("\n" + "=" * 60)
    print("LOAD: Saving Processed Data")
    print("=" * 60)
    
    # Save merged data
    print("Saving merged dataset...")
    save_merged_data(df_merged, output_dir)
    
    # Save aggregations
    print("Saving aggregated summaries...")
    save_aggregations(aggregations, output_dir)
    
    # Get file sizes
    merged_csv = os.path.join(output_dir, 'traffy_weather_merged.csv')
    merged_parquet = os.path.join(output_dir, 'traffy_weather_merged.parquet')
    
    csv_size = os.path.getsize(merged_csv) / (1024 * 1024)  # MB
    parquet_size = os.path.getsize(merged_parquet) / (1024 * 1024)  # MB
    
    print(f"\nLoad complete:")
    print(f"  - CSV size: {csv_size:.2f} MB")
    print(f"  - Parquet size: {parquet_size:.2f} MB")
    print(f"  - Compression: {(1 - parquet_size/csv_size)*100:.1f}% smaller")
    print(f"  - Output directory: {output_dir}")


if __name__ == "__main__":
    # Test the load module
    from extract import extract
    from transform import transform
    from aggregate import aggregate
    
    df_traffy, df_weather = extract()
    df_merged = transform(df_traffy, df_weather)
    aggregations = aggregate(df_merged)
    load(df_merged, aggregations)
    
    print("\n✓ Load module test successful")
