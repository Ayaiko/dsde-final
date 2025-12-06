"""
Weather Data Scraping Workflow
1. Load Bangkok Traffy data
2. Extract coordinates
3. Scrape weather data for each grid location
4. Clean and save results
"""
import asyncio
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.utils import get_grid_coordinates
from scrapers.weather_scrape import scrape_multiple_locations


async def main():
    # Load data
    # df = pd.read_csv('data/raw/bangkok_traffy.csv')
    print(f"Loaded {len(df):,} records")
    
    # Extract coordinates
    coords = get_grid_coordinates(df, delta=0.1)
    print(f"Found {len(coords)} grid locations")
    
    # Prompt
    response = input(f"Scrape how many? (all/{len(coords)}, or number): ")
    
    if response.lower() == 'all':
        coords_to_scrape = coords
    else:
        try:
            n = int(response)
            coords_to_scrape = coords[:n]
        except:
            print("Cancelled")
            return
    
    # Scrape
    await scrape_multiple_locations(coords_to_scrape)
    
    # Summary
    scraped = len([f for f in os.listdir('data/weather_scraped') if f.endswith('.csv')])
    print(f"\n✓ Done. {scraped} weather files in data/weather_scraped/")


if __name__ == "__main__":
    asyncio.run(main())
