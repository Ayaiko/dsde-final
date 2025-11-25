import os
from playwright.async_api import async_playwright
import asyncio

async def getInfo(latitude, longitude):
    url = f"https://open-meteo.com/en/docs/historical-weather-api?start_date=2021-08-01&end_date=2025-01-30&hourly=temperature_2m,dew_point_2m,relative_humidity_2m,rain,vapour_pressure_deficit,cloud_cover,wind_direction_10m,surface_pressure,wind_speed_10m&timezone=GMT&latitude={latitude}&longitude={longitude}"

    # Get project root directory (parent of scrapers/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    download_path = os.path.join(project_root, 'data', 'weather_scraped')
    os.makedirs(download_path, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)

        # ไม่ต้องใช้ downloads_path
        context = await browser.new_context(accept_downloads=True)

        page = await context.new_page()
        await page.goto(url)

        async with page.expect_download() as dl_info:
            await page.get_by_role("link", name="Download CSV").click()

        download = await dl_info.value

        # เอาชื่อไฟล์ที่เว็บตั้งมาให้
        filename = download.suggested_filename
        save_path = os.path.join(download_path, filename)

        # บังคับให้ดาวน์โหลดมาเก็บ path ปัจจุบัน
        await download.save_as(save_path)

        print("Downloaded to:", save_path)

        await browser.close()

        # -------------------------------------------------
        # 🧹 ลบ 3 บรรทัดแรกออกจากไฟล์ CSV
        # -------------------------------------------------
        with open(save_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # ข้าม 3 บรรทัดบน
        new_lines = lines[3:]

        # เขียนทับกลับ
        with open(save_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        print("Cleaned CSV (removed first 3 lines).")

async def scrape_multiple_locations(coords_list):
    """
    Scrape weather data for multiple grid coordinates
    
    Parameters:
    -----------
    coords_list : list of tuples
        List of (longitude, latitude) coordinates
    """
    total = len(coords_list)
    print(f"Starting to scrape weather data for {total} locations...")
    
    for i, (longitude, latitude) in enumerate(coords_list, 1):
        print(f"\n[{i}/{total}] Scraping: ({latitude:.5f}, {longitude:.5f})")
        await getInfo(latitude, longitude)
    
    print(f"\n✓ Completed scraping {total} locations!")


async def scrape_from_dataframe(df, coord_column='coords', delta=0.1):
    """
    Convenience function: Extract grid coordinates from DataFrame and scrape weather data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with coordinates
    coord_column : str
        Column name containing coordinates in 'lon,lat' format
    delta : float
        Grid size in degrees (default: 0.1 ≈ 11km)
    
    Returns:
    --------
    list of tuples
        The coordinates that were scraped
    """
    import pandas as pd
    
    # Split coordinates
    df_temp = df.copy()
    df_temp[['longitude', 'latitude']] = df_temp[coord_column].str.split(',', expand=True).astype(float)
    
    # Create grid
    df_temp['lon_bin'] = ((df_temp['longitude'] // delta) * delta).round(5)
    df_temp['lat_bin'] = ((df_temp['latitude'] // delta) * delta).round(5)
    
    # Get unique coordinates
    grid_df = df_temp[['lon_bin', 'lat_bin']].drop_duplicates()
    coords_list = list(grid_df.itertuples(index=False, name=None))
    
    print(f"Extracted {len(coords_list)} unique grid locations")
    
    # Scrape them
    await scrape_multiple_locations(coords_list)
    
    return coords_list
