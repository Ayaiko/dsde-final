from playwright.async_api import async_playwright
import os
import time
import asyncio

async def download_api_weather_bangkok(latitude,longitude):
    download_dir = os.getcwd()

    url=f"https://open-meteo.com/en/docs/historical-weather-api?start_date=2021-08-01&end_date=2025-01-30&hourly=temperature_2m,dew_point_2m,relative_humidity_2m,rain,vapour_pressure_deficit,cloud_cover,wind_direction_10m,surface_pressure,wind_speed_10m&timezone=GMT&latitude={latitude}&longitude={longitude}"
    
    # Get project root directory (parent of scrapers/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # download_path = os.path.join(project_root, 'data', 'weather_scraped')
    download_path = os.path.join(project_root, 'data','raw')
    os.makedirs(download_path, exist_ok=True)
    
    async with async_playwright() as p:
        # เพิ่ม stealth mode เพื่อหลีกเลี่ยงการตรวจจับบอต
        browser = await p.chromium.launch(
            headless=False,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage'
            ]
        )
        
        # ตั้งค่า User-Agent และ headers เหมือนเบราว์เซอร์จริง
        context = await browser.new_context(
            accept_downloads=True,
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        
        page = await context.new_page()
        
        # ซ่อน playwright
        page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => false})")
        
        await page.goto(url, timeout=60000, wait_until="networkidle")

        # เพิ่มการหน่วงเวลาเพื่อให้ดูเหมือนคนจริง
        time.sleep(2)

        async with page.expect_download() as dl_info:
            await page.get_by_role("link", name="Download CSV").click()

        download = await dl_info.value

        # เอาชื่อไฟล์ที่เว็บตั้งมาให้
        filename = download.suggested_filename
        save_path = os.path.join(download_path, filename)

        await download.save_as(save_path)

        print("Downloaded to:", save_path)

        await browser.close()

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


asyncio.run(download_api_weather_bangkok("13.75","100.5"))