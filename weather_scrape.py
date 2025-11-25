import os
from playwright.sync_api import sync_playwright

def getInfo(latitude, longitude):
    url = f"https://open-meteo.com/en/docs/historical-weather-api?start_date=2021-08-01&end_date=2025-01-30&hourly=temperature_2m,dew_point_2m,relative_humidity_2m,rain,vapour_pressure_deficit,cloud_cover,wind_direction_10m,surface_pressure,wind_speed_10m&timezone=GMT&latitude={latitude}&longitude={longitude}"

    download_path = os.path.join(os.getcwd(),'data')  # โฟลเดอร์ปัจจุบัน

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)

        # ไม่ต้องใช้ downloads_path
        context = browser.new_context(accept_downloads=True)

        page = context.new_page()
        page.goto(url)

        with page.expect_download() as dl_info:
            page.get_by_role("link", name="Download CSV").click()

        download = dl_info.value

        # เอาชื่อไฟล์ที่เว็บตั้งมาให้
        filename = download.suggested_filename
        save_path = os.path.join(download_path, filename)

        # บังคับให้ดาวน์โหลดมาเก็บ path ปัจจุบัน
        download.save_as(save_path)

        print("Downloaded to:", save_path)

        browser.close()

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

getInfo(13.81, 100.51)

# for latitude,longitude in new:
#     getInfo(latitude, longitude)