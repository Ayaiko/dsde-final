from playwright.sync_api import sync_playwright
import os
import time

def download_aqi_bangkok():
    # ใช้ path ปัจจุบันของสคริปต์
    download_dir = os.getcwd()

    with sync_playwright() as p:
        # เพิ่ม stealth mode เพื่อหลีกเลี่ยงการตรวจจับบอต
        browser = p.chromium.launch(
            headless=False,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage'
            ]
        )
        
        # ตั้งค่า User-Agent และ headers เหมือนเบราว์เซอร์จริง
        context = browser.new_context(
            accept_downloads=True,
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        
        page = context.new_page()
        
        # ซ่อน playwright
        page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => false})")
        
        page.goto("https://aqicn.org/city/bangkok/", timeout=60000, wait_until="networkidle")

        # เพิ่มการหน่วงเวลาเพื่อให้ดูเหมือนคนจริง
        time.sleep(2)

        # Scroll down ช้า ๆ เพื่อให้เหมือนคนจริง
        print("Scrolling down slowly...")
        for i in range(10):
            page.evaluate("window.scrollBy(0, 300)")
            time.sleep(0.5)
        print("✓ Scrolled down")
        
        time.sleep(1)

        # Try multiple selectors to find the download button
        try:
            button = page.wait_for_selector('text=Download this data (CSV format)', timeout=15000)
        except:
            print("First selector not found, trying alternative selectors...")
            # Try alternative selectors
            try:
                button = page.locator("button:has-text('Download this data')").first
                button.wait_for(timeout=5000)
            except:
                # Print all buttons on the page for debugging
                buttons = page.locator("button, a[href*='download']").all()
                print(f"Found {len(buttons)} potential download elements")
                for i, btn in enumerate(buttons):
                    try:
                        text = btn.inner_text()
                        print(f"Button {i}: {text}")
                    except:
                        pass
                raise Exception("Could not find download button. Check the printed buttons above.")

        # Click download button
        print("Clicking download button...")
        time.sleep(0.5)
        button.click()
        
        # Wait for dialog/modal to appear
        print("Waiting for dialog to appear...")
        try:
            page.wait_for_selector('div[role="dialog"], .modal, [class*="dialog"], [class*="modal"]', timeout=15000)
            print("✓ Dialog appeared")
        except:
            print("No dialog selector found, waiting anyway...")
        
        time.sleep(3)  # Wait for dialog to fully load
        
        # Scroll down inside dialog to see "I agree" button
        print("Scrolling down to find 'I agree' button...")
        for i in range(8):
            page.evaluate("document.querySelector('div[role=\"dialog\"], .modal, [class*=\"dialog\"]')?.scrollBy(0, 300) || window.scrollBy(0, 300)")
            time.sleep(0.5)
        print("✓ Scrolled down")
        time.sleep(1)
        
        # Look for "I agree" button and click it
        try:
            print("Looking for 'I agree' button...")
            # Try multiple selectors
            agree_button = None
            try:
                agree_button = page.wait_for_selector('button:has-text("I agree"), button:has-text("I Agree")', timeout=5000)
            except:
                # Try the div with class histui ui large primary button
                try:
                    agree_button = page.wait_for_selector('div.histui.ui.large.primary.button', timeout=5000)
                except:
                    # Try more general selector
                    agree_button = page.wait_for_selector('.histui.primary, div[class*="primary"][class*="button"]', timeout=5000)
            
            if agree_button:
                print("Found 'I agree' button, clicking...")
                time.sleep(0.5)
                agree_button.click()
                print("✓ Clicked 'I agree'")

            time.sleep(2)
        except:
            print("No 'I agree' button found, continuing...")

        with page.expect_download() as download_info:
            # If there's another button to click after agreeing
            try:
                final_button = page.wait_for_selector('button:has-text("Download")', timeout=5000)
                time.sleep(0.5)
                final_button.click()
            except:
                pass
        
        download = download_info.value

        # บันทึกไฟล์ลงโฟลเดอร์ปัจจุบัน
        saved_path = os.path.join(download_dir, download.suggested_filename)
        download.save_as(saved_path)

        print("✓ ดาวน์โหลดเสร็จ:", saved_path)

        browser.close()


download_aqi_bangkok()
