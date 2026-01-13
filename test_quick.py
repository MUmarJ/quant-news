"""Quick UI test to verify fixes."""

import asyncio
from playwright.async_api import async_playwright


async def test_quick():
    """Quick test to verify charts load without errors."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})

        print("Testing QuantNews Dashboard...")

        # Navigate to the app
        await page.goto("http://127.0.0.1:8050/")
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(2)

        # Take empty state screenshot
        await page.screenshot(path="screenshots/test_empty.png", full_page=True)
        print("1. Empty state captured")

        # Click AAPL quick add
        await page.click('button:has-text("AAPL")')
        await asyncio.sleep(5)  # Wait for data to load

        # Take screenshot with data
        await page.screenshot(path="screenshots/test_with_data.png", full_page=True)
        print("2. Data loaded state captured")

        # Check for error overlay
        errors = await page.locator('.dash-fe-error-item').count()
        if errors > 0:
            print(f"   WARNING: {errors} error(s) detected")
        else:
            print("   No errors detected!")

        await browser.close()
        print("\nTest completed!")


if __name__ == "__main__":
    import os
    os.makedirs("screenshots", exist_ok=True)
    asyncio.run(test_quick())
