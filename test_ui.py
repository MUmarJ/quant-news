"""UI Testing script using Playwright for QuantNews dashboard."""

import asyncio
from playwright.async_api import async_playwright


async def test_dashboard():
    """Test the QuantNews dashboard UI."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})

        print("Testing QuantNews Dashboard UI...")
        print("=" * 50)

        # Navigate to the app
        await page.goto("http://127.0.0.1:8050/")
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(2)  # Wait for React to render

        # Test 1: Empty state screenshot
        print("\n1. Testing Empty State...")
        await page.screenshot(path="screenshots/01_empty_state.png", full_page=True)
        print("   ✓ Empty state screenshot captured")

        # Check for empty state messages
        price_chart = await page.locator("#price-chart").count()
        print(f"   ✓ Price chart present: {price_chart > 0}")

        # Test 2: Add a stock using quick add
        print("\n2. Testing Quick Add Button...")
        await page.click('button:has-text("AAPL")')
        await asyncio.sleep(3)  # Wait for data to load
        await page.screenshot(path="screenshots/02_after_aapl_add.png", full_page=True)
        print("   ✓ Added AAPL stock")

        # Test 3: Check if charts loaded
        print("\n3. Checking Charts Loaded...")
        await asyncio.sleep(2)
        await page.screenshot(path="screenshots/03_charts_loaded.png", full_page=True)
        print("   ✓ Charts loaded screenshot captured")

        # Test 4: Add another stock
        print("\n4. Testing Multiple Stocks...")
        await page.click('button:has-text("MSFT")')
        await asyncio.sleep(3)
        await page.screenshot(path="screenshots/04_multiple_stocks.png", full_page=True)
        print("   ✓ Added MSFT stock")

        # Test 5: Change time period
        print("\n5. Testing Period Selector...")
        await page.click('button:has-text("3M")')
        await asyncio.sleep(2)
        await page.screenshot(path="screenshots/05_period_changed.png", full_page=True)
        print("   ✓ Changed to 3M period")

        # Test 6: Toggle indicators
        print("\n6. Testing Indicator Toggles...")
        bollinger_checkbox = page.locator('label:has-text("Bollinger")')
        if await bollinger_checkbox.count() > 0:
            await bollinger_checkbox.click()
            await asyncio.sleep(2)
            await page.screenshot(path="screenshots/06_bollinger_enabled.png", full_page=True)
            print("   ✓ Bollinger Bands toggled")

        # Test 7: Remove a stock
        print("\n7. Testing Stock Removal...")
        remove_btn = page.locator('button:has-text("x")').first
        if await remove_btn.count() > 0:
            await remove_btn.click()
            await asyncio.sleep(2)
            await page.screenshot(path="screenshots/07_stock_removed.png", full_page=True)
            print("   ✓ Stock removed")

        # Test 8: Test manual input
        print("\n8. Testing Manual Stock Input...")
        await page.fill("#symbol-input", "NVDA")
        await page.click("#add-symbol-btn")
        await asyncio.sleep(3)
        await page.screenshot(path="screenshots/08_manual_input.png", full_page=True)
        print("   ✓ Manual stock input tested")

        # Test 9: Test View Data button
        print("\n9. Testing View Data Modal...")
        view_btn = page.locator('button:has-text("View Data")')
        if await view_btn.count() > 0:
            await view_btn.click()
            await asyncio.sleep(1)
            await page.screenshot(path="screenshots/09_data_modal.png", full_page=True)
            print("   ✓ Data modal tested")
            # Close modal
            close_btn = page.locator('button:has-text("Close")')
            if await close_btn.count() > 0:
                await close_btn.click()

        # Test 10: Final state
        print("\n10. Final Dashboard State...")
        await asyncio.sleep(1)
        await page.screenshot(path="screenshots/10_final_state.png", full_page=True)
        print("   ✓ Final state captured")

        await browser.close()

        print("\n" + "=" * 50)
        print("All tests completed! Screenshots saved in screenshots/ folder")
        print("=" * 50)


if __name__ == "__main__":
    import os
    os.makedirs("screenshots", exist_ok=True)
    asyncio.run(test_dashboard())
