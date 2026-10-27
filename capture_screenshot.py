"""Capture genuine screenshots of the running Streamlit application.

Connects to http://localhost:8501 via Playwright headless Chromium,
waits for StyleGAN2 model inference and Matplotlib plots to finish rendering,
and captures high-resolution screenshots into screenshots/.
"""

import os
import time
from playwright.sync_api import sync_playwright

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SCREENSHOTS_DIR = os.path.join(PROJECT_ROOT, "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

def capture():
    print("Launching headless Chromium via Playwright...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Standard desktop viewport
        context = browser.new_context(viewport={"width": 1440, "height": 1600})
        page = context.new_page()

        print("Navigating to Streamlit app at http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=60000)

        # Wait for Streamlit to render image elements and metrics
        print("Waiting for images and ML diagnostics to render...")
        page.wait_for_selector("img", timeout=60000)
        time.sleep(5)  # Allow Matplotlib figure and Streamlit layout to settle

        # 1. Full-page capture showing header, controls, comparison, diagnostics, and sequence gallery
        full_path = os.path.join(SCREENSHOTS_DIR, "stylegan2_explorer_full.png")
        page.screenshot(path=full_path, full_page=True)
        print(f"Successfully saved full page screenshot to: {full_path}")

        # 2. Main interface capture showing Seed A, Interpolated Face, Seed B, Slider, and Distance Metrics
        main_path = os.path.join(SCREENSHOTS_DIR, "stylegan2_explorer_main.png")
        page.screenshot(path=main_path, full_page=False)
        print(f"Successfully saved main overview screenshot to: {main_path}")

        browser.close()

if __name__ == "__main__":
    capture()
