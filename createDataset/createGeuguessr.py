import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
import pyautogui
import requests
import traceback
import os
import time
from PIL import ImageGrab, Image
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs

load_dotenv()

options = webdriver.ChromeOptions()
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920x1080")
options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

service = Service("chromedriver")
browser = webdriver.Chrome(service=service, options=options)

known_coords = {}

MAP_X, MAP_Y = 1779, -183
GUESS_X, GUESS_Y = 1789, -40
CENTER_X, CENTER_Y = 801, -634


def load_cookies(browser, cookie_file, url):
    if not os.path.exists(cookie_file):
        print("Cookie file not found.")
        return

    browser.get(url)  # Open the site before adding cookies

    with open(cookie_file, "r") as file:
        for line in file:
            if line.startswith("#") or not line.strip():
                continue  # Skip comments and empty lines

            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue  # Skip malformed lines

            domain, flag, path, secure, expiry, name, value = parts

            cookie = {
                "domain": domain,
                "path": path,
                "secure": secure.lower() == "true",
                "name": name,
                "value": value
            }

            if expiry.isdigit():
                cookie["expiry"] = int(expiry)

            browser.add_cookie(cookie)

    browser.refresh()


def get_coords(logs):
    for entry in reversed(logs):  # Start from the most recent log
        if "message" in entry:
            try:
                log_message = json.loads(entry["message"])

                # Check if 'params' and 'request' are in the message
                if (
                    "params" in log_message["message"]
                    and "request" in log_message["message"]["params"]
                ):
                    url = log_message["message"]["params"]["request"]["url"]

                    # Check if the URL matches the Street View URL
                    if "streetviewpixels-pa.googleapis.com" in url:
                        parsed_url = urlparse(url)
                        params = parse_qs(parsed_url.query)
                        panoidID = params.get("panoid", [None])[0]

                        if panoidID:
                            print(panoidID)
                            break
            except (json.JSONDecodeError, KeyError):
                continue

    if not panoidID:
        raise Exception("Could not get panoidID")

    coords = known_coords.get(panoidID)
    if coords:
        return coords

    api_key = os.getenv("GOOGLE_API_KEY")
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"pano": panoidID, "key": api_key}

    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            raise Exception("Got status code", response.status_code)

        data = response.json()
        print(data)

        location = data["location"]
        known_coords[panoidID] = (location["lat"], location["lng"], panoidID)

        return location["lat"], location["lng"], panoidID

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        raise e


def take_screenshot(directory, panoidID):
    i = 1
    while os.path.exists(os.path.join(directory, f"{i}.png")):
        i += 1

    screenshot = ImageGrab.grab(bbox=(0, -1000, 1600, -100))

    screenshot_path = os.path.join(directory, f"{panoidID}.png")
    screenshot.save(screenshot_path)

    print(f"Screenshot saved to {screenshot_path}")

    return i, screenshot_path


def simulate_mouse_click(x, y):
    try:
        pyautogui.moveTo(x, y, duration=0.1)
        pyautogui.click()
    except pyautogui.FailSafeError:
        print("Mouse moved to a fail-safe corner. Click cancelled.")
    except Exception as e:
        print(f"An error occurred: {e}")


def next_round(counter):
    simulate_mouse_click(MAP_X, MAP_Y)  # open map
    simulate_mouse_click(MAP_X, MAP_Y)  # Click map
    pyautogui.press("space")  # Make guess
    pyautogui.press("space")  # Next round
    time.sleep(0.05)
    pyautogui.press("space")
    time.sleep(0.05)
    pyautogui.press("space")
    # Sometimes GeoGuessr hangs at a screen, refresh once in a while to
    # prevent this
    if counter % 20 == 0:
        pyautogui.hotkey("command", "r")
        time.sleep(6)
        pyautogui.press("space")
    simulate_mouse_click(CENTER_X, CENTER_Y)
    time.sleep(2)  # Wait for next round


if __name__ == "__main__":
    num_images = 10000
    geo_guessr_url = "https://www.geoguessr.com/"
    dataset_dir = "geoGuessrDataset/"

    load_cookies(browser, "cookies.txt", geo_guessr_url)
    try:
        start = input("Press any key to start scraping")
        for counter in range(num_images):
            logs = browser.get_log("performance")
            try:
                lat, lon, panoidID = get_coords(logs)
            except Exception as e:
                print(e)
                print()
                time.sleep(5)
                next_round(counter)
                continue

            num_existing_screens, screenshot_path = take_screenshot(dataset_dir, panoidID)

            with open(f"{dataset_dir}/anUrbanWorld.csv", "a+") as f:
                f.write(f"{panoidID},{lat},{lon}\n")

            next_round(counter)
    finally:
        browser.quit()
