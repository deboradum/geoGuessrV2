import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import random
import string
import requests
import os
import time
from PIL import Image
from io import BytesIO
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

def random_string(length=10):
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def get_image_urls(logs):
    """Extracts image URLs from browser logs."""
    image_urls = set()
    image_extensions = [".jpg", ".jpeg"]

    for entry in logs:
        try:
            log_entry = json.loads(entry["message"])["message"]
            if log_entry["method"] == "Network.responseReceived":
                url = log_entry["params"]["response"]["url"]
                mime_type = log_entry["params"]["response"].get("mimeType", "")

                # Check if it's an image based on extension or MIME type
                is_image = any(
                    url.lower().endswith(ext) for ext in image_extensions
                ) or mime_type.startswith("image/")

                if is_image:
                    image_urls.add(url)
        except (KeyError, json.JSONDecodeError):
            pass

    return image_urls


def download_images(urls, location_key, directory="images"):
    for i, url in enumerate(urls):
        print(f"Downloading {i+1}/{len(urls)}")
        try:
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                continue
            try:
                img = Image.open(BytesIO(response.content))
                width, height = img.size
                if width != 512 and height != 512:
                    continue

                filename = f"{location_key}_{i}.jpg"
                with open(os.path.join(directory, filename), "wb") as file:
                    file.write(response.content)
            except (IOError, OSError) as e:
                print(f"Unable to determine image dimensions: {e}")
        except Exception as _:
            print("Failed to download image")


def accept_cookies():
    try:
        shadow_host = WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.ID, "cmpwrapper"))
        )
        # Get the shadow root
        shadow_root = browser.execute_script(
            "return arguments[0].shadowRoot", shadow_host
        )
        # Find the accept button inside the shadow DOM
        accept_button = shadow_root.find_element(By.ID, "cmpwelcomebtnyes")
        accept_button.click()
        print("Accepted cookies")
    except Exception as e:
        print(f"No cookie prompt or error accepting cookies: {e}")


def goto_next_location():
    next_button = browser.find_element(By.ID, "next")
    ActionChains(browser).move_to_element(next_button).click().perform()
    time.sleep(5)  # Wait for new image to load


def get_location():
    address_element = browser.find_element(By.ID, "address")
    return address_element.text


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
        known_coords[panoidID] = (location["lat"], location["lng"])

        return location["lat"], location["lng"]

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        raise e


if __name__ == "__main__":
    country = "co"
    directory = "colombia"
    os.makedirs(directory, exist_ok=True)
    try:
        browser.get(f"https://randomstreetview.com/{country}")
        WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.ID, "next")))

        dont_download_urls = set()
        logs = browser.get_log("performance")
        image_urls = get_image_urls(logs)
        dont_download_urls.update(image_urls)

        accept_cookies()

        for _ in range(89):
            goto_next_location()

            logs = browser.get_log("performance")
            image_urls = get_image_urls(logs)
            lat, lng = get_coords(logs)
            to_download = [url for url in image_urls if url not in dont_download_urls]

            print(f"Downloading {len(to_download)} new images")
            location_key = random_string()
            with open(os.path.join(directory, "locations.csv"), "a+") as file:
                file.write(f"{location_key},{lat},{lng}\n")
            download_images(to_download, location_key, directory)
            dont_download_urls.update(to_download)
    finally:
        browser.quit()
