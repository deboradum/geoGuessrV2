import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
import os
import time
from PIL import Image
from io import BytesIO

# Set up Selenium WebDriver
options = webdriver.ChromeOptions()
# options.add_argument(
#     "--headless"
# )  # Run headless (remove if you want to see the browser)
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920x1080")
options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

service = Service("chromedriver")  # Adjust path if necessary
browser = webdriver.Chrome(service=service, options=options)


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


def download_images(urls, folder="images"):
    """Downloads images from given URLs."""
    os.makedirs(folder, exist_ok=True)

    for i, url in enumerate(urls):
        print(f"Downloading {i+1}/{len(urls)}: {url}")
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                try:
                    img = Image.open(BytesIO(response.content))
                    width, height = img.size

                    if width == 512 and height == 512:
                        # Extract filename from URL or use a generic one
                        filename = os.path.basename(url).split("?")[0]
                        if not any(
                            filename.lower().endswith(ext)
                            for ext in [
                                ".jpg",
                                ".jpeg",
                            ]
                        ):
                            ext = ".jpg"
                            filename = f"image_{i}{ext}"

                        with open(os.path.join(folder, filename), "wb") as file:
                            file.write(response.content)

                except (IOError, OSError) as e:
                    print(f"Unable to determine image dimensions: {e}")

                # Extract filename from URL or use a generic one
                filename = os.path.basename(url).split("?")[0]
                if not any(
                    filename.lower().endswith(ext)
                    for ext in [
                        ".jpg",
                        ".jpeg",
                    ]
                ):
                    ext = ".jpg"
                    filename = f"image_{i}{ext}"

                with open(os.path.join(folder, filename), "wb") as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
            else:
                print(f"Failed to download {url}: HTTP {response.status_code}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")


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



if __name__ == "__main__":
    try:
        browser.get("https://randomstreetview.com")
        WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.ID, "next")))

        dont_download_urls = set()
        logs = browser.get_log("performance")
        image_urls = get_image_urls(logs)
        dont_download_urls.update(image_urls)

        accept_cookies()

        for _ in range(10):
            goto_next_location()

            logs = browser.get_log("performance")
            image_urls = get_image_urls(logs)
            to_download = [url for url in image_urls if url not in dont_download_urls]
            print(
                f"Downloading {len(to_download)} new images"
            )
            download_images(to_download)
            dont_download_urls.update(to_download)
    finally:
        browser.quit()
