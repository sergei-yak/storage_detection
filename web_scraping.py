from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import requests
import time

# Setup Chrome Options
options = webdriver.EdgeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--enable-javascript')

# Initialize the WebDriver
driver = webdriver.Edge(options=options)
web_loc = 'https://www.storagetreasures.com/auctions/tx/dallas/'
driver.get(web_loc)

# Directory where you want to save the images
save_directory = "auction_images"
os.makedirs(save_directory, exist_ok=True)

# Initialize global counters
auction_counter = 1
image_counter = 1

# Dictionary to store image details: auction number -> list of image dictionaries
results = {}

# Function to download and save images and add details to the results dictionary
def download_image(image_url, auction_counter, image_counter):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            # Save the image locally
            filename = f"auction_{auction_counter}_image_{image_counter}.jpg"
            filepath = os.path.join(save_directory, filename)
            with open(filepath, "wb") as file:
                file.write(response.content)

            print(f"Image {image_counter} from auction {auction_counter} saved as {filename}")

            # Store the image details in the results dictionary
            if auction_counter not in results:
                results[auction_counter] = []  # Initialize a list for this auction if not already present
            results[auction_counter].append({
                "image_name": filename,
                "image_url": image_url
            })
        else:
            print(f"Failed to download image {image_counter} from auction {auction_counter}: {response.status_code}")
    except Exception as e:
        print(f"Error downloading image {image_counter} from auction {auction_counter}: {str(e)}")

# Function to scrape images from a given auction link
def scrape_auction_images(link, auction_counter, image_counter):
    driver.get(link)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".image-gallery-container")))
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    # Find all image elements in the auction page
    img_elements = soup.find_all('img')
    for img_element in img_elements:
        image_url = img_element.get('src')
        if image_url:
            download_image(image_url, auction_counter, image_counter)
            image_counter += 1
        else:
            print(f"No src attribute found for image {image_counter} from auction {auction_counter}")
    
    return image_counter  # Return the updated image counter

# Function to scrape auction links from the current page
def get_auction_links():
    auction_tiles = driver.find_elements(By.CSS_SELECTOR, ".auction-tiles-special-center a.auction-tile")
    return [tile.get_attribute("href") for tile in auction_tiles]

# Function to navigate to a specific page
def navigate_to_page(page_number):
    try:
        page_element = driver.find_element(By.ID, f'pagination-page-{page_number}')
        page_element.click()
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, f'pagination-page-{page_number}')))
        print(f"Navigated to page {page_number}")
    except Exception as e:
        print(f"Error navigating to page {page_number}: {e}")

# Main scraping loop for multiple pages
current_page = 1
total_pages = 10  # Adjust based on the number of pages

while current_page <= total_pages:
    print(f"Scraping data from page {current_page}")
    auction_links = get_auction_links()  # Get all auction links on the current page

    for link in auction_links:
        print(f"Processing auction link: {link}")
        image_counter = scrape_auction_images(link, auction_counter, image_counter)
        auction_counter += 1  # Increment auction counter after processing each auction

    if current_page < total_pages:
        current_page += 1
        navigate_to_page(current_page)  # Navigate to the next page
        time.sleep(2)  # Wait for the page to load completely
    else:
        print("Reached the last page.")
        break

# Close the WebDriver
driver.quit()
