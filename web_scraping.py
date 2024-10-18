from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import requests
import time
import json

# Setup Edge WebDriver options
options = webdriver.ChromeOptions() # for mac - chrome browser
#options = webdriver.EdgeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--enable-javascript')

# Directory where you want to save the images
save_directory = "auction_images"
os.makedirs(save_directory, exist_ok=True)

# Initialize global counters
auction_counter = 1
image_counter = 1

# Main scraping loop for multiple pages
current_page = 1
total_pages = 1  # Adjust based on the number of pages
url_template = 'https://www.storagetreasures.com/auctions/tx/dallas/?page={}'

# Dictionary to store image details: auction number -> list of image dictionaries
results = {}

# Deal with dublicate image URLs
downloaded_images = set()

# Initialize the WebDriver
driver = webdriver.Edge(options=options)
web_loc = 'https://www.storagetreasures.com/auctions/tx/dallas/'
driver.get(web_loc)

# Function to download and save images and add details to the results dictionary
def download_image(image_url, auction_id, auction_counter, image_counter):
    try:
        # Skip if the image URL has already been downloaded
        if image_url in downloaded_images:
            print(f"Image {image_counter} from auction {auction_counter} already downloaded, skipping.")
            return
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
                "image_url": image_url,
                "auction_id": auction_id.split('-')[-1] 
            })
            # Add the URL to the set to mark it as downloaded
            downloaded_images.add(image_url)
        else:
            print(f"Failed to download image {image_counter} from auction {auction_counter}: {response.status_code}")
    except Exception as e:
        print(f"Error downloading image {image_counter} from auction {auction_counter}: {str(e)}")


# Function to scrape images from a given auction link
def scrape_auction_images(link, auction_id, auction_counter, image_counter):
    driver.get(link)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".image-gallery-container")))
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    # Find all image elements in the auction page
    img_elements = soup.find_all('img')
    for img_element in img_elements:
        image_url = img_element.get('src')
        if image_url:
            download_image(image_url, auction_id, auction_counter, image_counter)
            image_counter += 1
        else:
            print(f"No src attribute found for image {image_counter} from auction {auction_counter}")
    
    return image_counter  # Return the updated image counter

# Function to scrape auction links from a given URL
def get_auction_links(url):
    driver.get(url)

    # Wait for the auction tiles to load
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".auction-tile")))

    # Get the page source
    html = driver.page_source

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    # Find all <a> tags with the class 'auction-tile' and extract their href attributes
    auction_data = []
    for a_tag in soup.find_all('a', class_='auction-tile'):
        href = a_tag.get('href')
        auction_id = a_tag.get('id')  # Assuming the ID is available for identification
        if href:
            # Construct the full URL if needed (base URL + href)
            full_url = f"https://www.storagetreasures.com{href}"
            auction_data.append({"link": full_url, "auction_id": auction_id})
    return auction_data

# Function to navigate to a specific page
def navigate_to_page(page_number):
    try:
        page_element = driver.find_element(By.ID, f'pagination-page-{page_number}')
        page_element.click()
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, f'pagination-page-{page_number}')))
        print(f"Navigated to page {page_number}")
    except Exception as e:
        print(f"Error navigating to page {page_number}: {e}")

# Function to save results to a JSON file
def save_results_to_json(results, filename="auction_results.json"):
    with open(filename, "w") as json_file:
        json.dump(results, json_file, indent=4)
    print(f"Results saved to {filename}")

while current_page <= total_pages:
    url = url_template.format(current_page)
    print(f"Scraping data from page {current_page}")
    auction_data = get_auction_links(url)  # Use get_auction_links to get auction links for the current page

    for auction in auction_data:
        link = auction['link']
        auction_id = auction['auction_id']
        print(f"Processing auction link: {link} (ID: {auction_id})")
        image_counter = scrape_auction_images(link, auction_id, auction_counter, image_counter)
        auction_counter += 1

    if current_page < total_pages:
        current_page += 1
        navigate_to_page(current_page)  # Navigate to the next page
        time.sleep(2)  # Wait for the page to load completely
    else:
        print("Reached the last page.")
        break

# Close the WebDriver
driver.quit()

# Save results to JSON
save_results_to_json(results)
