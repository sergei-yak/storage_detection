from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import os
import requests


chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--enable-javascript')


# Initialize the WebDriver
web_loc = 'https://www.storagetreasures.com/auctions/tx/dallas/'
driver = webdriver.Chrome(options=chrome_options)
driver.get(web_loc)
# Inject JavaScript code to navigate to the Dallas, TX page.
#driver.execute_script('window.location.href = "https://www.storagetreasures.com/auctions/tx/dallas/";')

# If you want to process the page source with BeautifulSoup
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

# Find all the div elements with the class 'auction-img'
img_elements = soup.find_all('div', attrs={'class': 'auction-img'})
# Loop through each div
url_list = []
id_url = {}
# Loop through the div elements and extract the background image URLs
for div_element in img_elements:
    # Get the style attribute content
    style = div_element.get('style')

    # Use regular expression to extract the URL within the url("...") pattern
    match = re.search(r'url\("([^"]+)"\)', style)

    if match:
        # Extract and print the background image URL
        background_image_url = match.group(1)
        url_list.append(background_image_url)
        print("Background Image URL:", background_image_url)

    else:
        print("No background image URL found in style attribute.")


# Directory where you want to save the images
save_directory = "auction_images"

# Create the directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Loop through the image URLs and download/save each image
for index, image_url in enumerate(url_list, start=1):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            # Generate a filename for the image
            filename = os.path.join(save_directory, f"image{index}.jpg")

            # Save the image to the local directory
            with open(filename, "wb") as file:
                file.write(response.content)
            for key, value in id_url.items():
              if image_url in value:
                id_url[key].append(filename)

            print(f"Image {index} saved as {filename}")
        else:
            print(f"Failed to download image {index}: {response.status_code}")
    except Exception as e:
        print(f"Error downloading image {index}: {str(e)}")

#### click tiles to open each one
# Find all the auction links
auction_tiles = driver.find_elements(By.CSS_SELECTOR, ".auction-tiles-special-center a.auction-tile")

# Extract the href attribute from each of the found elements
href_links = [tile.get_attribute("href") for tile in auction_tiles]

# Loop through each auction link
#for i in range(len(href_links)):
for i, link in enumerate(href_links):

    # Click the auction link
    #href_links[i].click()
    driver.get(link)

    # Wait for the auction page to load
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".image-gallery-container")))

    # Process the page source with BeautifulSoup
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    # Find all the div elements with the class 'auction-img'
    img_giants = soup.find_all('img')

    # Loop through each div and extract the background image URLs
    for idx, img_element in enumerate(img_giants, start=1):
        # Extract the src attribute
        giant_image_url = img_element.get('src')

        if giant_image_url:
            try:
                response = requests.get(giant_image_url)
                if response.status_code == 200:
                    # Generate a filename for the image
                    filename = os.path.join(save_directory, f"auction_{i + 1}_image_{idx}.jpg")

                    # Save the image to the local directory
                    with open(filename, "wb") as file:
                        file.write(response.content)

                    print(f"Image {idx} from auction {i + 1} saved as {filename}")
                else:
                    print(f"Failed to download image {idx} from auction {i + 1}: {response.status_code}")
            except Exception as e:
                print(f"Error downloading image {idx} from auction {i + 1}: {str(e)}")
        else:
            print(f"No src attribute found for image {idx} from auction {i + 1}")

    # Go back to the main auction listing page
    driver.back()

    # Wait for the auction listing page to load again
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".auction-tile.horizontal-tile")))

# Close the WebDriver
driver.quit()