import os
import requests
from tqdm import tqdm
from image_processor import ImageProcessor
from search_engine import HotelRoomSearchEngine
from dotenv import load_dotenv

load_dotenv()

def download_images(base_url, output_dir="hotel_images", num_images=25):
    """Download hotel room images."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in tqdm(range(1, num_images + 1), desc="Downloading images"):
        image_url = f"{base_url}{i}.jpg"
        image_path = f"{output_dir}/{i}.jpg"
        
        if not os.path.exists(image_path):
            try:
                response = requests.get(image_url, verify=False)
                response.raise_for_status()
                with open(image_path, "wb") as f:
                    f.write(response.content)
            except Exception as e:
                print(f"❌ Failed to download {image_url}: {e}")

def main():
    BASE_URL = "https://static.obilet.com.s3.eu-central-1.amazonaws.com/CaseStudy/HotelImages/"
    IMAGE_DIR = "hotel_images"
    CAPTIONS_CSV = "hotel_image_captions.csv"
    
    #Download images
    print("Step 1: Downloading images...")
    download_images(BASE_URL, IMAGE_DIR)
    
    #Process images
    print("\nStep 2: Processing images and generating captions...")
    processor = ImageProcessor()
    df = processor.process_image_folder(IMAGE_DIR, CAPTIONS_CSV)
    
    #search engine
    print("\nStep 3: Initializing search engine...")
    search_engine = HotelRoomSearchEngine()
    search_engine.load_data(CAPTIONS_CSV)
    
    # Searches
    print("\nStep 4: Performing searches...")
    queries = [
        "Query 1: Double rooms with sea view",
        "Query 2: Balcony + AC + City view",
        "Query 3: Triple room with desk",
        "Query 4: Capacity of 4"
    ]
    
    for query_type in queries:
        print(f"\n{query_type}")
        results = search_engine.search_rooms(query_type)
        for url, score in results:
            print(f" - {url} (score: {score:.4f})")

if __name__ == "__main__":
    main() 