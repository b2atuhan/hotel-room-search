import os
import base64
import openai
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
from cache_manager import CacheManager

load_dotenv()

class ImageProcessor:
    def __init__(self, api_key=None):
        """Initialize the image processor with OpenAI API key."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        openai.api_key = self.api_key
        self.cache = CacheManager()

    def encode_image_to_base64(self, image_path):
        """Encode image to base64 format."""
        with open(image_path, "rb") as img_file:
            b64_data = base64.b64encode(img_file.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64_data}"

    def caption_image(self, image_path):
        """Generate structured caption for a hotel room image."""
        # Check cache first
        cache_key = f"caption_{image_path}"
        cached_caption = self.cache.get(cache_key)
        if cached_caption:
            return cached_caption

        try:
            base64_image = self.encode_image_to_base64(image_path)

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": base64_image}
                            },
                            {
                                "type": "text",
                                "text": (
                                    "You are analyzing a hotel room image. Please describe it using the following structured format:\n\n"
                                    "bed_count: (1xDouble and 2xSingle ...)(Give the bed size details as a list)\n"
                                    "max_guest_capacity: (e.g., 2, 3, 4(you can understand by looking at the pillow count on beds and bed sizes))\n"
                                    "view: (e.g., sea, city, garden, mountain, none)\n"
                                    "heating_cooling: (e.g., air conditioning, fan, heater, none(for cooling only consider Split or Ductless Air Conditioners as they are the only ones that used for cooling in Turkey))\n"
                                    "furnitures: (e.g., desk(1)(which you can work on), chair(2), bed(2), lamp(3), wardrobe(1), bed side table(1))\n\n"
                                    "rooms: (balcony, guest room, master bedroom)(give as a list)"
                                    "If an attribute is unclear from the image, write \"unknown\" for that field."
                                    "You can select more than one option"
                                )
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            caption = response.choices[0].message.content
            # Cache the caption
            self.cache.set(cache_key, caption)
            return caption
        except Exception as e:
            error_msg = f"Error: {e}"
            self.cache.set(cache_key, error_msg)
            return error_msg

    def process_image_folder(self, image_folder, output_csv="hotel_image_captions.csv"):
        """Process all images in a folder and save captions to CSV."""
        image_paths = [os.path.join(image_folder, f"{i}.jpg") for i in range(1, 26)]
        
        captions = []
        for path in tqdm(image_paths, desc="Processing images"):
            caption = self.caption_image(path)
            captions.append({
                "image_path": path,
                "caption": caption,
                "image_url": f"https://static.obilet.com.s3.eu-central-1.amazonaws.com/CaseStudy/HotelImages/{os.path.basename(path)}"
            })

        df = pd.DataFrame(captions)
        df.to_csv(output_csv, index=False)
        return df 