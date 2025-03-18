import os
import cv2
import numpy as np
import json
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from fractions import Fraction

# Initialize FastAPI app
app = FastAPI()

# Adjusted Path: Images folder is **outside** the backend folder
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Parent of 'backend'
IMAGE_FOLDER = os.path.join(BASE_DIR, "images")

# Convert GPS DMS format to Decimal Degrees
def convert_to_decimal(gps_data):
    """ Convert DMS (degrees, minutes, seconds) to decimal degrees. """
    try:
        degrees, minutes, seconds = gps_data
        return float(degrees) + (float(minutes) / 60) + (float(seconds) / 3600)
    except Exception:
        return None

# Extract and Convert GPS Metadata
def get_gps_data(image):
    try:
        exif_data = image._getexif()
        if not exif_data:
            return None

        gps_data = {}
        lat, lon = None, None

        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name == "GPSInfo":
                for t in value.keys():
                    gps_tag = GPSTAGS.get(t, t)
                    # Convert IFDRational values to float
                    if isinstance(value[t], tuple):
                        gps_data[gps_tag] = [float(x) if isinstance(x, Fraction) else x for x in value[t]]
                    elif isinstance(value[t], Fraction):
                        gps_data[gps_tag] = float(value[t])
                    else:
                        gps_data[gps_tag] = value[t]

        # Convert DMS to Decimal Degrees
        if "GPSLatitude" in gps_data and "GPSLongitude" in gps_data:
            lat = convert_to_decimal(gps_data["GPSLatitude"])
            lon = convert_to_decimal(gps_data["GPSLongitude"])

            # Adjust for N/S and E/W
            if gps_data.get("GPSLatitudeRef") == "S":
                lat = -lat
            if gps_data.get("GPSLongitudeRef") == "W":
                lon = -lon

        return {
            "latitude": lat,
            "longitude": lon,
            "raw_gps_data": gps_data  # Optional: Include raw data for debugging
        }
    except Exception as e:
        return {"error": str(e)}

# Function to Process Local Images
def process_local_images():
    if not os.path.exists(IMAGE_FOLDER):
        return JSONResponse(content={"error": "Images folder not found!"}, status_code=500)

    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        return JSONResponse(content={"message": "⚠️ No images found in the 'images' folder!"})

    results = []
    for image_name in image_files:
        image_path = os.path.join(IMAGE_FOLDER, image_name)

        # Open image with PIL
        image = Image.open(image_path)
        gps_info = get_gps_data(image)

        # Convert image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image_resized = cv2.resize(image_cv, (224, 224)) / 255.0  # Normalize
        image_processed = np.expand_dims(image_resized, axis=0)  # Add batch dimension

        results.append({
            "filename": image_name,
            "gps_info": gps_info if gps_info else "No GPS metadata found",
            "image_shape": image_processed.shape
        })
    
    return JSONResponse(content=json.loads(json.dumps(results, default=str)))  # Ensure JSON serialization

# API Route - Home
@app.get("/")
def read_root():
    return JSONResponse(content={"message": "Hello, AI Relief API is running!"})

# API Route - Process Images
@app.get("/process-images/")
def api_process_images():
    return process_local_images()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
