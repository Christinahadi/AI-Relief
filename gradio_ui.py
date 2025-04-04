import gradio as gr
import os
import cv2
import numpy as np
from PIL import Image as PILImage
from backend.main import get_gps_data, reverse_geocode
from utils import default_transform

def process_uploaded_image(filepath):
    # Open original file with EXIF intact
    pil_image = PILImage.open(filepath)

    gps_info = get_gps_data(pil_image)
    lat = gps_info.get("latitude", "N/A") if gps_info else "N/A"
    lon = gps_info.get("longitude", "N/A") if gps_info else "N/A"
    address = "No GPS data"

    if isinstance(lat, float) and isinstance(lon, float):
        address = reverse_geocode(lat, lon)

    # Optionally apply model transform (for future model input compatibility)
    input_tensor = default_transform(pil_image).unsqueeze(0)

    result = f"""
###  Image Metadata

- **Shape**: `{input_tensor.shape}`
- **Latitude**: `{lat}`
- **Longitude**: `{lon}`
- **Address**: `{address}`
"""
    return result

gr.Interface(
    fn=process_uploaded_image,
    inputs=gr.Image(type="filepath"),
    outputs="markdown",
    title="AI Relief: Image Metadata Inspector",
    description="Upload a disaster-related photo to view GPS and reverse-geocoded address."
).launch()
