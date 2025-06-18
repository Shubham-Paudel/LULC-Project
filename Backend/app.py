from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Tuple  # Add Tuple to imports
import rasterio
import numpy as np
import tensorflow as tf
import cv2
import base64
import os
import csv
import zipfile
from pyproj import Geod

app = FastAPI()


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the trained segmentation model
MODEL_CACHE = {}

# Class-to-RGB mapping and category labels
CLASS_COLORS = {
    0: (0, 162, 255),  # Bareland
    1: (0, 13, 255),   # Building
    2: (240, 32, 160), # Road
    3: (0, 107, 18),   # Vegetation
    4: (255, 110, 0)   # Water
}
CATEGORY_LABELS = {
    0: "Bareland",
    1: "Building",
    2: "Road",
    3: "Vegetation",
    4: "Water"
}

CSV_DIRECTORY = "csv_files"
os.makedirs(CSV_DIRECTORY, exist_ok=True)

# Define TIFF files and their boundaries
TIFF_FILES = [
    {
        "file": "static/Advanceddtiff.tif",
        "bounds": [(85.3330463, 27.6336405), (85.3454091, 27.6481854)],
        "model": "Model/model_epoch_h200.h5"
    },
    {
        "file": "static/advance.tif",
        "bounds": [(85.2858866, 27.6838582), (85.2936236, 27.6914596)],
        "model": "Model/model_epoch_100.h5"
    }
]

PATCH_SIZE = 256  # Model input size

class Selection(BaseModel):
    topLeft: List[float]
    bottomRight: List[float]

# Preprocessing function
def preprocess_image(image, target_size=(PATCH_SIZE, PATCH_SIZE)):
    resized_image = cv2.resize(image, target_size)
    return resized_image / 255.0

# Create RGB mask
def create_rgb_mask(mask):
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        rgb_mask[mask == class_id] = color
    return rgb_mask

# Split image into patches
def split_image(image, patch_size=PATCH_SIZE, overlap=64):
    h, w, _ = image.shape
    patches = []
    stride = patch_size - overlap
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Extract patch with overlap
            patch = image[y:y+patch_size, x:x+patch_size]
            
            # Handle border patches with reflection padding
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                patch = cv2.copyMakeBorder(
                    patch,
                    0,
                    patch_size - patch.shape[0],
                    0,
                    patch_size - patch.shape[1],
                    cv2.BORDER_REFLECT101  # Better than constant padding
                )
            patches.append((x, y, patch))
    return patches, h, w

# Reconstruct full image
# Reconstruct full image
def reconstruct_image(prob_patches, h, w, num_classes, patch_size=PATCH_SIZE, overlap=64):
    # Initialize probability buffer and weight matrix
    prob_buffer = np.zeros((h, w, num_classes), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    
    # Create a 2D Hanning window for blending
    window = np.hanning(patch_size)
    window = np.sqrt(np.outer(window, window))  # 2D window
    
    for x, y, prob in prob_patches:
        # Get actual patch dimensions (might be smaller at borders)
        ph = min(patch_size, h - y)
        pw = min(patch_size, w - x)
        
        # Apply window to probabilities
        weighted_prob = prob[:ph, :pw] * window[:ph, :pw, None]
        
        # Accumulate probabilities and weights
        prob_buffer[y:y+ph, x:x+pw] += weighted_prob
        weight_sum[y:y+ph, x:x+pw] += window[:ph, :pw]
    
    # Normalize using safe division to avoid division by zero
    prob_buffer = np.divide(
        prob_buffer,
        weight_sum[..., None],
        out=np.zeros_like(prob_buffer),  # Set to zero where division is invalid
        where=weight_sum[..., None] > 0  # Only divide where weight_sum > 0
    )
    
    # Create final segmentation mask
    full_mask = np.argmax(prob_buffer, axis=-1).astype(np.uint8)
    return full_mask

# Function to overlay segmentation mask on cropped image
def blend_images(original, mask, alpha=0.5):
    """ Blend the segmentation mask over the cropped image with transparency. """
    mask_colored = create_rgb_mask(mask)
    blended = cv2.addWeighted(original, 1 - alpha, mask_colored, alpha, 0)
    return blended

def get_tiff_file_for_selection(top_left: Tuple[float, float], bottom_right: Tuple[float, float]):
    for tiff in TIFF_FILES:
        min_lon, min_lat = tiff["bounds"][0]
        max_lon, max_lat = tiff["bounds"][1]

        if (min_lon <= top_left[1] <= max_lon and min_lat <= top_left[0] <= max_lat and
            min_lon <= bottom_right[1] <= max_lon and min_lat <= bottom_right[0] <= max_lat):
            return tiff["file"], tiff["model"]

    return None, None  # No valid TIFF found

def calculate_pixel_area_km2(transform, src_bounds):
    pixel_width_deg = transform[0]
    pixel_height_deg = abs(transform[4])
    center_lat = (src_bounds.bottom + src_bounds.top) / 2
    
    geod = Geod(ellps="WGS84")
    _, _, lon_width = geod.inv(
        src_bounds.left, center_lat,
        src_bounds.left + pixel_width_deg, center_lat
    )
    _, _, lat_height = geod.inv(
        center_lat, src_bounds.left,
        center_lat, src_bounds.left + pixel_height_deg
    )
    
    # Calculate area per pixel in square meters
    area_per_pixel_m2 = lon_width * lat_height
    # Convert to square kilometers
    pixel_area_km2 = area_per_pixel_m2 / 10000
    return pixel_area_km2

@app.post("/crop_and_predict")
async def crop_and_predict(selection: Selection):
    selected_tiff, model_path = get_tiff_file_for_selection(tuple(selection.topLeft), tuple(selection.bottomRight))

    if not selected_tiff or not model_path:
        raise HTTPException(status_code=400, detail="Selected area is out of bounds.")
    
    try:
        if model_path not in MODEL_CACHE:
            MODEL_CACHE[model_path] = tf.keras.models.load_model(model_path, compile=False)
        model = MODEL_CACHE[model_path]

        with rasterio.open(selected_tiff) as src:
            transform = src.transform
            src_bounds = src.bounds

            # Calculate area per pixel
            pixel_area_km2 = calculate_pixel_area_km2(transform, src_bounds)

            row1, col1 = src.index(selection.topLeft[1], selection.topLeft[0])
            row2, col2 = src.index(selection.bottomRight[1], selection.bottomRight[0])

            row1, row2 = max(0, row1), min(src.height, row2)
            col1, col2 = max(0, col1), min(src.width, col2)

            if row1 >= row2 or col1 >= col2:
                raise HTTPException(status_code=400, detail="Invalid crop coordinates.")

            # Crop the image
            cropped_image = src.read([1, 2, 3], window=rasterio.windows.Window(col1, row1, col2 - col1, row2 - row1))
            cropped_rgb = np.moveaxis(cropped_image, 0, -1)  # Convert to (H, W, C)
            cropped_rgb = cv2.cvtColor(cropped_rgb, cv2.COLOR_BGR2RGB)

            # Save the cropped image as a file
            cropped_image_path = os.path.join(CSV_DIRECTORY, "user_selected_area.jpg")
            cv2.imwrite(cropped_image_path, cropped_rgb)

            # Encode the cropped image as base64 for the frontend
            _, cropped_image_base64 = cv2.imencode(".jpg", cropped_rgb)
            cropped_image_base64 = base64.b64encode(cropped_image_base64).decode("utf-8")

        # Segment the cropped image
        num_classes = model.output_shape[-1]
        patches, h, w = split_image(cropped_rgb, overlap=64)
        prob_patches = []
        for x, y, patch in patches:
            processed_patch = preprocess_image(patch)
            processed_patch = np.expand_dims(processed_patch, axis=0)
            prediction = model.predict(processed_patch)[0]  # Raw probabilities
            prob_patches.append((x, y, prediction))

        # Reconstruct the segmentation mask
        full_mask = reconstruct_image(prob_patches, h, w, num_classes, overlap=64)

        # Create RGB mask and blended image
        rgb_mask = create_rgb_mask(full_mask)
        segmentation_mask_path = os.path.join(CSV_DIRECTORY, "segmentation_mask.png")
        cv2.imwrite(segmentation_mask_path, rgb_mask)

        blended_image = blend_images(cropped_rgb, full_mask, alpha=0.5)
        blended_image_path = os.path.join(CSV_DIRECTORY, "overlapped_mask.png")
        cv2.imwrite(blended_image_path, blended_image)

        # Encode images as base64
        _, segmentation_mask_base64 = cv2.imencode(".png", rgb_mask)
        segmentation_mask_base64 = base64.b64encode(segmentation_mask_base64).decode("utf-8")

        _, blended_image_base64 = cv2.imencode(".png", blended_image)
        blended_image_base64 = base64.b64encode(blended_image_base64).decode("utf-8")

        # Compute area statistics
        pixel_counts = {label: int(np.sum(full_mask == class_id)) for class_id, label in CATEGORY_LABELS.items()}
        total_pixels = full_mask.shape[0] * full_mask.shape[1]
        total_area_km2 = pixel_area_km2 * total_pixels
        area_statistics = {
            label: {
                "pixels": count,
                "area_km2": count * pixel_area_km2,
                "proportion": (count * pixel_area_km2) / total_area_km2 if total_area_km2 > 0 else 0
            }
            for label, count in pixel_counts.items()
        }

        # Save area statistics to CSV
        csv_filename = os.path.join(CSV_DIRECTORY, "area_statistics.csv")
        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Category", "Pixels", "Area (sq. km)", "Proportion (%)"])
            for label, stats in area_statistics.items():
                writer.writerow([
                    label,
                    stats["pixels"],
                    f"{stats['area_km2']:.6f}",
                    f"{stats['proportion']*100:.2f}"
                ])

        return {
            "original_image": cropped_image_base64,
            "segmentation_map": segmentation_mask_base64,
            "blended_image": blended_image_base64,
            "area_statistics": area_statistics
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/download_zip")
async def download_zip():
    zip_filename = "segmentation_results.zip"
    zip_path = os.path.join(CSV_DIRECTORY, zip_filename)

    try:
        files_to_zip = [
            "user_selected_area.jpg",
            "segmentation_mask.png",
            "overlapped_mask.png",
            "area_statistics.csv"
        ]

        with zipfile.ZipFile(zip_path, "w") as zipf:
            for filename in files_to_zip:
                filepath = os.path.join(CSV_DIRECTORY, filename)
                if os.path.exists(filepath):
                    zipf.write(filepath, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating ZIP file: {str(e)}")

    return FileResponse(zip_path, media_type="application/zip", filename=zip_filename)
