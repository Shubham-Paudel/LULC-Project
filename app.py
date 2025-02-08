from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import rasterio
import numpy as np
import tensorflow as tf
import cv2
import base64
import os
import csv
import zipfile

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
model = tf.keras.models.load_model("model_epoch_50.h5", compile=False)

# Class-to-RGB mapping and category labels
CLASS_COLORS = {
    0: (0, 162, 255),  # Building
    1: (0, 13, 255),   # Land
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

TIFF_FILE = "static/Advancedtiff.tif"
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
def split_image(image, patch_size=PATCH_SIZE):
    h, w, _ = image.shape
    patches = []
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = image[y:y + patch_size, x:x + patch_size]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                patch = cv2.copyMakeBorder(
                    patch,
                    0,
                    patch_size - patch.shape[0],
                    0,
                    patch_size - patch.shape[1],
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
            patches.append((x, y, patch))
    return patches, h, w

# Reconstruct full image
def reconstruct_image(patches, h, w, patch_size=PATCH_SIZE):
    full_mask = np.zeros((h, w), dtype=np.uint8)
    for x, y, patch in patches:
        patch = patch[:min(h - y, patch_size), :min(w - x, patch_size)]
        full_mask[y:y + patch.shape[0], x:x + patch.shape[1]] = patch
    return full_mask

@app.post("/crop_and_predict")
async def crop_and_predict(selection: Selection):
    try:
        with rasterio.open(TIFF_FILE) as src:
            row1, col1 = src.index(selection.topLeft[1], selection.topLeft[0])
            row2, col2 = src.index(selection.bottomRight[1], selection.bottomRight[0])

            row1, row2 = max(0, row1), min(src.height, row2)
            col1, col2 = max(0, col1), min(src.width, col2)

            if row1 >= row2 or col1 >= col2:
                raise HTTPException(status_code=400, detail="Invalid crop coordinates.")

            # Crop the image
            cropped_image = src.read([1, 2, 3], window=rasterio.windows.Window.from_slices((row1, row2), (col1, col2)))
            cropped_rgb = np.moveaxis(cropped_image, 0, -1)  # Convert to (H, W, C)
            cropped_rgb = cv2.cvtColor(cropped_rgb, cv2.COLOR_BGR2RGB)

            # Save the cropped image as a file
            cropped_image_path = os.path.join(CSV_DIRECTORY, "cropped_image.jpg")
            cv2.imwrite(cropped_image_path, cropped_rgb)

            # Encode the cropped image as base64 for the frontend
            _, cropped_image_base64 = cv2.imencode(".jpg", cropped_rgb)
            cropped_image_base64 = base64.b64encode(cropped_image_base64).decode("utf-8")

            # Segment the cropped image
            patches, h, w = split_image(cropped_rgb)
            predictions = []
            for x, y, patch in patches:
                processed_patch = preprocess_image(patch)
                processed_patch = np.expand_dims(processed_patch, axis=0)
                prediction = model.predict(processed_patch)[0]
                mask = np.argmax(prediction, axis=-1).squeeze()
                predictions.append((x, y, mask))

            # Reconstruct the segmentation mask
            full_mask = reconstruct_image(predictions, h, w)
            rgb_mask = create_rgb_mask(full_mask)

            # Save the segmentation mask as a file
            segmentation_mask_path = os.path.join(CSV_DIRECTORY, "segmentation_mask.png")
            cv2.imwrite(segmentation_mask_path, rgb_mask)

            # Encode the segmentation mask as base64 for the frontend
            _, segmentation_mask_base64 = cv2.imencode(".png", rgb_mask)
            segmentation_mask_base64 = base64.b64encode(segmentation_mask_base64).decode("utf-8")

            # Compute area statistics
            pixel_area_km2 = 3.2919503606115937 / 1_000_000
            pixel_counts = {label: int(np.sum(full_mask == class_id)) for class_id, label in CATEGORY_LABELS.items()}
            total_area_km2 = pixel_area_km2 * full_mask.shape[0] * full_mask.shape[1]
            area_statistics = {
                label: {
                    "pixels": count,
                    "area_km2": count * pixel_area_km2,
                    "proportion": (count * pixel_area_km2) / total_area_km2
                }
                for label, count in pixel_counts.items()
            }

            # Save area statistics to a CSV file
            csv_filename = os.path.join(CSV_DIRECTORY, "area_statistics.csv")
            with open(csv_filename, "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Category", "Pixels", "Area (sq. km)", "Proportion (%)"])
                for label, stats in area_statistics.items():
                    writer.writerow([label, stats["pixels"], stats["area_km2"], stats["proportion"] * 100])

            return {
                "original_image": cropped_image_base64,
                "segmentation_map": segmentation_mask_base64,
                "area_statistics": area_statistics
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/download_zip")
async def download_zip():
    zip_filename = "segmentation_results.zip"
    zip_path = os.path.join(CSV_DIRECTORY, zip_filename)

    csv_file_path = os.path.join(CSV_DIRECTORY, "area_statistics.csv")
    cropped_image_path = os.path.join(CSV_DIRECTORY, "cropped_image.jpg")
    segmentation_mask_path = os.path.join(CSV_DIRECTORY, "segmentation_mask.png")

    try:
        # Create a ZIP file and add all required files
        with zipfile.ZipFile(zip_path, "w") as zipf:
            if os.path.exists(csv_file_path):
                zipf.write(csv_file_path, "area_statistics.csv")
            if os.path.exists(cropped_image_path):
                zipf.write(cropped_image_path, "cropped_image.jpg")
            if os.path.exists(segmentation_mask_path):
                zipf.write(segmentation_mask_path, "segmentation_mask.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating ZIP file: {str(e)}")

    # Return the ZIP file for download
    return FileResponse(zip_path, media_type="application/zip", filename=zip_filename)

