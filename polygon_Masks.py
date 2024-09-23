import os
import numpy as np
import json
from PIL import Image, ImageDraw

# Load JSON file
with open('D:\\project-8-at-2024-09-23-21-17-b65d6127\\result.json', 'r') as f:
    data = json.load(f)

# Define specific color for each category
category_colors = {
    0: (255, 13, 0),  # Red color for "Building" (category_id = 0)
    1: (0, 0, 0),
    2: (18, 107, 0),
    3: (0, 110, 255), 
}

# Create output directory if not exists
output_dir = 'D:\\project-8-at-2024-09-23-21-17-b65d6127\\masks'
os.makedirs(output_dir, exist_ok=True)

# Group annotations by image_id
annotations_by_image = {}
for annotation in data['annotations']:
    image_id = annotation['image_id']
    if image_id not in annotations_by_image:
        annotations_by_image[image_id] = []
    annotations_by_image[image_id].append(annotation)

# Process each image
for image_info in data['images']:
    image_id = image_info['id']
    width = image_info['width']
    height = image_info['height']
    file_name = image_info['file_name']

    # Create a blank RGB mask image for the current image
    mask = Image.new('RGB', (width, height), (0, 0, 0))  # 'RGB' mode for color

    # Create a drawing object to modify the mask
    draw = ImageDraw.Draw(mask)

    # Get annotations for the current image
    if image_id in annotations_by_image:
        for annotation in annotations_by_image[image_id]:
            # Extract segmentation points
            segmentation = annotation['segmentation'][0]  # First segmentation in the list
            category_id = annotation['category_id']
            
            # Prepare points as (x1, y1, x2, y2, ..., xn, yn)
            points = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]
            
            # Draw the polygon with the color corresponding to the category_id
            draw.polygon(points, outline=category_colors[category_id], fill=category_colors[category_id])

    # Save the colored mask as a JPG file using the image's original file name
    output_file_name = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_name))[0]}_mask.jpg")
    mask.save(output_file_name)

    print(f"Mask saved for {file_name} as {output_file_name}")

print("All masks generated and saved.")
