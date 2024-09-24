import os
import numpy as np
import json
from PIL import Image, ImageDraw

# Load JSON file
with open('D:\\project-8-at-2024-09-24-19-06-4b110dca\\result.json', 'r') as f:
    data = json.load(f)

# Define specific color for each category (RGB format)
category_colors = {
    0: (255, 162, 0),  # Orange
    1: (255, 13, 0),   # Red
    2: (160, 32, 240), # Purple
    3: (18, 107, 0),   # Green
    4: (0, 110, 255)   # Blue
}

# Create output directory if not exists
output_dir = 'D:\\project-8-at-2024-09-24-19-06-4b110dca\\masks'
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

    # Create a blank RGBA mask image for transparency support
    mask = Image.new('RGBA', (width, height), (0, 0, 0, 0))  # 'RGBA' mode for transparency

    # Debug: Draw a visible rectangle to ensure the mask is working
    draw = ImageDraw.Draw(mask)
    draw.rectangle([10, 10, 100, 100], outline=(255, 0, 0, 255), fill=(255, 0, 0, 255))  # Red rectangle

    # Get annotations for the current image
    if image_id in annotations_by_image:
        for annotation in annotations_by_image[image_id]:
            # Extract segmentation points
            segmentation = annotation['segmentation'][0]  # First segmentation in the list
            category_id = annotation['category_id']
            
            # Check if segmentation values are already in pixels or need scaling
            points = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]
            
            # Check if points are within the expected range
            print(f"Image ID: {image_id}, Category: {category_id}, Points: {points}")

            # Create a separate layer for each polygon
            poly_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))  # Transparent layer
            draw_poly = ImageDraw.Draw(poly_layer)
            
            # Ensure category_colors[category_id] is valid and add the alpha channel
            color = category_colors.get(category_id, (0, 0, 0))  # Default to black if category_id not found
            color_with_alpha = color + (255,)  # Add alpha (255 = fully opaque)
            
            # Fill the polygon with a color and transparency and also draw the outline
            draw_poly.polygon(points, fill=color_with_alpha, outline=color_with_alpha)

            # Composite the polygon layer onto the main mask (which supports transparency)
            mask = Image.alpha_composite(mask, poly_layer)

    # Save the mask as PNG to preserve transparency and check the output
    output_file_name = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_name))[0]}_mask.png")
    mask.save(output_file_name)

    print(f"Mask saved for {file_name} as {output_file_name}")

print("All masks generated and saved.")
