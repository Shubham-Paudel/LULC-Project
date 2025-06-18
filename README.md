# 🌍 Land Use and Land Cover (LULC) Analysis Using AI with Drone-Captured Aerial Footage

This project is an end-to-end AI-powered platform for automating **Land Use and Land Cover (LULC) classification** using high-resolution drone-captured imagery. It uses deep learning-based segmentation (U-Net) and web technologies to allow interactive analysis of selected areas.

> 📍 Study Area: Harisiddhi, Lalitpur, Nepal  
> 🎓 Developed as a Major Project for B.E. in Electronics, Communication and Information Engineering  
> 🏫 Advanced College of Engineering and Management, Tribhuvan University

---

## 📸 Project Overview

This system allows users to:

- Upload or access drone-based orthomosaic GeoTIFF images
- Select areas interactively on a web map
- Use a trained U-Net model to segment land into:
  - **Bareland**
  - **Buildings**
  - **Roads**
  - **Vegetation**
  - **Water**
- Generate land classification statistics (area in hectares and percentage)
- Download results (mask, statistics, original crop)

---

## 🧰 Technologies Used

**Hardware & Data Acquisition**
- DJI Mavic 3 Enterprise (RTK enabled)
- Agisoft Metashape (for orthomosaic generation)

**Software Stack**
- **Frontend:** Leaflet.js, Leaflet Draw, Chart.js
- **Backend:** FastAPI, OpenCV, NumPy, TensorFlow
- **Annotation:** Label Studio
- **Model:** U-Net (semantic segmentation)

---

## 🚀 How It Works

1. **Drone Flight Planning**  
   Configured 75% course overlap, 65% side overlap, 100m altitude, 7.1 m/s speed.

2. **Photogrammetry Processing**  
   Used Agisoft Metashape for dense cloud, DEM, and GeoTIFF export.

3. **Data Preprocessing**  
   - Cropped into 256x256 patches
   - Annotated using Label Studio
   - Augmented using rotation, flipping, etc.

4. **Model Training**  
   - U-Net trained with annotated images
   - Achieved high IoU on test dataset

5. **Web Platform Integration**  
   - Users interact with a web map
   - Selected area is sent to backend
   - Prediction mask is generated and returned with analysis
   - Users can visualize and download results

---

## 🧠 Model Details

| Class       | Label | Color (RGB)     | IoU Score |
|-------------|-------|------------------|-----------|
| Bareland    | 0     | (255, 255, 0)    | 0.7595    |
| Buildings   | 1     | (255, 0, 0)      | 0.7491    |
| Roads       | 2     | (128, 128, 128)  | 0.5674    |
| Vegetation  | 3     | (0, 255, 0)      | 0.6679    |
| Water       | 4     | (0, 0, 255)      | 0.4783    |

- **Training Accuracy:** 95.69%  
- **Validation Accuracy:** 87.88%  
- **Patch Size:** 256x256  
- **Augmentation:** Yes  
- **Input Format:** GeoTIFF  
- **Output Format:** PNG, CSV

---

## 🖥️ Web Features

- Interactive map with TIFF overlay using Leaflet.js
- Area selection tool using Leaflet Draw
- Real-time segmentation prediction via FastAPI
- Land class statistics with pie charts using Chart.js
- Export:
  - Cropped original image
  - Predicted segmentation mask
  - Area breakdown CSV
  - ZIP bundle of results

---

## Additional Resources

For the static folder here is the link https://drive.google.com/drive/folders/1YykOcfThE6ftLRhTu8DEEZiQlUSu5Gon?usp=sharing
