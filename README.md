# 🌍 Land Use and Land Cover (LULC) Analysis Using AI with Drone-Captured Aerial Footage

This project is an end-to-end AI-powered platform for automating **Land Use and Land Cover (LULC) classification** from drone-captured aerial imagery. We built a web-based tool that uses a trained **U-Net model** to perform semantic segmentation and calculate area statistics on high-resolution orthomosaic GeoTIFF images.

> 📍 Study Area: Harisiddhi, Lalitpur, Nepal  
> 🚀 Developed as a Major Project for Bachelor in Electronics, Communication and Information Engineering

---

## 📸 Overview

The system combines:
- Drone-based data acquisition 🛩️
- Photogrammetry processing 📷
- Deep learning segmentation 🧠
- Web-based spatial interaction 🌐

Users can select areas on an interactive map, and the backend returns real-time segmentation masks and land use statistics.

---

## 🧰 Tech Stack

| Component            | Technology Used                   |
|----------------------|-----------------------------------|
| Drone                | DJI Mavic 3 Enterprise             |
| Photogrammetry Tool  | Agisoft Metashape                 |
| Annotation Tool      | Label Studio                      |
| Deep Learning        | TensorFlow, U-Net Architecture    |
| Backend              | FastAPI, OpenCV, NumPy            |
| Frontend             | Leaflet.js, Chart.js              |
| File Format Support  | GeoTIFF, COCO JSON, PNG, CSV      |

---

