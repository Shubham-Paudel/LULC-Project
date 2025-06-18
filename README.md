
# 🌍 Land Use and Land Cover (LULC) Analysis Using AI with Drone-Captured Aerial Footage

This is an end-to-end AI-powered platform for automating **Land Use and Land Cover (LULC)** classification using high-resolution drone-captured imagery. It combines deep learning (U-Net) and geospatial web technologies to perform real-time interactive segmentation and statistical analysis of land types.

> 📍 **Study Area:** Harisiddhi, Lalitpur, Nepal and Advanced College of Engineering and Management, Balkhu, Kathmandu, Nepal 
> 🎓 **Developed as:** Major Project for B.E. in Electronics, Communication and Information Engineering  
> 🏫 **College:** Advanced College of Engineering and Management, Tribhuvan University

---

## 📁 Project Folder Structure

```
LULC-Project/
│
├── Backend/                # FastAPI backend for image processing and model inference
├── Frontend/               # Leaflet.js based frontend web interface
├── csv_files/              # Output statistics in CSV format
├── Images/                 # Images for model training
├── masks/                  # Ground truth mask
├── Model/                  # Trained U-Net model
├── static/                 # Static images to load in frontend and draw prediction
│
├── Model-training.ipynb    # U-Net training notebook
├── Major_Project_Final_Report.pdf  # Full academic report
├── README.md             
├── .gitignore
├── requirements.txt        # Python dependencies
```

📎 **Static Folder:** [Download here](https://drive.google.com/drive/folders/1YykOcfThE6ftLRhTu8DEEZiQlUSu5Gon?usp=sharing)

---

## 🚀 How to Run the Project

### 🔧 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### 🖥️ 2. Start the Backend (FastAPI)

```bash
cd Backend
uvicorn main:app --reload
```

> This will start your FastAPI server at `http://127.0.0.1:8000`

---

### 🌐 3. Open the Frontend

Open `Frontend/index.html` in your browser.

You should see:
- Interactive map with GeoTIFF overlay
- Rectangle area selection
- "Predict" and "Download" buttons
- Output mask and statistics chart

---

## 📊 System Functionality

Users can:
- Select area on an interactive map
- Send selected area to backend
- Receive segmented mask (5 classes)
- View area stats with pie chart
- Download:
  - Cropped image
  - Predicted mask
  - CSV with land class breakdown
  - ZIP bundle of results

---

## 🧠 Model Overview

| Class       | Label | RGB Color        | IoU Score |
|-------------|-------|------------------|-----------|
| Bareland    | 0     | (255, 255, 0)    | 0.7595    |
| Buildings   | 1     | (255, 0, 0)      | 0.7491    |
| Roads       | 2     | (128, 128, 128)  | 0.5674    |
| Vegetation  | 3     | (0, 255, 0)      | 0.6679    |
| Water       | 4     | (0, 0, 255)      | 0.4783    |

- **Training Accuracy:** 95.69%  
- **Validation Accuracy:** 87.88%  
- **Patch Size:** 256×256  
- **Input Format:** GeoTIFF  
- **Output Format:** PNG, CSV

---

## 🔧 Technologies Used

| Category        | Tools |
|----------------|-------|
| Drone & Mapping | DJI Mavic 3 Enterprise, Agisoft Metashape |
| Annotation      | Label Studio |
| Model Training  | U-Net (TensorFlow, Keras) |
| Backend         | FastAPI, OpenCV, NumPy |
| Frontend        | Leaflet.js, Leaflet Draw, Chart.js |

---

## 📄 Project Report

Full documentation: `Major_Project_Final_Report.pdf`

---

## 📦 Outputs

- 🖼️ Cropped aerial image  
- 🧩 Predicted segmentation mask  
- 📊 Land class area statistics (CSV)  
- 📁 Download ZIP with all results

---

## ✅ Status

✔️ Model trained  
✔️ Web integration complete  
✔️ End-to-end pipeline tested  
✔️ Ready for deployment

---
