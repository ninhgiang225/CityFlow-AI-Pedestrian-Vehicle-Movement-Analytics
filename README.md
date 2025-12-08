# Real-Time Pedestrian & Vehicle Detection and Tracking  
**Course:** CS366 – Computer Vision  
**Author:** Ninh Giang Nguyen  

---

## Overview

Urban intersections with dense pedestrian and vehicle traffic—such as Shibuya Crossing in Tokyo—highlight the complexity of managing transportation safety and efficiency. This project builds a **real-time computer vision system** to detect and track pedestrians and vehicles from video footage and transform raw visual data into **actionable traffic analytics**.

Beyond object detection, the system demonstrates how computer vision outputs can support **pedestrian counting, traffic flow analysis, and adaptive traffic-light logic**, illustrating applications in **smart-city infrastructure**.

---
## 📷 Results & Visualizations
👉 **Watch here:** [![CityFlow AI Demo](https://drive.google.com/file/d/1hpNFu5CamWcfKmLmWS4PMGbeVpIoBnT7/view?usp=sharing)
Preview 

<img width="1451" height="793" alt="image" src="https://github.com/user-attachments/assets/e8aa80f7-f929-4e72-bc2d-c616a8264d66" />

<img width="1438" height="751" alt="image" src="https://github.com/user-attachments/assets/b1885974-ab6f-4c6a-ac8f-6246bb3030dc" />

---

## Project Objectives

- **Real-Time Detection**  
  Detect pedestrians and vehicles from video using state-of-the-art object detection models (YOLOv8).

- **Multi-Object Tracking (MOT)**  
  Assign consistent IDs to moving objects across frames using Deep SORT or BoT-SORT, even under occlusion and crowding.

- **Analytics Layer**  
  Extract higher-level insights, including:
  - Pedestrian counts per time interval  
  - Vehicle vs. pedestrian flow ratios  
  - Density heatmaps of crowded regions  

- **Application Demonstration**  
  Provide a software pipeline that allows users to upload traffic videos and trained detection models to generate analytics and traffic-control suggestions.

---

## Methodology

### Data Collection & Preparation
- Used the **COCO dataset** for initial model evaluation  
- Collected real intersection footage from public YouTube videos  
- Annotated additional frames using **Roboflow** to fine-tune pedestrian and vehicle detection

---

### Object Detection
- Implemented **YOLOv8 (Ultralytics)** for its real-time speed–accuracy balance  
- Fine-tuned pretrained YOLOv8 weights on pedestrian and vehicle classes  
- Models used:
  - `yolov8s.pt` (baseline)
  - `yolov8n.pt` (lightweight fine-tuned model)

---

### Multi-Object Tracking
Integrated YOLO detections with modern MOT algorithms:

- **Deep SORT**  
  Uses Kalman filtering + appearance embeddings for ID consistency  

- **BoT-SORT**  
  Improves association accuracy in dense and crowded scenes  

---

### System Pipeline
Video Input
↓
YOLOv8 Object Detection -> yolov8s.pt
↓
Fine-tune YOLOv8 model -> yolov8n.pt
↓
Multi-Object Tracking (Deep SORT / BoT-SORT)
↓
Visualization (Bounding Boxes + IDs)
↓
Analytics (Counts, Flow Ratios, Density)


---

### Application & Simulation
- Implemented **traffic-light adaptation logic**:
  - If pedestrian count > threshold → extend pedestrian green phase
  - If vehicle density is high → prioritize vehicle flow
- Designed software allowing users to:
  - Upload traffic videos
  - Upload trained detection models
  - Receive analytics and policy suggestions

---

### 6️⃣ Evaluation
- **Detection Metrics**
  - Mean Average Precision (mAP)
  - Precision, Recall, F1-score
  - PR Curves
  - Confusion Matrix  

- **Tracking Metrics**
  - Evaluated using the MOTChallenge toolkit (`motmetrics`)



