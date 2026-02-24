

# CityFlow AI: Real-Time Pedestrian & Vehicle Detection and Tracking
*Smart-city traffic analytics powered by computer vision*

**Author:** Ninh Giang Nguyen

---

## 🛠 Skills & Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-111F68?style=for-the-badge&logo=yolo&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-1572B6?style=for-the-badge&logoColor=white)
![Deep SORT](https://img.shields.io/badge/Deep%20SORT-6A0DAD?style=for-the-badge&logoColor=white)
![BoT-SORT](https://img.shields.io/badge/BoT--SORT-02A88E?style=for-the-badge&logoColor=white)
![Roboflow](https://img.shields.io/badge/Roboflow-6706CE?style=for-the-badge&logo=roboflow&logoColor=white)
![COCO](https://img.shields.io/badge/COCO%20Dataset-F7931E?style=for-the-badge&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logoColor=white)

---

## Overview

Urban intersections like Shibuya Crossing in Tokyo highlight the complexity of managing transportation safety and efficiency at scale. This project builds a **real-time computer vision system** that detects and tracks pedestrians and vehicles from video footage, transforming raw visual data into **actionable traffic analytics**.

The system supports **pedestrian counting, traffic flow analysis, and adaptive traffic-light logic** — demonstrating practical applications in **smart-city infrastructure**.

---

## Quick Start

1. **Clone the repository and install dependencies:**
   ```bash
   git clone https://github.com/your-username/cityflow-ai.git
   cd cityflow-ai
   pip install -r requirements.txt
   ```

2. **Run the detection pipeline on a video:**
   ```bash
   python main.py --source your_video.mp4 --model yolov8s.pt --tracker botsort
   ```

3. **Explore the full analysis:**  
   Open `main.ipynb` in Jupyter to walk through detection, tracking, and analytics step by step.

---
## Results & Visualizations

Check main.ipynb or 👉 **Watch here:** [CityFlow AI Demo](https://drive.google.com/file/d/1hpNFu5CamWcfKmLmWS4PMGbeVpIoBnT7/view?usp=sharing)

Preview 

<img width="500" height="300" padding-bottom="100" alt="image" src="https://github.com/user-attachments/assets/e8aa80f7-f929-4e72-bc2d-c616a8264d66" /><img width="500" height="350" alt="image" src="https://github.com/user-attachments/assets/b1885974-ab6f-4c6a-ac8f-6246bb3030dc" />

---

## Project Objectives

- **Real-Time Detection** — Detect pedestrians and vehicles from video using YOLOv8.
- **Multi-Object Tracking (MOT)** — Assign consistent IDs across frames using Deep SORT or BoT-SORT, even under occlusion and crowding.
- **Analytics Layer** — Generate pedestrian counts per time interval, vehicle-to-pedestrian flow ratios, and density heatmaps of crowded regions.
- **Application Pipeline** — Allow users to upload traffic videos and trained models to produce analytics and traffic-control suggestions.

---

## Methodology

### Data Collection & Preparation
- Used the **COCO dataset** for initial model evaluation
- Collected real intersection footage from public YouTube videos
- Annotated additional frames using **Roboflow** to fine-tune detection on pedestrians and vehicles

### Object Detection
- Implemented **YOLOv8 (Ultralytics)** for its real-time speed–accuracy balance
- Fine-tuned pretrained weights on pedestrian and vehicle classes
- Models: `yolov8s.pt` (baseline) and `yolov8n.pt` (lightweight fine-tuned)

### Multi-Object Tracking
| Algorithm | Strengths |
|---|---|
| **Deep SORT** | Kalman filtering + appearance embeddings for ID consistency |
| **BoT-SORT** | Improved association accuracy in dense, crowded scenes |

### System Pipeline

```
Video Input
    ↓
YOLOv8 Object Detection  →  yolov8s.pt (baseline)
    ↓
Fine-Tuned YOLOv8        →  yolov8n.pt
    ↓
Multi-Object Tracking (Deep SORT / BoT-SORT)
    ↓
Visualization (Bounding Boxes + Persistent IDs)
    ↓
Analytics (Counts, Flow Ratios, Density Heatmaps)
```

---

## Evaluation

| Metric | Score |
|---|---|
| Mean Average Precision (mAP) | 79% |
| Additional Metrics | F1-Score, Precision, Recall, PR Curves, Confusion Matrix |

---

## Future Work

- Expand to night-time and adverse weather footage
- Deploy as a real-time web dashboard
- Integrate with live traffic camera feeds
- Explore transformer-based detectors (e.g., RT-DETR)

---

*Built to explore the intersection of computer vision and smart-city infrastructure.*

