# Real-Time PPE Compliance Monitoring Using Detection, Pose Estimation, and Tracking

## Overview

This repository contains the implementation and experimental evaluation of a real-time Personal Protective Equipment (PPE) compliance monitoring system for construction environments. The system integrates object detection, human pose estimation, and multi-object tracking to determine whether workers are properly wearing required PPE, including helmets, safety vests, gloves, and boots.

The project was developed as part of an undergraduate Computer Science thesis and evaluates multiple combinations of detection, tracking, and pose estimation models to identify the most effective configuration for real-time deployment.

---

## Features

- PPE Detection (Helmet, Vest, Gloves, Boots)
- Human Pose Estimation
- Multi-Object Tracking
- Pose-Guided PPE Association
- Compliance Verification Logic
- Real-Time Video Processing
- Performance Evaluation and Logging
- Automated Compliance Classification

---

## Repository Structure

```text
├── Custom_dataset.py
├── class_optimizer.py
├── img_counter.py
├── detectmodel_training.py
├── Pipeline_inference.py
└── README.md
```

### Custom_dataset.py
Dataset integration and annotation remapping utility used to merge PPE datasets and standardize class labels.

### class_optimizer.py
Dataset preprocessing script used to remove unnecessary classes and reorganize annotation indices.

### img_counter.py
Utility script for analyzing dataset class distributions and counting object instances.

### detectmodel_training.py
YOLO11 training script used for PPE detector development and model optimization.

### Pipeline_inference.py
Main inference pipeline integrating PPE detection, pose estimation, worker tracking, and compliance verification.

---

## Methodology

The proposed system processes video frames through the following stages:

1. PPE Detection using YOLO11s
2. Human Pose Estimation using YOLOv8 Pose
3. Worker Tracking using Norfair
4. Pose-Guided PPE Association
5. Compliance Verification and Classification

Detected PPE objects are associated with worker body keypoints to determine whether equipment is properly worn according to expected anatomical locations.

---

## Dataset Preparation Workflow

1. Merge and standardize datasets using `Custom_dataset.py`
2. Optimize class annotations using `class_optimizer.py`
3. Analyze class distributions using `img_counter.py`
4. Train the detector using `detectmodel_training.py`
5. Perform inference and evaluation using `Pipeline_inference.py`

---

## Requirements

Install the required dependencies:

```bash
pip install ultralytics
pip install opencv-python
pip install norfair
pip install numpy
```

---

## Running the Project

Train the PPE detector:

```bash
python detectmodel_training.py
```

Run the realtime monitoring pipeline:

```bash
python Pipeline_inference.py
```

---

## Evaluation Metrics

The system evaluates performance using:

- Precision
- Recall
- F1-Score
- Compliance Accuracy
- Non-Compliance Rate
- Average FPS
- Average Processing Time (ms)
- Tracking Stability

---

## Research Objectives

- Detect workers, PPE items, and body keypoints from construction site video frames.
- Track workers across consecutive frames for temporal consistency.
- Verify PPE compliance using pose-guided anatomical validation.
- Compare detection, tracking, and pose estimation model combinations.
- Identify an effective realtime PPE compliance monitoring pipeline.

---

## Author

**Ken Lugpatan**  
Bachelor of Science in Computer Science  
Caraga State University

Research Interests:
- Artificial Intelligence
- Computer Vision
- Object Detection
- Human Pose Estimation
- Multi-Object Tracking
- Real-Time Safety Monitoring Systems

---

## License

This project was developed for academic and research purposes. Proper citation is encouraged when referencing this work.
