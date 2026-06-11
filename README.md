# Human Pose Estimation using YOLOv8

## Overview

This project implements a **Real-Time Human Pose Estimation System** using **YOLOv8 Nano**. The model is trained on a custom dataset collected from multiple online sources, annotated manually, augmented using Roboflow, and deployed as an interactive web application using Streamlit.

The trained model achieves an **mAP@0.5 of 85%**, enabling accurate real-time pose detection and keypoint estimation.

---

## Features

* Real-time Human Pose Estimation
* Custom Dataset Collection
* Manual Data Annotation
* Data Augmentation using Roboflow
* YOLOv8 Nano Fine-Tuning
* Streamlit-based Web Deployment
* High Accuracy (mAP@0.5 = 85%)

---

## Project Structure

```text
Human-Pose-Estimation/
│
├── Model_Training/
│   ├── model_training.ipynb
│   └── best.pt
│
├── Deployment/
│   ├── app.py
│   └── best.pt
│
└── README.md
```

### Folder Description

#### Model_Training/

Contains all files related to model training.

* **model_training.ipynb** – Jupyter Notebook used for data preprocessing, training, validation, and evaluation.
* **best.pt** – Best-performing YOLOv8 model weights obtained after training.

#### Deployment/

Contains files required for application deployment.

* **app.py** – Streamlit application for real-time pose estimation.
* **best.pt** – Trained YOLOv8 model used for inference.

---

## Dataset Preparation

### Data Collection

The dataset was collected from various publicly available online sources containing images of humans in different poses and environments.

### Annotation

Images were manually annotated to ensure high-quality pose labels and keypoint information.

### Data Augmentation

Data augmentation was performed using **Roboflow** to improve model robustness and generalization. Augmentation techniques included:

* Rotation
* Flipping
* Scaling
* Brightness Adjustment
* Cropping
* Other Roboflow-supported transformations

---

## Model Training

The project uses **YOLOv8 Nano (YOLOv8n-Pose)** as the base model.

### Training Workflow

1. Data Collection
2. Manual Annotation
3. Data Augmentation using Roboflow
4. Dataset Export
5. YOLOv8 Fine-Tuning
6. Model Evaluation
7. Deployment

### Performance

| Metric  | Score |
| ------- | ----- |
| mAP@0.5 | 85%   |

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/Human-Pose-Estimation.git

cd Human-Pose-Estimation
```

### Install Dependencies

```bash
pip install ultralytics
pip install streamlit
pip install opencv-python
pip install numpy
```

Or

```bash
pip install -r requirements.txt
```

---

## Running the Application

Navigate to the Deployment folder:

```bash
cd Deployment
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser and allow image/video-based pose estimation using the trained YOLOv8 model.

---

## Technologies Used

* Python
* YOLOv8 (Ultralytics)
* OpenCV
* Roboflow
* Streamlit
* NumPy
* Jupyter Notebook

---

## Future Improvements

* Multi-person pose tracking
* Video stream optimization
* Mobile deployment
* Edge-device inference
* Higher-capacity YOLO models for improved accuracy

---

## Results

The trained YOLOv8 Nano model successfully performs real-time human pose estimation with an **85% mAP@0.5**, demonstrating strong performance on custom-collected and augmented datasets.

---

## Author

**Akshat Agarwal**

B.Tech Computer Science Engineering (Data Science)

Specialization in Machine Learning, Computer Vision, and AI Applications.
