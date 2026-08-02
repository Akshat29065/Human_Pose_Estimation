# Real-Time Human Pose Detection using YOLOv8

A computer vision application for detecting and classifying human poses in real time using a custom-trained **YOLOv8 Nano** model. The system processes live webcam frames using **OpenCV** and provides real-time predictions through an interactive **Streamlit** interface.

## Overview

This project explores real-time human pose detection using deep learning and computer vision.

A custom dataset containing different human postures was prepared and managed using **Roboflow** and exported in YOLOv8 format. A YOLOv8 Nano model was then trained on the dataset and evaluated using standard object-detection metrics.

The trained model is integrated into a Streamlit application that captures frames from a webcam, performs inference, and displays the detected pose along with its confidence score.

## Key Features

* Real-time human pose detection using a webcam
* Custom-trained YOLOv8 Nano model
* Custom annotated dataset prepared using Roboflow
* Bounding-box-based pose detection and classification
* Confidence score for each detected pose
* Streamlit-based interactive interface
* OpenCV-based real-time video processing
* Separate training and deployment components

## Pose Classes

The model is trained to recognize the following pose categories:

| # | Pose                  |
| - | --------------------- |
| 1 | Good Posture Standing |
| 2 | Plank Pose            |
| 3 | Warrior Pose          |
| 4 | Sitting Front View    |
| 5 | Sitting Side View     |
| 6 | Tree Pose             |

## Project Workflow

```text
Custom Image Dataset
        │
        ▼
Dataset Annotation & Preparation
        │
        ▼
Roboflow
        │
        ▼
YOLOv8 Dataset Format
        │
        ▼
YOLOv8 Nano Training
        │
        ▼
Model Evaluation
        │
        ▼
best.pt
        │
        ▼
Streamlit Application
        │
        ▼
Webcam Input
        │
        ▼
Real-Time Pose Detection
```

## Project Structure

```text
Human_Pose_Estimation/
│
├── app/
│   └── app.py
│
├── models/
│   └── best.pt
│
├── notebooks/
│   └── model_training.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

### Directory Description

* **`app/`** — Contains the Streamlit application used for real-time webcam inference.
* **`models/`** — Contains the trained YOLOv8 model weights.
* **`notebooks/`** — Contains the model training and evaluation workflow.
* **`requirements.txt`** — Lists the Python dependencies required for the project.

## Model Training

The model training workflow is available in:

```text
notebooks/model_training.ipynb
```

The dataset was prepared using **Roboflow** and exported in YOLOv8 format.

The YOLOv8 Nano architecture was trained using the following configuration:

| Parameter          | Value              |
| ------------------ | ------------------ |
| Model              | YOLOv8 Nano        |
| Epochs             | 80                 |
| Image Size         | 400                |
| Framework          | Ultralytics YOLOv8 |
| Dataset Format     | YOLOv8             |
| Dataset Management | Roboflow           |

The best model weights generated during training are stored as:

```text
models/best.pt
```

## Technologies Used

| Technology           | Purpose                               |
| -------------------- | ------------------------------------- |
| Python               | Core programming language             |
| YOLOv8 / Ultralytics | Model training and inference          |
| OpenCV               | Webcam capture and image processing   |
| Streamlit            | Interactive application interface     |
| Roboflow             | Dataset preparation and management    |
| NumPy                | Numerical and image-array operations  |
| Matplotlib           | Training and evaluation visualization |
| Jupyter Notebook     | Model experimentation and training    |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Akshat29065/Human_Pose_Estimation.git
```

Move into the project directory:

```bash
cd Human_Pose_Estimation
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

Make sure the trained model is available at:

```text
models/best.pt
```

Then run the Streamlit application from the project root:

```bash
streamlit run app/app.py
```

Streamlit will start the application locally, typically at:

```text
http://localhost:8501
```

Allow access to the webcam if required by your system.

The application will process webcam frames using the trained YOLOv8 model and display the detected pose and confidence score in real time.

## Requirements

The project uses the following primary dependencies:

```text
ultralytics
streamlit
opencv-python
numpy
pillow
matplotlib
python-dotenv
roboflow
```

The complete dependency configuration is available in `requirements.txt`.

## Dataset

The dataset consists of images representing multiple human poses and postures.

Dataset preparation and management were performed using **Roboflow**, with annotations exported in YOLOv8-compatible format for model training.

The dataset itself is not included in this repository.

## Model Inference

During real-time inference:

1. OpenCV captures frames from the connected webcam.
2. Each frame is passed to the trained YOLOv8 model.
3. YOLOv8 detects and classifies the human pose.
4. The predicted bounding box is rendered on the frame.
5. The corresponding pose class and confidence score are displayed.
6. The processed frame is displayed through the Streamlit interface.

## Limitations

* Detection performance depends on lighting, camera quality, background conditions, and pose visibility.
* The model is limited to the pose classes represented in the training dataset.
* Performance may decrease for poses or viewing angles that differ significantly from the training data.
* Webcam performance can vary depending on the operating system and connected camera device.

## Future Improvements

Potential improvements include:

* Expanding the dataset with more pose categories and viewing angles
* Increasing variation in lighting, backgrounds, and subjects
* Evaluating larger YOLO architectures
* Improving real-time inference performance
* Adding browser-based webcam support for cloud deployment
* Extending the system toward keypoint-based human pose estimation

## Author

**Akshat Agarwal**

B.Tech Computer Science & Engineering
Specialization in Data Science
UPES, Dehradun

GitHub: `Akshat29065`


