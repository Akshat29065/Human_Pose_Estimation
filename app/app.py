import cv2
import streamlit as st
import math
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

model = YOLO(str(MODEL_PATH))

# Define class names (you need to define this based on your model classes)
class_names = [
    "Good_posture_standing",
    "Plank_Pose",
    "Warrior_Pose",
    "sitting_frontview",
    "sitting_sideview",
    "treepose"
]

# Create a Streamlit app
st.title("Real-Time Human Pose Detection")
st.write("YOLOv8-based human pose detection using a live webcam feed.")

# Create a webcam capture object
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    st.write("Error: Could not open webcam.")
else:
    st.write("Webcam successfully opened.")

# Create a placeholder for the image
frame_placeholder = st.empty()

while cap.isOpened():

    ret, img = cap.read()

    if not ret:
        st.write("Error capturing image from webcam")
        break

    try:
        results = model(img, stream=True)

        # Iterate over results directly without checking length
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                confidence = float(box.conf[0])
                cls = int(box.cls[0])

                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 255),
                    3
                )

                # Ensure class index is within the range of class_names
                if cls < len(class_names):
                    label = (
                        f"{class_names[cls]} "
                        f"{confidence:.2f}"
                    )
                    cv2.putText(
                        img,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2
                    )


        # Update the Streamlit image placeholder
        frame_placeholder.image(
            img,
            channels="BGR"
        )

    except Exception as e:
        st.error(f"Error processing image: {e}")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
