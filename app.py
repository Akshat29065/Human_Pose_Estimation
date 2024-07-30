import cv2
import streamlit as st
import math
from ultralytics import YOLO

# Load the custom YOLOv8 model
model = YOLO("C:\\Users\\asus\\Desktop\\hello\\best.pt")

# Define class names (you need to define this based on your model classes)
classNames = ["Good_posture_standing", "Plank_Pose", "Warrior_Pose", "sitting_frontview", "sitting_sideview", "treepose"]

# Create a Streamlit app
st.title("YOLOv8 Object Detection with Webcam")
st.write("Press 'q' to quit")

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

while True:
    ret, img = cap.read()
    if not ret:
        st.write("Error capturing image from webcam")
        break

    try:
        results = model(img, stream=True)

        # Iterate over results directly without checking length
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                # Ensure class index is within the range of classNames
                if cls < len(classNames):
                    cv2.putText(img, classNames[cls], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                else:
                    st.write(f"Class index {cls} out of range for classNames")

        # Update the Streamlit image placeholder
        frame_placeholder.image(img, channels="BGR")

    except Exception as e:
        st.write(f"Error processing image: {e}")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
