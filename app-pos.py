import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best-pos.pt")

model = load_model()

st.title("Coconut Position Detection")
st.write("Live webcam feed with YOLO object detection.")

# Start webcam
run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break
            
        # Run YOLO detection
        results = model(frame)
        
        # Draw results on the frame
        annotated_frame = results[0].plot()
        
        # Convert BGR to RGB for Streamlit
        FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

    cap.release()
