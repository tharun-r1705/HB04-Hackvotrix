import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import math

# Load YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("best-color.pt")

model = load_model()

def classify_coconut_size(xmin, ymin, xmax, ymax, frame_width, frame_height):
    # Calculate bounding box dimensions
    width = xmax - xmin
    height = ymax - ymin
    
    # Estimate coconut area (modeling as an ellipse)
    coconut_area = (math.pi / 4) * width * height
    
    # Compute area ratio relative to the frame
    frame_area = frame_width * frame_height
    ratio = coconut_area / frame_area
    
    # Classify based on calibrated thresholds
    if ratio < 0.08:  # Threshold for Small (~1 kg)
        return "Small (~200 - 300 kg)"
    elif ratio < 0.20:  # Threshold for Medium (~1.5 kg)
        return "Medium (~350 - 500 kg)"
    else:  # Threshold for Large (~2+ kg)
        return "Large (~500 - 850 kg)"

st.title("YOLOv8 Coconut Detection")
st.write("Upload an image or use your webcam to detect coconuts and classify their size.")

# Sidebar for input choice
option = st.sidebar.selectbox("Choose input type", ["Upload Image", "Webcam"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        # Run detection
        results = model.predict(source=img_array, conf=0.25)
        frame_h, frame_w = img_array.shape[:2]

        annotated_image = results[0].plot()

        # Add coconut size labels
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            for box in boxes:
                xmin, ymin, xmax, ymax = map(int, box[:4])
                size_label = classify_coconut_size(xmin, ymin, xmax, ymax, frame_w, frame_h)
                cv2.putText(
                    annotated_image, size_label, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

        st.image(annotated_image, caption="Detected Image", use_container_width=True)

elif option == "Webcam":
    st.write("Press 'Start' to use webcam. Press 'Stop' to end detection.")
    run_webcam = st.checkbox("Start Webcam Detection")

    stframe = st.empty()
    cap = None

    if run_webcam:
        cap = cv2.VideoCapture(0)

        while run_webcam and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to access webcam")
                break

            results = model.predict(frame, conf=0.25)
            annotated_frame = results[0].plot()
            frame_h, frame_w = frame.shape[:2]

            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    xmin, ymin, xmax, ymax = map(int, box[:4])
                    size_label = classify_coconut_size(xmin, ymin, xmax, ymax, frame_w, frame_h)
                    cv2.putText(
                        annotated_frame, size_label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )

            stframe.image(annotated_frame, channels="BGR")

        cap.release()
        cv2.destroyAllWindows()
