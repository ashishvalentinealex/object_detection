import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize TFLite model
base_options = python.BaseOptions(model_asset_path='/home/user/Documents/Object_detection_raspi/tflite-mediapipe/exported_model/best.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Get video frame width & height
frame_width = int(cap.get(3))  # 640 default
frame_height = int(cap.get(4))  # 480 default
center_y = frame_height // 2  # Midpoint in vertical direction

# Tracking and counting setup
totalCount_up = []
totalCount_down = []
previous_centers = {}  # Store previous positions for tracking

# Define horizontal counting line across full frame width
limits_up = [0, center_y, frame_width, center_y]
limits_down = [0, center_y + 20, frame_width, center_y + 20]

TEXT_COLOR = (255, 0, 0)  # Red
rect_color = (255, 0, 255)  # Magenta

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert OpenCV BGR frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Perform object detection
    detection_result = detector.detect(mp_image)

    detected_objects = {}

    for i, detection in enumerate(detection_result.detections):
        # Get bounding box
        bbox = detection.bounding_box
        x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
        w, h = int(bbox.width), int(bbox.height)
        x2, y2 = x1 + w, y1 + h
        conf = round(detection.categories[0].score, 2)
        
        # Calculate object center
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        detected_objects[i] = (cx, cy)

        # Draw rectangle around detected object
        cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 3)

        # Get category name
        category_name = detection.categories[0].category_name
        result_text = f"{category_name} ({conf})"

        # Put text on the image
        text_location = (x1, y1 - 10)
        cv2.putText(frame, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, 1, TEXT_COLOR, 1)

        # Track movement and count objects crossing the center line
        if i in previous_centers:
            prev_cx, prev_cy = previous_centers[i]

            if prev_cy < center_y and cy >= center_y:
                totalCount_down.append(1)
            elif prev_cy > center_y and cy <= center_y:
                totalCount_up.append(1)

    previous_centers = detected_objects  # Update previous positions

    # Draw counting lines (full width)
    cv2.line(frame, (limits_up[0], limits_up[1]), (limits_up[2], limits_up[3]), (0, 0, 255), 4)
    cv2.line(frame, (limits_down[0], limits_down[1]), (limits_down[2], limits_down[3]), (255, 0, 255), 4)

    # Display count
    cv2.putText(frame, f"Up: {len(totalCount_up)}", (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
    cv2.putText(frame, f"Down: {len(totalCount_down)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
