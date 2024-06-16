import os
import time
from ultralytics import YOLO
import cv2
import numpy as np

# Directory and file paths
VIDEOS_DIR = 'C:\\Users\\tarhe\\Desktop\\AI Foul Detector'
video_path = os.path.join(VIDEOS_DIR, '4.mov')
video_path_out = '{}_out.mp4'.format(video_path)

# Print paths for debugging
print(f"VIDEOS_DIR: {VIDEOS_DIR}")
print(f"video_path: {video_path}")
print(f"video_path_out: {video_path_out}")

# Check if the video file exists
if not os.path.exists(video_path):
    print(f"Error: Video file {video_path} does not exist.")
    exit()

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file can be opened
if not cap.isOpened():
    print(f"Error: Cannot open video file {video_path}.")
    exit()

# Read the first frame to get video dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Cannot read the first frame of the video.")
    cap.release()
    exit()

# Get video dimensions
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Load the YOLO model
model_path = 'C:\\Users\\tarhe\\Desktop\\AI Foul Detector\\runs\detect\\train9\\weights\\best.pt'
model = YOLO(model_path)  # Load a custom model
class_name_dict = {0: 'shoe'}

# Detection threshold
threshold = 0.5

# Process each frame
while ret:
    # Perform detection
    results = model(frame)[0]

    # Iterate through the detected objects
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        # Draw bounding box and label if the score is above the threshold
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Write the processed frame to the output video
    out.write(frame)
    
    # Read the next frame
    ret, frame = cap.read()

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved as {video_path_out}")
