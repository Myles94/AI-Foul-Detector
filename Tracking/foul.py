import os
import cv2
from ultralytics import YOLO

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
model_path = 'C:\\Users\\tarhe\\Desktop\\AI Foul Detector\\runs\\detect\\train9\\weights\\best.pt'
model = YOLO(model_path)  # Load a custom model

# Detection threshold
threshold = 0.5

# Define x-coordinate threshold for foul detection
X_COORD_THRESHOLD = 1000  # Example threshold, adjust according to your setup

# Flag to track if foul is detected
foul_detected = False

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

        # Check if the shoe is past the x-coordinate threshold
        x_center = (x1 + x2) / 2
        if x_center > X_COORD_THRESHOLD:
            foul_detected = True
            break  # Exit the loop once foul is detected

    # Display "FAIR" message if no foul detected
    if not foul_detected:
        cv2.putText(frame, 'FAIR', (int(W / 4) + 100, int(H / 2) + -300),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 10, cv2.LINE_AA)
    else:
        # Make the region behind the text red
        frame[int(H / 2 - 100):int(H / 2 + 100), int(W / 4 - 200):int(W / 4 + 200)] = (0, 0, 255)

        # Add "FOUL!" text with a simpler font
        cv2.putText(frame, 'FOUL!', (int(W / 4) + 100, int(H / 2) + -300),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 10, cv2.LINE_AA)

    # Write the processed frame to the output video
    out.write(frame)

    # Display the frame with detection
    cv2.imshow('Frame', frame)

    # Read the next frame
    ret, frame = cap.read()

    # Exit loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved as {video_path_out}")

