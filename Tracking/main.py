import os 
import cv2
from ultralytics import YOLO 
import random
from tracker import Tracker

VIDEOS_DIR = 'C:\\Users\\tarhe\\Desktop\\AI Foul Detector'
video_path = os.path.join(VIDEOS_DIR, '2.mov')

cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()

model_path = 'C:\\Users\\tarhe\\Desktop\\AI Foul Detector\\runs\detect\\train9\\weights\\best.pt'
model = YOLO(model_path)  # Load a custom model
class_name_dict = {0: 'shoe'}

tracker = Tracker()
colors = [(random.randint(0,255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]  # Use `_` for unused loop variable
while ret:
    results = model(frame)
    
    for result in results:
        detections = []
        for r in result.xyxy[0].tolist():  # Use result.xyxy to access the bounding box data
            x1, y1, x2, y2, _, class_id = map(int, r)  # Unpack the bounding box data
            detections.append([x1, y1, x2, y2, 1.0])  # Assuming all scores are 1.0 since YOLO doesn't provide scores in Ultralytics YOLOv5
        
        tracker.update(frame, detections)
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = map(int, bbox)  # Convert bbox to integer values
            track_id = track.track_id
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[track_id % len(colors)], 3)  # Draw rectangle
            cv2.putText(frame, f'Track {track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[track_id % len(colors)], 2)  # Add track ID label
    
    cv2.imshow('frame', frame)
    cv2.waitKey(25)

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()

