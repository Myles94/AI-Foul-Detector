# AI Foul Detection System
## Overview
The AI Foul Detection System is a video analysis tool designed to detect fouls in track and field events using artificial intelligence and computer vision techniques. This project focuses on real-time detection of athlete's shoes crossing specific boundaries during events such as discus throws or shot puts. The system provides visual feedback on the video feed to indicate potential fouls.

## Project Components
1. Object Detection Model
Framework: YOLO v8 (You Only Look Once)
Description: Trained a custom YOLO model using YOLOv5 and Ultralytics framework to detect athlete's shoes in real-time video frames.
2. Video Processing
Library: OpenCV (Open Source Computer Vision Library)
Description: Used OpenCV for video input/output, frame processing, and visualization of detection results.
3. Foul Detection Logic
Implementation: Python programming language
Description: Implemented logic to determine when an athlete's shoe crosses a predefined boundary (e.g., circle boundary in discus throw) and marks it as a foul.
4. Visual Feedback
Features: Overlay messages and visual cues
## Description: Provided visual feedback on the video stream to indicate fouls using text overlays and colored bounding boxes around detected shoes.
## How to Use
Requirements
Python 
YOLOv8
OpenCV
Ultralytics
Installation
Clone the repository: git clone https://github.com/Myles94/AI-Foul-Detection-System.git
Install dependencies: pip install -r requirements.txt
Usage
Navigate to the project directory: cd AI-Foul-Detection-System
Run the detection script: python foul_detection.py --video <path_to_video>
Customization
Adjust detection thresholds and boundary coordinates in foul_detection.py to suit different track and field event setups.
