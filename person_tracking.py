import torch
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# Load YOLOv5 model for person detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=30)

# Function to process each frame and detect persons
def process_frame(frame):
    results = model(frame)  # Perform person detection using YOLOv5
    detections = results.xyxy[0].cpu().numpy()  # Get the detection results
    dets = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0:  # Class 0 corresponds to 'person' in COCO dataset
            dets.append([x1, y1, x2, y2, conf])

    tracks = tracker.update_tracks(np.array(dets), frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltrb()
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    return frame

# Video processing function
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)  # Process each frame
        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    video_path = 'input_video.mp4'
    output_path = 'output_video.mp4'
    process_video(video_path, output_path)
