from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')
classNames=["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
             "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
             "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
             "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
             "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
             "hot dog", "pizza", "donut", "cake", "chain", "sofa", "pottedplant", "bed", "diningtable", "toilet",
             "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
             "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Open the video file
video_path = "1.mp4"
cap = cv2.VideoCapture(video_path)
label=[]

# Store the track history
track_history = defaultdict(lambda: [])
counter = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    counter+=1
    if counter%3!=0:
        continue

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        vehicle_count = 0
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        label.append(results[0].names.values())
        print(results)
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        for i in label:
            print("****************************************************************")
            for j in i:
                if j=="car" or j=="truck":
                    vehicle_count+=1
                    # Plot the tracksq
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))  # x, y center point
                        if len(track) > 30:  # retain 90 tracks for 90 frames
                            track.pop(0)

                        # Draw the tracking lines

                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Display the annotated frame

        cv2.putText(annotated_frame, f"Total Vehicle is: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
