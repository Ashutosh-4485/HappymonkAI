from ultralytics import YOLO
import cv2
import time
from collections import defaultdict
#Downloading and loading the YOLO version 8 nano model
model=YOLO('yolov8s.pt')


# This is the list of objects that YOLO has been trained to detect. We will use this to find out the labels in the detection.
classNames=["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
             "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
             "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
             "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
             "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
             "hot dog", "pizza", "donut", "cake", "chain", "sofa", "pottedplant", "bed", "diningtable", "toilet",
             "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
             "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

cy1=322
cy2=368
offset=6
vh_down={}
counter=[]
vh_up={}
counter1=[]
vehicle_id=[]


#Funtion to rescale the video
def rescale(img,scale=0.5):
    width=int(img.shape[1]*scale)
    height = int(img.shape[0] * scale)
    dimension=(width,height)
    return cv2.resize(img,dimension,interpolation=cv2.INTER_AREA)

prev_y_coords = defaultdict(int)

# Function to calculate speed
def calculate_speed(cx, cy, x1, y1, x2, y2):
    if cy1 < (cy + offset) and cy1 > (cy - offset):
        vh_down[id] = time.time()
    if id in vh_down:
        if cy2 < (cy + offset) and cy2 > (cy - offset):
            elapsed_time = time.time() - vh_down[id]
            if counter.count(id) == 0:
                counter.append(id)
                distance = 10  # meters
                a_speed_ms = distance / elapsed_time
                a_speed_kh = a_speed_ms * 3.6
                cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(img, str(id), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(img, str(int(a_speed_kh)) + ' Km/h', (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (0, 255, 255), 2)
# Funtion to detect cell phone and count the number of people in a frame.

def vehicle_detection(img,start_time1=0,start_time2=0,time1=0,time2=0):
    results = model(img, stream=True)
    vehicle_count = 0

    #iterating through the result
    for r in results:
        boxes = r.boxes
        #iterating through every detection one by one
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            # print(x1,y1,x2,y2)
            cls = int(box.cls[0])  # changing the class number from tensor to integer
            label = classNames[cls]  # retrieving the class name
            conf_score = int(box.conf[0] * 100)

            # Checking the labels if a person or cell phone has been detected,
            # if a car is detected then counting the number of people

            if label == 'car':
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                vehicle_count += 1

                # Calculate and display speed
                if prev_y_coords.get(id, 0) != 0:
                    calculate_speed(cx, cy, x1, y1, x2, y2)

                # Store current y-coordinate for the next frame
                prev_y_coords[id] = cy

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


        cv2.line(img, (10, cy1), (1200, cy1), (255, 255, 255), 1)
        cv2.putText(img, ('L1'), (50, 320), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        cv2.line(img, (0, cy2+50), (1200, cy2+50), (255, 255, 255), 1)
        cv2.putText(img, ('L2'), (52, 367), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(img, f"Vehicle count {vehicle_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        #if no car is detected, then show "Face not detected"
        if vehicle_count == 0:
            cv2.putText(img,"No car detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return img



cap=cv2.VideoCapture("1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frame Per Second is: {fps}")
while True:
    success,img=cap.read()
    # img = rescale(img)    # Rescaling in case a video file has been given other than using web cam
    #calling the "vehicle_count" function


    img_after_detection=vehicle_detection(img)

    cv2.imshow("Image", img_after_detection)

    #press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break