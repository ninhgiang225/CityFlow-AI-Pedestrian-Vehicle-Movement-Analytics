from ultralytics import YOLO
import cv2


model = YOLO('yolov8s.pt')

## Change your input video here 
video_path = "vietnam_street_2.mp4"
# video_path = "data/videos/shibuya_crossing.mp4"
# video_path = "data/videos/shibuya_crossing_speed_down.mp4"



cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    results = model(frame)

    annotated_frame = results[0].plot()
    cv2.imshow("Yolo8s detection", annotated_frame)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# or run yolo track model=yolov8s.pt tracker=bytetrack.yaml source=vietnam_street_2.mp4  in command line