import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from deepface import DeepFace
import mediapipe as mp

yolo_model = YOLO("yolov8n.pt")  

tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

total_crowd = 0
male_count = 0
female_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            if cls == 0 and conf > 0.3:
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, None))

    tracks = tracker.update_tracks(detections, frame=frame)

    total_crowd = 0
    male_count = 0
    female_count = 0

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        total_crowd += 1

        face_crop = frame[y1:y2, x1:x2]
        gender = "Unknown"

        if face_crop.size > 0:
            try:
                face_analysis = DeepFace.analyze(face_crop, actions=['gender'], enforce_detection=False)
                gender = face_analysis[0]['dominant_gender']

                if gender == "Man":
                    male_count += 1
                    color = (0, 0, 255) 
                else:
                    female_count += 1
                    color = (255, 0, 0)  
            except:
                color = (0, 255, 0)  

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID: {track_id} | {gender}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, f"Total: {total_crowd} | Male: {male_count} | Female: {female_count}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Real-Time Crowd Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
