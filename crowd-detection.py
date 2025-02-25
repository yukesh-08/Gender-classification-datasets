import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from deepface import DeepFace
import xgboost as xgb
import pickle

yolo_model = YOLO("yolov8n.pt")  

tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

xgb_model_path = r"D:\YUKESH\GI-project\xgboost_gender_model_b3.pkl"  
try:
    with open(xgb_model_path, "rb") as file:
        xgb_model = pickle.load(file)
    print("✅ XGBoost model loaded successfully!")
except Exception as e:
    print(f"⚠️ XGBoost model load error: {e}")
    xgb_model = None

cap = cv2.VideoCapture(0)

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
    total_male = 0
    total_female = 0

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        face_crop = frame[y1:y2, x1:x2]
        gender = "Unknown"

        if face_crop.size > 0:
            try:
                face_analysis = DeepFace.analyze(face_crop, actions=['gender'], enforce_detection=False)
                print("🔍 DeepFace Output:", face_analysis)  
                
                gender = face_analysis[0]['dominant_gender']
                print(f"✅ Gender Detected: {gender}")

            except Exception as e:
                print(f"⚠️ DeepFace Error: {e}")

        if gender.lower() == "man":
            color = (0, 0, 255)  
            total_male += 1
        elif gender.lower() == "female":
            color = (255, 0, 0) 
            total_female += 1
        else:
            color = (0, 255, 0)  # Green for unknown

        total_crowd += 1

        # Draw Bounding Box & Info
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID: {track_id} | {gender}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display count information
    cv2.putText(frame, f"Total Crowd: {total_crowd}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Male: {total_male} | Female: {total_female}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("Real-Time Crowd Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
