import cv2
import numpy as np
from ultralytics import YOLO 
from deep_sort_realtime.deepsort_tracker import DeepSort
from deepface import DeepFace
import mediapipe as mp

# Load YOLOv8 model
yolo_model = YOLO("yolov8s.pt")
#result
# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

# Initialize MediaPipe Pose Estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open webcam video stream
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 object detection
    results = yolo_model(frame, conf=0.5)  # Increase confidence threshold

    detections = []
    objects_detected = set()  # Store unique detected objects

    for result in results:
        for obj in result.boxes:
            x1, y1, x2, y2 = map(int, obj.xyxy[0])  # Get bounding box
            conf = float(obj.conf[0])  # Confidence score
            cls = int(obj.cls[0])  # Class index
            class_name = yolo_model.names[cls] 

            objects_detected.add(class_name)  # Store detected objects

            # Only track people for gender detection
            if class_name == "person" and conf > 0.3:
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, None))

    # Track detected objects
    tracks = tracker.update_tracks(detections, frame=frame)

    total_crowd = 0
    male_count = 0
    female_count = 0

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        total_crowd += 1

        # Face detection for gender classification
        face_crop = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
        gender = "Unknown"
        age = "Unknown"
        color = (0, 255, 0)  # Default: Green

        if face_crop.size > 0:
            try:
                if face_crop.shape[0] >= 20 and face_crop.shape[1] >= 20:
                    face_analysis = DeepFace.analyze(face_crop, actions=['age', 'gender'], enforce_detection=False)
                    gender = face_analysis[0]['dominant_gender']
                    age = int(face_analysis[0]['age'])

                    if gender == "Man":
                        male_count += 1
                        color = (0, 0, 255)  # Red for Male
                    else:
                        female_count += 1
                        color = (255, 0, 0)  # Blue for Female
            except:
                pass

        # Draw bounding box and labels
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID: {track_id} | {gender} | Age: {age}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)
    if pose_results.pose_landmarks:
        for landmark in pose_results.pose_landmarks.landmark:
            h, w, _ = frame.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)  # Yellow for pose tracking

    # Display total counts and detected objects
    object_text = " | ".join(objects_detected) if objects_detected else "None"
    cv2.putText(frame, f"Total: {total_crowd} | Male: {male_count} | Female: {female_count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Objects Detected: {object_text}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Show the frame
    cv2.imshow("Enhanced Real-Time Detection", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
