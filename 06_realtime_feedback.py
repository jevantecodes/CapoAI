import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import os
import json

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

SEQUENCE_LENGTH = 50

MODEL_CANDIDATES = [
    "models/capoeira_model_best.keras",
    "models/capoeira_model_best.h5",
    "models/capoeira_model_final.keras",
]
MODEL_PATH = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), MODEL_CANDIDATES[0])
model = load_model(MODEL_PATH)

LABEL_MAP_PATH = "dataset/label_map.json"
if os.path.exists(LABEL_MAP_PATH):
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        MOVES = json.load(f)
else:
    MOVES = [f"class_{i}" for i in range(model.output_shape[-1])]

sequence = deque(maxlen=SEQUENCE_LENGTH)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = results.pose_landmarks.landmark
        frame_landmarks = np.array([[l.x, l.y, l.z] for l in lm]).flatten()
        sequence.append(frame_landmarks)

        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(sequence, axis=0)
            preds = model.predict(input_data)
            move = MOVES[np.argmax(preds)]
            conf = np.max(preds)
            cv2.putText(frame, f"{move} ({conf*100:.1f}%)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('Capoeira AI', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
