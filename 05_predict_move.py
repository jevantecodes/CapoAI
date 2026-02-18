from tensorflow.keras.models import load_model
import numpy as np
import mediapipe as mp
import cv2
import os
import json

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def extract_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    landmarks_all = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            frame_landmarks = np.array([[l.x, l.y, l.z] for l in lm]).flatten()
            landmarks_all.append(frame_landmarks)

    cap.release()
    return np.array(landmarks_all)

# Load model + labels
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
SEQUENCE_LENGTH = 50

def predict_move(video_path):
    landmarks = extract_landmarks(video_path)

    # Normalize sequence
    if len(landmarks) < SEQUENCE_LENGTH:
        pad = np.zeros((SEQUENCE_LENGTH - len(landmarks), landmarks.shape[1]))
        landmarks = np.vstack([landmarks, pad])
    else:
        landmarks = landmarks[:SEQUENCE_LENGTH]

    input_data = np.expand_dims(landmarks, axis=0)
    preds = model.predict(input_data)
    move = MOVES[np.argmax(preds)]
    conf = np.max(preds)
    return move, conf

# Example
video = "data/ginga/capoeria_ginga_2.mp4"
pred_move, conf = predict_move(video)
print(f"Predicted Move: {pred_move} ({conf*100:.2f}% confidence)")
