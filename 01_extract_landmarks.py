import mediapipe as mp
import cv2
import numpy as np
import os

# --- Initialize MediaPipe Pose ---
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


DATA_DIR = "data/mini_au"
OUTPUT_DIR = "processed_landmarks/mini_au"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🔁 Loop through each video file in data/ginga
for file_name in os.listdir(DATA_DIR):
    if not file_name.lower().endswith(('.mp4', '.mov', '.avi')):
        continue

    video_path = os.path.join(DATA_DIR, file_name)
    base_name = os.path.splitext(file_name)[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}.npy")

    # Skip if already processed
    if os.path.exists(output_path):
        print(f"⏭️  Skipping already processed: {file_name}")
        continue

    print(f"🎥 Extracting from: {video_path}")

    cap_test = cv2.VideoCapture(video_path)
    if not cap_test.isOpened():
        print(f"❌ ERROR: Cannot open video: {video_path}")
        cap_test.release()
        continue
    cap_test.release()

    landmarks = extract_landmarks(video_path)

    if landmarks.size == 0:
        print(f"⚠️  No landmarks detected for {file_name}")
        continue

    np.save(output_path, landmarks)
    print(f"✅ Saved: {output_path}")

print("\n🏁 Processing complete! All landmarks saved to:", OUTPUT_DIR)
