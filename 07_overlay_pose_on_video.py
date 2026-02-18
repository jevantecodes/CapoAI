import cv2
import mediapipe as mp
import os

# Paths
INPUT_VIDEO = "data/au/capoeria_au_5.mp4"     # ← change to 2, 3, etc
OUTPUT_VIDEO = "data/au/capoeria_au_5_overlay.mp4"

# Mediapipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {INPUT_VIDEO}")

# Get video properties for writer
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# FourCC: use 'mp4v' for .mp4 files
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

frame_idx = 0
print(f"🎥 Processing {INPUT_VIDEO} ...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # Convert to RGB for mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Draw landmarks on the original frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_style.get_default_pose_landmarks_style(),
        )

    # Show frame
    cv2.imshow("Capoeira Overlay", frame)

    # Write frame to output video
    out.write(frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Done! Saved overlay video to: {OUTPUT_VIDEO}")
