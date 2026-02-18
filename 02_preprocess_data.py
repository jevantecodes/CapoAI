import numpy as np
import os
import json

LANDMARK_DIR = "processed_landmarks"
OUTPUT_DIR = "dataset"
SEQUENCE_LENGTH = 50  # frames per clip (pad or cut)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----- simple to_categorical replacement -----
def to_categorical_np(labels, num_classes=None):
    labels = np.array(labels, dtype=int)
    if num_classes is None:
        num_classes = labels.max() + 1
    one_hot = np.eye(num_classes)[labels]
    return one_hot

# each subfolder is a move label
MOVES = sorted(
    [
        d for d in os.listdir(LANDMARK_DIR)
        if os.path.isdir(os.path.join(LANDMARK_DIR, d))
    ]
)
print("📋 Moves (label order):", MOVES)

X, y = [], []

for label_idx, move in enumerate(MOVES):
    move_folder = os.path.join(LANDMARK_DIR, move)
    for file_name in os.listdir(move_folder):
        if not file_name.endswith(".npy"):
            continue

        sample = np.load(os.path.join(move_folder, file_name))  # (frames, 99)

        # normalize sequence length
        if len(sample) < SEQUENCE_LENGTH:
            pad = np.zeros((SEQUENCE_LENGTH - len(sample), sample.shape[1]))
            sample = np.vstack([sample, pad])
        else:
            sample = sample[:SEQUENCE_LENGTH]

        X.append(sample)
        y.append(label_idx)

X = np.array(X)
y = to_categorical_np(y)  # 🔥 no TensorFlow needed

np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

# save label mapping for later
with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
    json.dump(MOVES, f, indent=2)

print("✅ Saved dataset to 'dataset/'")
print("X shape:", X.shape)   # (num_samples, 50, 99)
print("y shape:", y.shape)   # (num_samples, num_classes)
