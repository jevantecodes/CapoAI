import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

POSE_CONNECTIONS = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12),
    (23, 24),
    (11, 23), (12, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

GINGA_DIR = "processed_landmarks/mini_au"

# Get all .npy files
files = [f for f in os.listdir(GINGA_DIR) if f.endswith(".npy")]

for file_name in files:
    print(f"📂 Visualizing {file_name}")
    npy_path = os.path.join(GINGA_DIR, file_name)

    data = np.load(npy_path)
    num_frames = data.shape[0]

    fig, ax = plt.subplots(figsize=(5, 7))
    scat = ax.scatter([], [])
    lines = [ax.plot([], [])[0] for _ in POSE_CONNECTIONS]

    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 0)
    ax.set_title(file_name)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        for line in lines:
            line.set_data([], [])
        return [scat, *lines]

    def update(frame_idx):
        frame = data[frame_idx].reshape(-1, 3)
        x = frame[:, 0]
        y = -frame[:, 1]

        coords = np.column_stack([x, y])
        scat.set_offsets(coords)

        for i, (a, b) in enumerate(POSE_CONNECTIONS):
            lines[i].set_data([x[a], x[b]], [y[a], y[b]])

        ax.set_title(f"{file_name} — Frame {frame_idx+1}/{num_frames}")
        return [scat, *lines]

    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, interval=33)
    plt.show()
