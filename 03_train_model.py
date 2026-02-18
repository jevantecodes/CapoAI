import os
import json
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ----- Paths -----
DATA_PATH = "dataset"
MODEL_PATH = "models"
os.makedirs(MODEL_PATH, exist_ok=True)

# ----- Load data -----
X = np.load(os.path.join(DATA_PATH, "X.npy"))  # (num_samples, seq_len, features)
y = np.load(os.path.join(DATA_PATH, "y.npy"))  # (num_samples, num_classes)

label_map_path = os.path.join(DATA_PATH, "label_map.json")
if os.path.exists(label_map_path):
    with open(label_map_path, "r") as f:
        MOVES = json.load(f)
else:
    MOVES = [f"class_{i}" for i in range(y.shape[1])]

num_samples = X.shape[0]
SEQUENCE_LENGTH = X.shape[1]
FEATURE_COUNT = X.shape[2]
LABEL_COUNT = y.shape[1]

print("📊 Data summary:")
print("  Samples      :", num_samples)
print("  Seq length   :", SEQUENCE_LENGTH)
print("  Feature count:", FEATURE_COUNT)
print("  Num classes  :", LABEL_COUNT)
print("  Labels       :", MOVES)

# ----- Build smaller, stabler model -----
model = Sequential([
    LSTM(
        64,
        return_sequences=True,
        activation="tanh",          # 👈 use tanh (safer than relu here)
        input_shape=(SEQUENCE_LENGTH, FEATURE_COUNT),
    ),
    LSTM(32, return_sequences=False, activation="tanh"),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(LABEL_COUNT, activation="softmax"),
])

optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)  # 👈 smaller lr + gradient clipping

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ----- Callbacks -----
es = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
)

mc = ModelCheckpoint(
    os.path.join(MODEL_PATH, "capoeira_model_best.keras"),  # 👈 new format
    save_best_only=True,
)

# ----- Train -----
history = model.fit(
    X,
    y,
    validation_split=0.25,   # tiny dataset -> keep some for val
    epochs=40,
    batch_size=2,            # small batch for tiny data
    callbacks=[es, mc],
)

# ----- Save final model -----
final_model_path = os.path.join(MODEL_PATH, "capoeira_model_final.keras")
model.save(final_model_path)
print(f"✅ Model trained and saved to: {final_model_path}")

# ----- Save training history -----
history_path = os.path.join(MODEL_PATH, "training_history.json")
with open(history_path, "w") as f:
    json.dump(history.history, f, indent=2)

print(f"📈 Training history saved to: {history_path}")
