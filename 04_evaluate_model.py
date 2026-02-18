import matplotlib.pyplot as plt
import json
import os

# Load training history (saved manually if you prefer)
# You can modify your training script to save history to a JSON file
with open("/Users/jevanteqaiyim/Desktop/CapoAI/models/training_history.json", "r") as f:
    history = json.load(f)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(history["accuracy"], label="train_acc")
plt.plot(history["val_accuracy"], label="val_acc")
plt.legend(); plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history["loss"], label="train_loss")
plt.plot(history["val_loss"], label="val_loss")
plt.legend(); plt.title("Loss")

plt.show()
