import cv2
import numpy as np
import csv
import os
from collections import defaultdict
from utils.state_detection import extract_features

# Initialize webcam (using AVFoundation backend for macOS)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"📷 Webcam resolution: {int(width)}x{int(height)}")

# Initialize storage
features_list = []
labels_list = []
label_counts = defaultdict(int)

# Define class labels and number of examples per class
class_labels = {"a": "attentive", "c": "confused", "d": "distracted"}
target_per_class = 50

print("\n--- DATA COLLECTION MODE ---")
print("Press A / C / D to label a frame as Attentive / Confused / Distracted")
print("Press Q to quit.\n")

# Start data collection loop
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print(" Frame not captured.")
        break

    # Display label counters on frame
    overlay = frame.copy()
    y = 20
    for key, label in class_labels.items():
        cv2.putText(
            overlay,
            f"{label}: {label_counts[label]}/{target_per_class}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        y += 25

    # Show webcam with overlay
    cv2.imshow("Webcam - Press A/C/D to record", overlay)

    key = cv2.waitKeyEx(1)

    if key in [ord("q"), ord("Q")]:
        break

    elif key in [ord(k) for k in class_labels]:
        label = class_labels[chr(key).lower()]
        features = extract_features(frame)

        if features is not None and not all(f == 0 for f in features):
            features_list.append(features)
            labels_list.append(label)
            label_counts[label] += 1
            print(f" Captured: {label} ({label_counts[label]}/{target_per_class})")
        else:
            print(" Invalid features. Adjust pose or lighting.")

    # Stop if all classes have enough samples
    if all(label_counts[l] >= target_per_class for l in class_labels.values()):
        print("\n All class samples collected.")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Create dataset directory if not exists
dataset_dir = os.path.join(os.getcwd(), "dataset")
os.makedirs(dataset_dir, exist_ok=True)

# Save raw features and labels in NumPy format
np.save(os.path.join(dataset_dir, "labeled_features.npy"), np.array(features_list))
np.save(os.path.join(dataset_dir, "labels.npy"), np.array(labels_list))

# Save dataset in CSV format with readable feature names
csv_path = os.path.join(dataset_dir, "dataset.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "eye_openness_left",
        "eye_openness_right",
        "eyebrow_distance_left",
        "eyebrow_distance_right",
        "mouth_openness",
        "head_yaw_asymmetry",
        "shoulder_distance",
        "head_tilt_vertical",
        "label"
    ])
    for feats, label in zip(features_list, labels_list):
        writer.writerow(list(feats) + [label])

print(f"\n Saved dataset to: {dataset_dir}")