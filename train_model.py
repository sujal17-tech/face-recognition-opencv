import os
import pickle
import cv2
import numpy as np
from config import DATASET_DIR, MODEL_PATH, LABELS_PATH


def gather_training_data():
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    people = sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    )

    if not people:
        raise RuntimeError("No people found in dataset/. Register a face first.")

    for person_name in people:
        person_dir = os.path.join(DATASET_DIR, person_name)
        image_files = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".png"))]

        if not image_files:
            print(f"[WARN] No images found for '{person_name}', skipping.")
            continue

        label_map[current_label] = person_name

        for filename in image_files:
            path = os.path.join(person_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (200, 200))
            faces.append(img)
            labels.append(current_label)

        print(f"[INFO] Loaded {len(image_files)} images for '{person_name}' (label {current_label})")
        current_label += 1

    return faces, labels, label_map


def train_and_save():
    faces, labels, label_map = gather_training_data()

    print(f"\n[INFO] Training on {len(faces)} face images across {len(label_map)} people...")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)

    with open(LABELS_PATH, "wb") as f:
        pickle.dump(label_map, f)

    print(f"[INFO] Model saved to: {MODEL_PATH}")
    print(f"[INFO] Registered people: {list(label_map.values())}")
    print("\n[INFO] Now run 'python3 recognize_faces.py'")


if __name__ == "__main__":
    print("=" * 60)
    print(" Training Face Recognition Model")
    print("=" * 60)
    train_and_save()


