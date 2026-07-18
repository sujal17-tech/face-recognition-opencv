import os
import sys
import pickle
import argparse
import cv2
from config import (
    MODEL_PATH, LABELS_PATH, FACE_SIZE,
    get_face_detector, detect_faces, CONFIDENCE_THRESHOLD,
)


def load_recognizer():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            "No trained model found. Run register_faces.py then train_model.py first."
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    with open(LABELS_PATH, "rb") as f:
        label_map = pickle.load(f)

    return recognizer, label_map


def recognize_and_annotate(frame, detector, recognizer, label_map):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, detector)

    for (x, y, w, h) in faces:
        face_crop = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)

        label_id, confidence = recognizer.predict(face_crop)

        if confidence <= CONFIDENCE_THRESHOLD:
            name = label_map.get(label_id, "Unknown")
            box_color = (0, 255, 0)
        else:
            name = "Unknown"
            box_color = (0, 0, 255)

        display_text = f"{name} ({confidence:.0f})"

        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        cv2.rectangle(frame, (x, y - 30), (x + w, y), box_color, -1)
        cv2.putText(frame, display_text, (x + 5, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame, len(faces)


def run_webcam(camera_index=0):
    detector = get_face_detector()
    recognizer, label_map = load_recognizer()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError("Could not access the webcam. Check camera_index or permissions.")

    print("[INFO] Starting live recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, face_count = recognize_and_annotate(frame, detector, recognizer, label_map)
        cv2.putText(annotated, f"Faces detected: {face_count}", (10, annotated.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Face Recognition - press 'q' to quit", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_on_image(image_path, output_path="recognized_output.jpg"):
    detector = get_face_detector()
    recognizer, label_map = load_recognizer()

    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    annotated, face_count = recognize_and_annotate(frame, detector, recognizer, label_map)
    cv2.imwrite(output_path, annotated)
    print(f"[INFO] Detected {face_count} face(s). Saved annotated image to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face detection & recognition")
    parser.add_argument("--image", type=str, help="Path to an image file")
    args = parser.parse_args()

    try:
        if args.image:
            run_on_image(args.image)
        else:
            run_webcam()
    except (FileNotFoundError, IOError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
        
