import os
import cv2
from config import DATASET_DIR, FACE_SIZE, get_face_detector, detect_faces

SAMPLES_TO_CAPTURE = 30
CAPTURE_EVERY_N_FRAMES = 3


def register_person(name, camera_index=0):
    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    existing = [f for f in os.listdir(person_dir) if f.endswith(".jpg")]
    start_index = len(existing)

    detector = get_face_detector()
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise IOError("Could not access the webcam. Check camera_index or permissions.")

    print(f"\n[INFO] Capturing faces for '{name}'.")
    print("[INFO] Look at the camera. Press 'q' to stop early.\n")

    count = 0
    frame_num = 0

    while count < SAMPLES_TO_CAPTURE:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from webcam.")
            break

        frame_num += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, detector)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {count}/{SAMPLES_TO_CAPTURE}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if frame_num % CAPTURE_EVERY_N_FRAMES == 0:
                face_crop = gray[y:y + h, x:x + w]
                face_crop = cv2.resize(face_crop, FACE_SIZE)

                filename = os.path.join(person_dir, f"{start_index + count}.jpg")
                cv2.imwrite(filename, face_crop)
                count += 1

            break

        cv2.imshow("Register Face - press 'q' to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[INFO] Done. Saved {count} new face images to: {person_dir}")
    print("[INFO] Now run 'python3 train_model.py' to (re)train the recognizer.")


if __name__ == "__main__":
    print("=" * 60)
    print(" Face Registration")
    print("=" * 60)
    person_name = input("Enter the person's name: ").strip()

    if not person_name:
        print("[ERROR] Name cannot be empty.")
    else:
        register_person(person_name)
