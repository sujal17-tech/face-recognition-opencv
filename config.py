import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
MODEL_PATH = os.path.join(TRAINER_DIR, "trainer.yml")
LABELS_PATH = os.path.join(TRAINER_DIR, "labels.pickle")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAINER_DIR, exist_ok=True)

CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

FACE_SIZE = (200, 200)
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_FACE_SIZE = (60, 60)

CONFIDENCE_THRESHOLD = 70


def get_face_detector():
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise IOError(f"Could not load Haar cascade from: {CASCADE_PATH}")
    return detector


def detect_faces(gray_frame, detector=None):
    if detector is None:
        detector = get_face_detector()
    faces = detector.detectMultiScale(
        gray_frame,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=MIN_FACE_SIZE,
    )
    return faces
