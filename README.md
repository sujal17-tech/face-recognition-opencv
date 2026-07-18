# Face Detection & Recognition (OpenCV)

This is a small Python project I built that can detect faces from a webcam and actually recognize *who* it's looking at — not just "there's a face here," but "that's Sujal." It uses OpenCV for everything: Haar Cascades to find faces in a frame, and an LBPH recognizer to tell people apart.

It was built as Task 3 of my AI internship at SYNTECXHUB.

## What it actually does

- Detects one or more faces in a live webcam feed (or a photo)
- Draws a box around each face with the person's name on it
- If it doesn't recognize someone, it labels them "Unknown" instead of guessing
- Comes with a simple script to register new people — just run it, say your name, and look at the camera for a few seconds

## Why I didn't use the `face_recognition` library

Most tutorials online use the `face_recognition` Python library, which is built on `dlib`. The problem is `dlib` needs to be compiled from C++ source when you install it, and that install step fails constantly on Windows unless you already have Visual Studio Build Tools set up — which most students don't.

So instead, I used OpenCV's own tools for both detection and recognition:
- **Haar Cascade** for finding faces (ships built into OpenCV, no extra downloads)
- **LBPH (Local Binary Patterns Histograms)** for recognizing them (part of `opencv-contrib-python`)

It's a lighter, older approach than deep-learning face embeddings, but it's dependable, installs cleanly everywhere, and still demonstrates the full pipeline: detect → learn faces → recognize.

## The files

| File | What it's for |
|---|---|
| `config.py` | All the shared settings — file paths, detection thresholds — in one place |
| `register_faces.py` | Run this first, once per person. Opens your webcam and saves ~30 cropped face photos |
| `train_model.py` | Trains the recognizer on everyone you've registered |
| `recognize_faces.py` | Runs live recognition — this is the "final" script that shows boxes + names |
| `haarcascade_frontalface_default.xml` | The face-detection model file OpenCV needs (included here directly, since it wasn't bundling correctly in some environments) |

> Note: I didn't upload my own `dataset/` (face photos) or `trainer/` (trained model) folders here, since those contain my actual face and I'd rather not have that public. If you clone this repo, just run `register_faces.py` and `train_model.py` yourself to generate your own.

## How to run it

```bash
pip install opencv-contrib-python numpy

python3 register_faces.py     # add yourself (or anyone) as a known face
python3 train_model.py        # train the recognizer on everyone you added
python3 recognize_faces.py    # watch it recognize you live
```

That's it — three commands, in that order.

## How it works, briefly

1. **Detection**: OpenCV's Haar Cascade scans the frame and returns a box `(x, y, w, h)` for every face it finds. Since it returns a *list*, the code loops over it — so multiple people in frame all get boxed and labeled, not just the first one.
2. **Registration**: each detected face gets cropped, resized, and saved as a training photo for that person.
3. **Training**: LBPH looks at texture patterns across all your registered photos and learns to tell people apart.
4. **Recognition**: for each new face, LBPH gives back a confidence *distance* — the lower the number, the more confident the match. If it's too high (not confident), the person gets labeled "Unknown" instead of a wrong guess.

## Honest notes

LBPH isn't as accurate as modern deep-learning face recognition — it can get confused by lighting changes or if you only register a handful of photos. For a real production system I'd reach for something like FaceNet embeddings or the `face_recognition`/dlib approach mentioned above. But for learning how detection + recognition actually fit together, and for something that installs painlessly on any machine, this did the job well.
