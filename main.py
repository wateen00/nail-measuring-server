# Nail Measuring Server - Python Flask + MediaPipe + OpenCV
# Simplified version: skips card detection, uses fixed card width in pixels for accurate measurement

from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os

app = Flask(__name__)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

size_chart = [
    {"size": 0, "width": 18},
    {"size": 1, "width": 17},
    {"size": 2, "width": 16},
    {"size": 3, "width": 15},
    {"size": 4, "width": 14},
    {"size": 5, "width": 13},
    {"size": 6, "width": 12},
    {"size": 7, "width": 11},
    {"size": 8, "width": 10},
    {"size": 9, "width": 9},
    {"size": 10, "width": 8},
    {"size": 11, "width": 7},
]

def match_size(width):
    closest = min(size_chart, key=lambda x: abs(x["width"] - width))
    return closest["size"]

@app.route('/measure', methods=['POST'])
def measure():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files['image']

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        img_path = tmp.name
        img_file.save(img_path)

    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if not result.hand_landmarks:
        os.remove(img_path)
        return jsonify({"error": "No hand detected"}), 400

    # Skip card detection; assume fixed width for now
    card_pixel_width = 300  # manually assumed
    mm_per_pixel = 85.6 / card_pixel_width

    h, w, _ = image.shape
    landmarks = result.hand_landmarks[0].landmark

    # Fingertip widths: index(8), middle(12), ring(16), pinky(20), thumb(4)
    pairs = [(6, 10), (10, 14), (14, 18), (18, 22), (2, 4)]
    widths = []

    for a, b in pairs:
        xa, ya = landmarks[a].x * w, landmarks[a].y * h
        xb, yb = landmarks[b].x * w, landmarks[b].y * h
        pixel_width = ((xa - xb)**2 + (ya - yb)**2) ** 0.5
        widths.append(pixel_width)

    widths_mm = [round(w * mm_per_pixel, 1) for w in widths]
    sizes = [match_size(w) for w in widths_mm]

    os.remove(img_path)
    return jsonify({"widths_mm": widths_mm, "sizes": sizes})

if __name__ == '__main__':
    app.run(debug=True)
