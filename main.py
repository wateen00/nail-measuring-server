from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os

app = Flask(__name__)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

size_chart = [{"size": i, "width": 18 - i} for i in range(12)]

def match_size(width):
    return min(size_chart, key=lambda x: abs(x["width"] - width))["size"]

@app.route("/measure", methods=["POST"])
def measure():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img_path = tmp.name
        img_file.save(img_path)

    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if not result.hand_landmarks:
        os.remove(img_path)
        return jsonify({"error": "No hand detected"}), 400

    h, w, _ = image.shape
    landmarks = result.hand_landmarks[0].landmark
    pairs = [(6, 10), (10, 14), (14, 18), (18, 22), (2, 4)]
    widths = [
        ((landmarks[a].x * w - landmarks[b].x * w) ** 2 +
         (landmarks[a].y * h - landmarks[b].y * h) ** 2) ** 0.5
        for a, b in pairs
    ]

    card_pixel_width = 300
    mm_per_pixel = 85.6 / card_pixel_width
    widths_mm = [round(p * mm_per_pixel, 1) for p in widths]
    sizes = [match_size(mm) for mm in widths_mm]

    os.remove(img_path)
    return jsonify({"widths_mm": widths_mm, "sizes": sizes})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
