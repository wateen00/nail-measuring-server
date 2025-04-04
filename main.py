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

    if not hasattr(result, "multi_hand_landmarks") or not result.multi_hand_landmarks:
        os.remove(img_path)
        return jsonify({"error": "No hand detected"}), 400

    h, w, _ = image.shape
    landmarks = result.multi_hand_landmarks[0].landmark

    pairs = [(6, 10), (10, 14), (14, 18), (18, 22), (2, 4)]
    widths = [
        ((landmarks[a].x * w - landmarks[b].x * w) ** 2 +
         (landmarks[a].y * h - landmarks[b].y * h) ** 2) ** 0.5
        for a, b in pairs
    ]

    card_pixel_width = 300
    mm_per_pixel = 85.6 / card_pixel_width
    widths_mm = [round(w * mm_per_pixel, 1) for w in widths]
    sizes = [match_size(w) for w in widths_mm]

    os.remove(img_path)
    return jsonify({"widths_mm": widths_mm, "sizes": sizes})
