"""
app.py  –  Flask web app for Invoice OCR / Translation / Summarisation
Run: python app.py
"""

import base64
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2

from utils.pipeline import process_invoice, LANG_CONFIG, warmup
from utils.preprocess import save_preview

app = Flask(__name__, template_folder="ui/templates", static_folder="ui/static")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024   # 20 MB


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", languages=list(LANG_CONFIG.keys()))


@app.route("/process", methods=["POST"])
def process():
    if request.is_json:
        data     = request.get_json()
        b64_data = data.get("image_b64", "")
        src_lang = data.get("src_lang", "english").lower()
        tgt_lang = data.get("tgt_lang", "english").lower()
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        img_array = _bytes_to_array(base64.b64decode(b64_data))
    else:
        file     = request.files.get("image")
        src_lang = request.form.get("src_lang", "english").lower()
        tgt_lang = request.form.get("tgt_lang", "english").lower()
        if not file:
            return jsonify({"error": "No image provided"}), 400
        img_array = _bytes_to_array(file.read())

    if img_array is None:
        return jsonify({"error": "Could not decode image"}), 400
    if src_lang not in LANG_CONFIG:
        return jsonify({"error": f"Unsupported source language: {src_lang}"}), 400
    if tgt_lang not in LANG_CONFIG:
        return jsonify({"error": f"Unsupported target language: {tgt_lang}"}), 400

    try:
        result = process_invoice(img_array, src_lang=src_lang, tgt_lang=tgt_lang)
        result["preview_b64"] = _make_preview_b64(img_array)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _bytes_to_array(data: bytes) -> np.ndarray | None:
    try:
        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _make_preview_b64(img_array: np.ndarray) -> str:
    try:
        rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((512, 512))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


# ─── Entry ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    # Pre-load EasyOCR (latin family) + T5 before the server starts.
    # Add more languages here if you want them warm on startup,
    # e.g. warmup(["english", "mandarin", "arabic"])
    # MT models (Helsinki) still load on first use per language pair — they are
    # too numerous to pre-load all at once and load in ~10-15s each.
    warmup(["english"])

    print(f"\n Invoice AI running on http://localhost:{port}\n")
    app.run(debug=False, port=port, threaded=True)