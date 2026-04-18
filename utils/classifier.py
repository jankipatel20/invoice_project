"""
classifier.py  –  Invoice / Not-Invoice image classifier
Uses a lightweight MobileNetV3 fine-tuned on the invoice_images dataset.
Falls back to keyword-based heuristic when model is not yet trained.
"""

import os
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ─── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "models" / "invoice_classifier.pth"
THRESHOLD  = 0.65   # Confidence threshold for "invoice" label

INVOICE_KEYWORDS = [
    # Generic billing terms
    "invoice", "facture", "rechnung", "factura", "fattura", "請求書", "发票",
    "bill", "receipt", "montant", "total", "subtotal", "tax", "tva",
    "mwst", "iva", "量", "合計", "金額", "税", "付款",
    # Date / number fields
    "date", "number", "n°", "ref", "due", "amount", "qty", "quantity",
    # Payment-related
    "payment", "paid", "balance", "discount", "price", "unit",
]

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─── Model definition ────────────────────────────────────────────────────────
def _build_model() -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    return model


# ─── Singleton loader ────────────────────────────────────────────────────────
_model     = None
_device    = None
_use_model = False

def _load_model():
    global _model, _device, _use_model
    if _model is not None:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if MODEL_PATH.exists():
        _model = _build_model().to(_device)
        state  = torch.load(MODEL_PATH, map_location=_device)
        _model.load_state_dict(state)
        _model.eval()
        _use_model = True
        print(f"[Classifier] Loaded trained model from {MODEL_PATH}")
    else:
        print("[Classifier] No trained model found – using keyword heuristic.")
        _use_model = False


# ─── Public API ──────────────────────────────────────────────────────────────
def is_invoice(image_source, ocr_text: str = "") -> dict:
    """
    Returns:
      {
        "is_invoice": bool,
        "confidence": float,
        "method": "model" | "heuristic",
        "reason": str
      }
    """
    _load_model()

    if _use_model:
        return _classify_with_model(image_source)
    return _classify_heuristic(ocr_text, image_source)


def _classify_with_model(source) -> dict:
    if isinstance(source, (str, Path)):
        img = Image.open(str(source)).convert("RGB")
    elif isinstance(source, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB) if source.ndim == 3 else source)
    else:
        img = source  # PIL already

    tensor = _TRANSFORM(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        probs = torch.softmax(_model(tensor), dim=1)[0]
    conf       = probs[1].item()   # class 1 = invoice
    is_inv     = conf >= THRESHOLD
    return {
        "is_invoice": is_inv,
        "confidence": round(conf, 3),
        "method":     "model",
        "reason":     "Classified by fine-tuned MobileNetV3" if is_inv
                      else f"Confidence {conf:.1%} below threshold {THRESHOLD:.0%}",
    }


def _classify_heuristic(text: str, source=None) -> dict:
    """
    Fast heuristic: keyword hit-rate in extracted text + basic structural check.
    """
    lower    = text.lower()
    hits     = [kw for kw in INVOICE_KEYWORDS if kw in lower]
    ratio    = len(hits) / max(len(lower.split()), 1)
    score    = min(1.0, len(hits) * 0.12 + ratio * 2)

    # Structural hint: invoices tend to have numbers / monetary amounts
    if re.search(r'\d+[.,]\d{2}', text):
        score = min(1.0, score + 0.15)
    if re.search(r'(total|montant|betrag|importe)[^\n]{0,30}\d', lower):
        score = min(1.0, score + 0.2)

    is_inv = score >= THRESHOLD
    return {
        "is_invoice": is_inv,
        "confidence": round(score, 3),
        "method":     "heuristic",
        "reason":     f"Keyword hits: {hits[:5]}" if is_inv
                      else f"Too few invoice keywords (score={score:.2f})",
    }
