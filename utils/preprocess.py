"""
preprocess.py  –  Universal invoice image preprocessing

Does only what genuinely helps EasyOCR:
  1. Rescale to a resolution where text is reliably readable
  2. Deskew if the document is rotated (common in scans/photos)

No binarization. No CLAHE. No denoising.
EasyOCR's internal CNN handles all of that better than we can.
Works on any invoice style: digital, scanned, photo, GAN-generated.
"""

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path


MIN_LONG_EDGE  = 1400   # upscale below this — small text hurts OCR accuracy
MAX_LONG_EDGE  = 3000   # downscale above this — no accuracy gain, just slower
DESKEW_LIMIT   = 0.5    # degrees — ignore skew smaller than this


def load_image(source) -> np.ndarray:
    """Accept file path (str/Path), raw bytes, or numpy array (BGR)."""
    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {source}")
        return img
    if isinstance(source, bytes):
        arr = np.frombuffer(source, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return source  # already ndarray


def _rescale(img: np.ndarray) -> np.ndarray:
    h, w      = img.shape[:2]
    long_edge = max(h, w)
    if long_edge < MIN_LONG_EDGE:
        scale, interp = MIN_LONG_EDGE / long_edge, cv2.INTER_CUBIC
    elif long_edge > MAX_LONG_EDGE:
        scale, interp = MAX_LONG_EDGE / long_edge, cv2.INTER_AREA
    else:
        return img
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=interp)


def _deskew(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 200:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    if abs(angle) < DESKEW_LIMIT or abs(angle) > 10:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    print(f"  [Preprocess] Deskewed {angle:.2f}°")
    return rotated


def fast_preprocess(source) -> np.ndarray:
    """
    Returns a color BGR numpy array ready for EasyOCR.
    Handles any invoice type — digital, scanned, photo, synthetic.
    """
    img = load_image(source)
    img = _rescale(img)
    img = _deskew(img)
    return img


def preprocess_batch(sources: list, max_workers: int = 4) -> list:
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(fast_preprocess, sources))


def save_preview(source, out_path: str = "output_preview.png") -> str:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    orig  = load_image(source)
    final = fast_preprocess(source)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (title, im) in zip(axes, [
        ("Original",     cv2.cvtColor(orig,  cv2.COLOR_BGR2RGB)),
        ("Preprocessed", cv2.cvtColor(final, cv2.COLOR_BGR2RGB)),
    ]):
        ax.imshow(im); ax.set_title(title, fontsize=11); ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    t0   = time.perf_counter()
    arr  = fast_preprocess(path)
    print(f"Done in {time.perf_counter()-t0:.3f}s  shape={arr.shape}")
    save_preview(path)
    print("Preview saved → output_preview.png")