"""
pipeline.py  –  OCR → Translate → Summarise for multilingual invoices
Uses EasyOCR (extraction) + Helsinki-NLP/opus-mt (translation) + BART/mT5 (summarise)
"""

import re
import time
from pathlib import Path
from typing import Optional

import easyocr
from transformers import (
    MarianMTModel, MarianTokenizer,
    pipeline as hf_pipeline,
)

from .preprocess import fast_preprocess
from .classifier import is_invoice

# ─── Language config ─────────────────────────────────────────────────────────
LANG_CONFIG = {
    "arabic":    {"easyocr": ["ar"],       "iso": "ar"},
    "french":    {"easyocr": ["fr"],       "iso": "fr"},
    "german":    {"easyocr": ["de"],       "iso": "de"},
    "japanese":  {"easyocr": ["ja"],       "iso": "ja"},
    "mandarin":  {"easyocr": ["ch_sim"],   "iso": "zh"},
    "spanish":   {"easyocr": ["es"],       "iso": "es"},
    "english":   {"easyocr": ["en"],       "iso": "en"},
}

# Helsinki-NLP model naming pattern:  opus-mt-{src}-{tgt}
# Some pairs route through English (pivot)
DIRECT_PAIRS = {
    # (src_iso, tgt_iso) : model_name
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",
    ("ja", "en"): "Helsinki-NLP/opus-mt-ja-en",
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
    ("en", "ja"): "Helsinki-NLP/opus-mt-en-jap",
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    ("fr", "de"): "Helsinki-NLP/opus-mt-fr-de",
    ("fr", "es"): "Helsinki-NLP/opus-mt-fr-es",
    ("de", "fr"): "Helsinki-NLP/opus-mt-de-fr",
    ("es", "fr"): "Helsinki-NLP/opus-mt-es-fr",
}

# ─── Caches ──────────────────────────────────────────────────────────────────
_ocr_readers: dict   = {}
_mt_models:   dict   = {}
_summariser          = None


# ─── OCR ─────────────────────────────────────────────────────────────────────
def _get_reader(lang_name: str) -> easyocr.Reader:
    key = lang_name
    if key not in _ocr_readers:
        codes = LANG_CONFIG[lang_name]["easyocr"]
        print(f"  [OCR] Loading EasyOCR model for {lang_name}…")
        _ocr_readers[key] = easyocr.Reader(codes, gpu=False)
    return _ocr_readers[key]


def extract_text(image_source, lang_name: str) -> tuple[str, list]:
    """
    Returns (full_text, raw_results).
    raw_results: list of (bbox, text, confidence)
    """
    processed = fast_preprocess(image_source)
    reader    = _get_reader(lang_name)
    results   = reader.readtext(processed)
    full_text = "\n".join(t for (_, t, _) in results)
    return full_text, results


# ─── Translation ─────────────────────────────────────────────────────────────
def _get_mt_model(src_iso: str, tgt_iso: str):
    pair = (src_iso, tgt_iso)
    if pair not in _mt_models:
        if pair in DIRECT_PAIRS:
            model_name = DIRECT_PAIRS[pair]
        else:
            # Pivot through English
            print(f"  [MT] No direct pair {src_iso}→{tgt_iso}, pivoting via English")
            return None   # caller will handle pivot
        print(f"  [MT] Loading {model_name}…")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model     = MarianMTModel.from_pretrained(model_name)
        _mt_models[pair] = (tokenizer, model)
    return _mt_models[pair]


def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Translate text from src_lang to tgt_lang (both as lang names, e.g. "french").
    """
    src_iso = LANG_CONFIG[src_lang]["iso"]
    tgt_iso = LANG_CONFIG[tgt_lang]["iso"]

    if src_iso == tgt_iso:
        return text   # Already same language

    result = _mt_translate(text, src_iso, tgt_iso)
    return result


def _mt_translate(text: str, src_iso: str, tgt_iso: str) -> str:
    pair = (src_iso, tgt_iso)
    mt   = _get_mt_model(src_iso, tgt_iso)

    if mt is None:
        # Pivot: src → en → tgt
        en_text = _mt_translate(text, src_iso, "en")
        return _mt_translate(en_text, "en", tgt_iso)

    tokenizer, model = mt
    # Split into manageable chunks (Helsinki models have 512 token limit)
    chunks     = _chunk_text(text, max_chars=400)
    translated = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt",
                           padding=True, truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=512)
        translated.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return " ".join(translated)


def _chunk_text(text: str, max_chars: int = 400) -> list[str]:
    sentences = re.split(r'(?<=[.!?\n])\s+', text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks or [text]


# ─── Summarisation ───────────────────────────────────────────────────────────
def _get_summariser():
    global _summariser
    if _summariser is None:
        print("  [Summariser] Loading facebook/bart-large-cnn…")
        _summariser = hf_pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1,           # CPU
            framework="pt",
        )
    return _summariser


def summarise(text: str, max_len: int = 180, min_len: int = 50) -> str:
    """Summarise an English invoice text into plain language."""
    summariser = _get_summariser()
    # BART has a 1024 token input limit
    trimmed = text[:3000]
    try:
        result = summariser(trimmed, max_length=max_len,
                            min_length=min_len, do_sample=False)
        return result[0]["summary_text"]
    except Exception as e:
        return f"[Summarisation failed: {e}]"


# ─── Master pipeline ──────────────────────────────────────────────────────────
def process_invoice(
    image_source,
    src_lang: str = "french",
    tgt_lang: str = "english",
    do_summarise: bool = True,
) -> dict:
    """
    Full pipeline: preprocess → OCR → validate → translate → summarise.

    Returns a structured dict with all results.
    """
    t0 = time.perf_counter()
    result = {
        "status":        "ok",
        "src_lang":      src_lang,
        "tgt_lang":      tgt_lang,
        "extracted_text": "",
        "translated_text": "",
        "summary":       "",
        "is_invoice":    False,
        "confidence":    0.0,
        "processing_time_s": 0.0,
        "lines":         0,
        "words":         0,
    }

    # 1. Extract text
    print(f"[Pipeline] Extracting text ({src_lang})…")
    raw_text, ocr_results = extract_text(image_source, src_lang)
    result["extracted_text"] = raw_text
    result["lines"]          = len(ocr_results)
    result["words"]          = len(raw_text.split())

    # 2. Validate
    print("[Pipeline] Classifying document…")
    verdict = is_invoice(image_source, raw_text)
    result["is_invoice"]  = verdict["is_invoice"]
    result["confidence"]  = verdict["confidence"]
    result["reject_reason"] = verdict.get("reason", "")

    if not verdict["is_invoice"]:
        result["status"] = "rejected"
        result["processing_time_s"] = round(time.perf_counter() - t0, 2)
        return result

    # 3. Translate
    print(f"[Pipeline] Translating {src_lang} → {tgt_lang}…")
    translated = translate_text(raw_text, src_lang, tgt_lang)
    result["translated_text"] = translated

    # 4. Summarise (always summarise in English for best BART quality)
    if do_summarise:
        print("[Pipeline] Summarising…")
        english_text = translated if tgt_lang == "english" else \
                       translate_text(raw_text, src_lang, "english")
        result["summary"] = summarise(english_text)

    result["processing_time_s"] = round(time.perf_counter() - t0, 2)
    print(f"[Pipeline] Done in {result['processing_time_s']}s")
    return result
