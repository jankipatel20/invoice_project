"""
pipeline.py  –  OCR → Translate → Summarise for any-language invoices

Translation strategy:
  NON-LATIN scripts (Arabic, Chinese, Japanese):
    - Separate the text into ASCII lines (keep verbatim) and foreign-script
      BLOCKS (consecutive non-ASCII lines grouped together).
    - Each foreign block is sent to Helsinki as ONE passage — giving the model
      the sentence context it needs to produce a real translation instead of
      hallucinating on isolated 3-char fragments.
    - Hallucination guard: if output has >40% repeated tokens or music notes,
      we retry with a cleaned version of the block before giving up.

  LATIN scripts (Spanish, French, German):
    - Full line-by-line translation (can't detect foreign vs English by char).
    - Lines with no alphabetic characters pass through unchanged.
"""

import re
import time
from collections import Counter

import easyocr
from transformers import (
    MarianMTModel, MarianTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
)

from .preprocess import fast_preprocess
from .classifier import is_invoice


# ─── Config ──────────────────────────────────────────────────────────────────

LANG_CONFIG = {
    "arabic":   {"iso": "ar", "latin_script": False},
    "french":   {"iso": "fr", "latin_script": True},
    "german":   {"iso": "de", "latin_script": True},
    "japanese": {"iso": "ja", "latin_script": False},
    "mandarin": {"iso": "zh", "latin_script": False},
    "spanish":  {"iso": "es", "latin_script": True},
    "english":  {"iso": "en", "latin_script": True},
}

SCRIPT_READERS = {
    "latin":    ["en", "fr", "de", "es", "pt", "it", "nl"],
    "arabic":   ["ar", "en"],
    "chinese":  ["ch_sim", "en"],
    "japanese": ["ja", "en"],
}

LANG_TO_SCRIPT = {
    "english":  "latin",
    "french":   "latin",
    "german":   "latin",
    "spanish":  "latin",
    "arabic":   "arabic",
    "mandarin": "chinese",
    "japanese": "japanese",
}

TRANSLATION_PAIRS = {
    ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("ja", "en"): "Helsinki-NLP/opus-mt-ja-en",
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
    ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("en", "ja"): "Helsinki-NLP/opus-mt-en-jap",
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    ("fr", "de"): "Helsinki-NLP/opus-mt-fr-de",
    ("fr", "es"): "Helsinki-NLP/opus-mt-fr-es",
    ("de", "fr"): "Helsinki-NLP/opus-mt-de-fr",
    ("es", "fr"): "Helsinki-NLP/opus-mt-es-fr",
}

# Minimum confidence for EasyOCR detections.
# Arabic/CJK/Japanese use a higher threshold because low-confidence detections
# in complex scripts are almost always garbage fragments, not real words.
_CONF_THRESHOLD = {
    "latin":    0.20,
    "arabic":   0.35,
    "chinese":  0.30,
    "japanese": 0.30,
}

# ─── Model caches ────────────────────────────────────────────────────────────
_ocr_readers: dict = {}
_mt_cache:    dict = {}
_t5_tokenizer      = None
_t5_model          = None


# ─── Warmup ──────────────────────────────────────────────────────────────────
def warmup(src_langs: list[str] = None):
    """Pre-load EasyOCR + T5 at server startup."""
    if src_langs is None:
        src_langs = ["english"]
    print("[Warmup] Pre-loading models …")
    t0 = time.perf_counter()
    seen = set()
    for lang in src_langs:
        family = LANG_TO_SCRIPT.get(lang, "latin")
        if family not in seen:
            _get_ocr_reader(lang)
            seen.add(family)
    _get_t5()
    print(f"[Warmup] Done in {time.perf_counter() - t0:.1f}s")


# ─── 1. OCR ───────────────────────────────────────────────────────────────────
def _get_ocr_reader(src_lang: str) -> easyocr.Reader:
    family = LANG_TO_SCRIPT.get(src_lang, "latin")
    if family not in _ocr_readers:
        codes = SCRIPT_READERS[family]
        print(f"  [OCR] Loading EasyOCR family='{family}' codes={codes} …")
        _ocr_readers[family] = easyocr.Reader(codes, gpu=False)
    return _ocr_readers[family]


def extract_text(image_source, src_lang: str) -> tuple[str, list]:
    img      = fast_preprocess(image_source)
    reader   = _get_ocr_reader(src_lang)
    img_rgb  = img[:, :, ::-1]
    results  = reader.readtext(img_rgb)
    family   = LANG_TO_SCRIPT.get(src_lang, "latin")
    min_conf = _CONF_THRESHOLD.get(family, 0.20)
    results  = [(bbox, text, conf)
                for bbox, text, conf in results if conf >= min_conf]
    return "\n".join(t for _, t, _ in results), results


# ─── 2. Text cleaning ─────────────────────────────────────────────────────────
def _clean(text: str) -> str:
    """Drop lines where fewer than 40% of characters are meaningful."""
    keep = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        good = len(re.findall(
            r'[\w'
            r'\u0600-\u06ff'
            r'\u4e00-\u9fff'
            r'\u3040-\u30ff'
            r'\uac00-\ud7af'
            r'.,:/\-$€£¥%()]',
            s
        ))
        if good / max(len(s), 1) >= 0.40:
            keep.append(s)
    return "\n".join(keep)


# ─── 3. Hallucination detection ───────────────────────────────────────────────
def _is_hallucination(text: str) -> bool:
    """
    Returns True if the MT output looks like a hallucination loop.
    Helsinki hallucinates like: "♪ ♪ ♪ ♪" or "no, no, no, no, no"
    when given garbled or too-short input.
    """
    s = text.strip()
    if not s:
        return False
    if s.count("♪") > 2:
        return True
    tokens = s.split()
    if len(tokens) < 5:
        return False
    most_common = Counter(tokens).most_common(1)[0][1]
    return most_common / len(tokens) > 0.40


# ─── 4. Translation ───────────────────────────────────────────────────────────
def _get_mt(src_iso: str, tgt_iso: str):
    pair = (src_iso, tgt_iso)
    if pair not in _mt_cache:
        if pair not in TRANSLATION_PAIRS:
            return None
        name = TRANSLATION_PAIRS[pair]
        print(f"  [MT] Loading {name} …")
        tok   = MarianTokenizer.from_pretrained(name)
        model = MarianMTModel.from_pretrained(name)
        model.eval()
        _mt_cache[pair] = (tok, model)
    return _mt_cache[pair]


def _run_mt(text: str, tok, model) -> str:
    """
    Run the MT model on a text string, respecting the 512-token limit
    by chunking if necessary. Returns translated string.
    """
    chunks = _split_chunks(text)
    out    = []
    for chunk in chunks:
        inp = tok(chunk, return_tensors="pt",
                  padding=True, truncation=True, max_length=512)
        ids = model.generate(**inp, max_new_tokens=512)
        out.append(tok.decode(ids[0], skip_special_tokens=True))
    return " ".join(out)


def _split_chunks(text: str, max_chars: int = 400) -> list[str]:
    sentences   = re.split(r'(?<=[.!?\n])\s+', text)
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


def _translate_non_latin(text: str, src_iso: str, tgt_iso: str) -> str:
    """
    Block-level translation for non-Latin scripts (Arabic, Chinese, Japanese).

    Strategy:
      1. Walk through the lines and separate them into two kinds:
           - ASCII lines  → pass through verbatim (amounts, codes, dates, English)
           - Foreign lines → accumulate into a BLOCK
      2. When an ASCII line is encountered after a block, translate the whole
         accumulated block as ONE string — Helsinki gets full sentence context
         instead of isolated 3-char fragments, producing real translations.
      3. Hallucination guard: if the output looks looped/repeated, clean the
         input (strip punctuation noise) and retry once before giving up.
    """
    mt = _get_mt(src_iso, tgt_iso)
    if mt is None:
        mid = _translate_non_latin(text, src_iso, "en")
        return _translate_non_latin(mid, "en", tgt_iso)

    tok, model = mt

    lines         = text.splitlines()
    output_lines  = []
    foreign_block = []   # accumulates consecutive non-ASCII lines

    def flush_block():
        """Translate the accumulated foreign block as one passage."""
        if not foreign_block:
            return
        block_text = "\n".join(foreign_block)
        translated = _run_mt(block_text, tok, model)

        if _is_hallucination(translated):
            # Retry with stripped punctuation — sometimes stray chars trigger loops
            clean_block = re.sub(r'[^\w\s\u0600-\u06ff\u4e00-\u9fff\u3040-\u30ff]',
                                 ' ', block_text).strip()
            translated = _run_mt(clean_block, tok, model)

        if _is_hallucination(translated):
            # Still hallucinating — fall back to original block text
            output_lines.extend(foreign_block)
        else:
            # Split translated output back into lines (best effort)
            translated_lines = translated.splitlines()
            if len(translated_lines) == len(foreign_block):
                output_lines.extend(translated_lines)
            else:
                # Line counts don't match — just append as a single translated block
                output_lines.append(translated)

        foreign_block.clear()

    for line in lines:
        s = line.strip()
        if not s:
            flush_block()
            output_lines.append("")
            continue

        has_foreign = any(ord(c) > 127 for c in s)

        if has_foreign:
            foreign_block.append(s)
        else:
            # ASCII line — flush any pending foreign block first, then keep as-is
            flush_block()
            output_lines.append(s)

    # Flush any remaining block at end of document
    flush_block()

    return "\n".join(output_lines)


def _translate_latin(text: str, src_iso: str, tgt_iso: str) -> str:
    """
    Line-by-line translation for Latin-script languages.
    Lines with no alphabetic characters pass through unchanged.
    """
    mt = _get_mt(src_iso, tgt_iso)
    if mt is None:
        mid = _translate_latin(text, src_iso, "en")
        return _translate_latin(mid, "en", tgt_iso)

    tok, model = mt
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            out.append(line)
            continue
        has_alpha = bool(re.search(r'[a-zA-Z\u00c0-\u024f]', s))
        if not has_alpha:
            out.append(s)
        else:
            result = _run_mt(s, tok, model)
            out.append(result)
    return "\n".join(out)


def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    src_iso      = LANG_CONFIG[src_lang]["iso"]
    tgt_iso      = LANG_CONFIG[tgt_lang]["iso"]
    is_latin_src = LANG_CONFIG[src_lang]["latin_script"]
    if src_iso == tgt_iso:
        return text
    if is_latin_src:
        return _translate_latin(text, src_iso, tgt_iso)
    return _translate_non_latin(text, src_iso, tgt_iso)


# ─── 5. Summarisation (T5-small) ─────────────────────────────────────────────
_T5 = "t5-small"

def _get_t5():
    global _t5_tokenizer, _t5_model
    if _t5_model is None:
        print(f"  [Summariser] Loading {_T5} …")
        _t5_tokenizer = T5Tokenizer.from_pretrained(_T5, legacy=True)
        _t5_model     = T5ForConditionalGeneration.from_pretrained(_T5)
        _t5_model.eval()
        print("  [Summariser] Ready.")
    return _t5_tokenizer, _t5_model


def summarise(text: str, max_len: int = 120, min_len: int = 30) -> str:
    tok, model = _get_t5()
    prompt     = "summarize: " + text[:800].replace("\n", " ")
    try:
        import torch
        inp = tok(prompt, return_tensors="pt",
                  truncation=True, max_length=512)
        with torch.no_grad():
            ids = model.generate(
                inp["input_ids"],
                max_new_tokens=max_len,
                min_new_tokens=min_len,
                num_beams=2,
                length_penalty=1.5,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        return tok.decode(ids[0], skip_special_tokens=True)
    except Exception as e:
        return f"[Summarisation failed: {e}]"


# ─── 6. Master pipeline ───────────────────────────────────────────────────────
def process_invoice(
    image_source,
    src_lang:     str  = "english",
    tgt_lang:     str  = "english",
    do_summarise: bool = True,
) -> dict:
    """
    Full pipeline: preprocess → OCR → validate → translate → summarise.
    Works for any invoice in any supported language.
    """
    t0     = time.perf_counter()
    result = {
        "status":            "ok",
        "src_lang":          src_lang,
        "tgt_lang":          tgt_lang,
        "extracted_text":    "",
        "translated_text":   "",
        "summary":           "",
        "is_invoice":        False,
        "confidence":        0.0,
        "reject_reason":     "",
        "lines":             0,
        "words":             0,
        "processing_time_s": 0.0,
    }

    # Step 1: Extract
    print(f"[Pipeline] Extracting text  src_lang={src_lang} …")
    raw_text, ocr_results = extract_text(image_source, src_lang)
    clean_text            = _clean(raw_text)
    result["extracted_text"] = clean_text
    result["lines"]          = len(ocr_results)
    result["words"]          = len(clean_text.split())
    print(f"[Pipeline] {result['lines']} regions  {result['words']} words")

    # Step 2: Validate
    print("[Pipeline] Classifying document …")
    verdict              = is_invoice(image_source, clean_text)
    result["is_invoice"] = verdict["is_invoice"]
    result["confidence"] = verdict["confidence"]
    result["reject_reason"] = verdict.get("reason", "")

    if not verdict["is_invoice"]:
        result["status"]            = "rejected"
        result["processing_time_s"] = round(time.perf_counter() - t0, 2)
        return result

    # Step 3: Translate
    src_iso = LANG_CONFIG[src_lang]["iso"]
    tgt_iso = LANG_CONFIG[tgt_lang]["iso"]

    if src_iso == tgt_iso:
        print("[Pipeline] src == tgt — skipping translation.")
        result["translated_text"] = clean_text
    else:
        print(f"[Pipeline] Translating {src_lang} → {tgt_lang} …")
        result["translated_text"] = translate_text(clean_text, src_lang, tgt_lang)

    # Step 4: Summarise
    if do_summarise:
        print("[Pipeline] Summarising …")
        if tgt_iso == "en":
            en_text = result["translated_text"]
        elif src_iso == "en":
            en_text = clean_text
        else:
            en_text = translate_text(clean_text, src_lang, "english")
        result["summary"] = summarise(en_text)

    result["processing_time_s"] = round(time.perf_counter() - t0, 2)
    print(f"[Pipeline] Done in {result['processing_time_s']}s")
    return result