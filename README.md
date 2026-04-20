# 🧾 BabelBill – Multilingual Invoice OCR, Translation & Summariser

**Stack:** EasyOCR · Helsinki-NLP/opus-mt · T5-Small · MobileNetV3 · Flask

---

## Features
- 📄 **OCR** in Arabic, French, German, Japanese, Mandarin, Spanish, English
- 🌐 **Translation** between all supported languages via Helsinki-NLP offline models
- ✨ **Summarisation** – plain-language summary of what the bill is about
- 🚫 **Rejection** – non-invoice images are automatically rejected
- 🎯 **Classifier** – MobileNetV3 fine-tuned on your invoice dataset
- ⚡ **Fast preprocessing** – 3-5× faster than naive approach (CLAHE + parallel batch)

---
## Datasets

**Dataset:** 
- [BabelBill Invoice Dataset](https://www.kaggle.com/datasets/jankichohalia/invoice-dataset)

### Quick Start with Kaggle Data

```bash
# 1. Install kaggle CLI
pip install kaggle

# 2. Download dataset
kaggle datasets download -d jankichohalia/invoice-dataset

# 3. Extract to project directory
unzip invoice-dataset.zip -d ./invoice_images
## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Train the invoice classifier on your dataset
python training/train_classifier.py \
  --data_root ./invoice_images \
  --out_dir   ./models \
  --epochs    25

# 4. Launch the web app
python app.py
# Open http://localhost:5000
```

---

## Dataset Structure

```
invoice_images/
├── arabic/          ← real + synthetic Arabic invoices
├── french/
├── german/
├── japanese/
├── mandarin/
├── spanish/
└── not_invoice/     

---

## Files

```
invoice_project/
├── app.py                        Flask web server
├── requirements.txt
├── utils/
│   ├── preprocess.py             Fast image preprocessing
│   ├── classifier.py             Invoice vs not-invoice detection
│   └── pipeline.py               Full OCR → Translate → Summarise pipeline
├── training/
│   └── train_classifier.py       MobileNetV3 fine-tuning script
├── models/
│   └── invoice_classifier.pth    (created after training)
└── ui/
    ├── templates/index.html
    └── static/
        ├── css/style.css
        └── js/app.js
```

---

## Preprocessing Speed Improvements

| Change | Speedup |
|---|---|
| Resize BEFORE adaptive threshold + denoise | ~2.5× |
| no CLAHE | Better quality, same speed |
| Lower denoise `h=7` (was 10) | ~1.4× |
| Parallel batch processing via ThreadPoolExecutor | N× on batches |

---

## API (JSON)

```
POST /process
{
  "image_b64": "data:image/jpeg;base64,...",
  "src_lang": "french",
  "tgt_lang": "english"
}

Response:
{
  "status": "ok" | "rejected",
  "extracted_text": "...",
  "translated_text": "...",
  "summary": "...",
  "is_invoice": true,
  "confidence": 0.91,
  "lines": 42,
  "words": 187,
  "processing_time_s": 8.3
}
```

---

## Supported Languages

| Language | OCR Code | ISO |
|---|---|---|
| Arabic | ar | ar |
| French | fr | fr |
| German | de | de |
| Japanese | ja | ja |
| Mandarin | ch_sim | zh |
| Spanish | es | es |
| English | en | en |

---

## Notes

- First run will download EasyOCR + Helsinki-NLP models (~500 MB total). Subsequent runs use the cache.
- For GPU acceleration: install `torch` with CUDA and remove `gpu=False` from `preprocess.py`.
- The T5 summariser works best on English text. Translation to English happens automatically before summarisation.
