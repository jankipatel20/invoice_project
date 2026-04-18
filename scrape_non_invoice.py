"""
scrape_non_invoice.py  –  Scrape non-invoice images from multiple free sources
Targets ~600-700 images across diverse categories to balance the invoice dataset
(80-100 invoices × 6 languages = ~540 invoice images total)

Sources used (no API key required):
  1. Unsplash Source  – random photos by category
  2. Picsum Photos    – random stock photos
  3. Lorem Flickr     – category-tagged random images
  4. Wikipedia Commons (via API)  – document/screenshot images

Usage:
  python scrape_non_invoice.py --out_dir ./invoice_images/not_invoice --count 700
"""

import argparse
import os
import time
import random
import hashlib
import logging
import urllib.request
import urllib.parse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Image categories to scrape (clearly NOT invoices) ────────────────────────
UNSPLASH_CATEGORIES = [
    "food", "travel", "nature", "technology", "architecture",
    "people", "animals", "sports", "cars", "fashion",
    "music", "art", "city", "beach", "mountain",
    "office", "coffee", "books", "flowers", "sky",
]

LOREM_FLICKR_CATEGORIES = [
    "cats", "dogs", "landscape", "portrait", "street",
    "food", "travel", "abstract", "vintage", "urban",
]

# Wikipedia Commons search terms that yield misc non-invoice images
WIKI_COMMONS_QUERIES = [
    "photograph person outdoor", "landscape photograph",
    "food photograph", "building exterior", "animal photograph",
    "map diagram", "chart graph statistics",        # intentional: documents ≠ invoices
    "screenshot software interface",
    "newspaper front page",                          # documents but not invoices
    "menu restaurant",                               # similar layout, NOT invoice
    "certificate diploma",                           # similar layout, NOT invoice
    "flyer advertisement",
    "poster announcement",
    "brochure pamphlet",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ── URL generators ────────────────────────────────────────────────────────────
def unsplash_urls(count: int) -> list[str]:
    """Unsplash Source – free, no API key, random image per category."""
    urls = []
    per_cat = max(1, count // len(UNSPLASH_CATEGORIES))
    sizes   = [(640, 480), (800, 600), (1024, 768), (900, 600)]
    for cat in UNSPLASH_CATEGORIES:
        for _ in range(per_cat):
            w, h = random.choice(sizes)
            # ?sig= forces a unique image each time
            sig = random.randint(1, 99999)
            urls.append(f"https://source.unsplash.com/{w}x{h}/?{cat}&sig={sig}")
    return urls


def picsum_urls(count: int) -> list[str]:
    """Lorem Picsum – totally random photos, no duplication via seed."""
    seeds = random.sample(range(1, 10000), min(count, 9999))
    return [f"https://picsum.photos/seed/{s}/800/600" for s in seeds]


def lorem_flickr_urls(count: int) -> list[str]:
    """LoremFlickr – category tagged random images."""
    urls = []
    per_cat = max(1, count // len(LOREM_FLICKR_CATEGORIES))
    for cat in LOREM_FLICKR_CATEGORIES:
        for _ in range(per_cat):
            lock = random.randint(1, 99999)
            urls.append(f"https://loremflickr.com/640/480/{cat}?lock={lock}")
    return urls


def wiki_commons_urls(count: int) -> list[str]:
    """Wikipedia Commons API – fetch image file URLs by search term."""
    urls  = []
    per_q = max(2, count // len(WIKI_COMMONS_QUERIES))
    for query in WIKI_COMMONS_QUERIES:
        try:
            api = (
                "https://commons.wikimedia.org/w/api.php"
                "?action=query&generator=search&gsrnamespace=6"
                f"&gsrsearch=filetype:bitmap+{urllib.parse.quote(query)}"
                f"&gsrlimit={per_q}&prop=imageinfo&iiprop=url&format=json"
            )
            req  = urllib.request.Request(api, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data  = json.loads(resp.read())
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                info = page.get("imageinfo", [{}])[0]
                url  = info.get("url", "")
                if url.lower().endswith((".jpg", ".jpeg", ".png")):
                    urls.append(url)
            time.sleep(0.3)   # Be polite to Wikimedia
        except Exception as e:
            log.warning(f"Wiki Commons query '{query}' failed: {e}")
    return urls


# ── Downloader ────────────────────────────────────────────────────────────────
def download_one(url: str, out_dir: Path, idx: int) -> bool:
    """Download a single URL and save with a content-hash filename."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data        = resp.read()
            content_type = resp.headers.get("Content-Type", "")

        if not data or len(data) < 5000:          # Skip tiny/broken responses
            return False
        if "image" not in content_type.lower():
            return False

        # Determine extension
        if "png" in content_type:
            ext = ".png"
        else:
            ext = ".jpg"

        # Use content hash to deduplicate
        name = hashlib.md5(data).hexdigest()[:16] + ext
        path = out_dir / name
        if path.exists():
            return False   # Already downloaded (duplicate)

        path.write_bytes(data)
        log.info(f"[{idx:04d}] ✓  {path.name}  ({len(data)//1024}KB)")
        return True

    except Exception as e:
        log.debug(f"[{idx:04d}] ✗  {url[:60]}…  ({e})")
        return False


def scrape(out_dir: Path, total: int = 700, workers: int = 8):
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Target: {total} non-invoice images → {out_dir}")

    # Build URL pool (overshoot target by 2× to account for failures)
    n = total * 2
    pool = (
        unsplash_urls(n // 3) +
        picsum_urls(n // 4) +
        lorem_flickr_urls(n // 6) +
        wiki_commons_urls(n // 6)
    )
    random.shuffle(pool)
    log.info(f"URL pool: {len(pool)} candidates")

    downloaded = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_one, url, out_dir, i): url
                   for i, url in enumerate(pool)}
        for fut in as_completed(futures):
            if fut.result():
                downloaded += 1
            if downloaded >= total:
                # Cancel pending futures once we hit the target
                for f in futures:
                    f.cancel()
                break

    existing = len(list(out_dir.glob("*.jpg"))) + len(list(out_dir.glob("*.png")))
    log.info(f"\n✓  Done.  {existing} images in {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape non-invoice images")
    parser.add_argument("--out_dir", default="./invoice_images/not_invoice",
                        help="Output folder (default: invoice_images/not_invoice)")
    parser.add_argument("--count",   type=int, default=700,
                        help="Target number of images (default: 700)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel download threads (default: 8)")
    args = parser.parse_args()
    scrape(Path(args.out_dir), total=args.count, workers=args.workers)
