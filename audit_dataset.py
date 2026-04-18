"""
audit_dataset.py  –  Print a summary table of your invoice_images dataset.
Run this before training to know exactly where you stand.

Usage:  python audit_dataset.py --data_root ./invoice_images
"""

import argparse
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def audit(data_root: Path):
    print(f"\n{'─'*50}")
    print(f"  Dataset audit: {data_root.resolve()}")
    print(f"{'─'*50}")
    print(f"  {'Folder':<20} {'Images':>8}  {'Synthetic':>10}  {'Real':>8}")
    print(f"  {'─'*44}")

    total_real = 0
    total_synth = 0
    total_all   = 0

    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        imgs  = [p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS]
        synth = [p for p in imgs if "synthetic_" in p.name]
        real  = len(imgs) - len(synth)
        print(f"  {d.name:<20} {len(imgs):>8}  {len(synth):>10}  {real:>8}")
        total_real  += real
        total_synth += len(synth)
        total_all   += len(imgs)

    print(f"  {'─'*44}")
    print(f"  {'TOTAL':<20} {total_all:>8}  {total_synth:>10}  {total_real:>8}")
    print(f"\n  Recommendation:")
    print(f"  • not_invoice/ should have ≥ {total_real} images to match real invoice count")
    print(f"  • Run GAN to augment any language below 200 images")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./invoice_images")
    args = parser.parse_args()
    audit(Path(args.data_root))
