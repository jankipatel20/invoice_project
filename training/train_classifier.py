"""
train_classifier.py  –  Train MobileNetV3 invoice classifier
Dataset layout expected:
  invoice_images/
    arabic/
    french/
    german/
    japanese/
    mandarin/
    spanish/
    not_invoice/       ← put non-invoice images here (if you have them)

Run:
  python train_classifier.py --data_root ./invoice_images --epochs 20
"""

import argparse
import random
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report
import numpy as np


# ─── Dataset ─────────────────────────────────────────────────────────────────
INVOICE_LANGS = {"arabic", "french", "german", "japanese", "mandarin", "spanish", "english"}

class InvoiceDataset(Dataset):
    """
    Dynamically builds (path, label) pairs from the folder structure.
    label 1 = invoice,  label 0 = not_invoice
    """

    TRAIN_TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    EVAL_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self, data_root: Path, split: str = "train", val_ratio: float = 0.15):
        self.transform = self.TRAIN_TRANSFORM if split == "train" else self.EVAL_TRANSFORM
        all_samples = []

        # Positive samples (invoice)
        for lang_dir in data_root.iterdir():
            if lang_dir.is_dir() and lang_dir.name.lower() in INVOICE_LANGS:
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"):
                    for img_path in lang_dir.glob(ext):
                        all_samples.append((img_path, 1))

        # Negative samples (not invoice)
        neg_dir = data_root / "not_invoice"
        if neg_dir.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_path in neg_dir.glob(ext):
                    all_samples.append((img_path, 0))
        else:
            print("[Dataset] WARNING: No 'not_invoice' folder found. "
                  "Classifier will only see positive samples – accuracy may be poor.")

        random.shuffle(all_samples)
        n_val = int(len(all_samples) * val_ratio)
        if split == "train":
            self.samples = all_samples[n_val:]
        else:
            self.samples = all_samples[:n_val]

        pos = sum(1 for _, l in self.samples if l == 1)
        neg = len(self.samples) - pos
        print(f"[Dataset] {split}: {len(self.samples)} samples  "
              f"(invoice={pos}, not_invoice={neg})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=255)
        return self.transform(img), label


# ─── Model ───────────────────────────────────────────────────────────────────
def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model   = models.mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    return model


# ─── Training loop ────────────────────────────────────────────────────────────
def train(args):
    data_root = Path(args.data_root)
    out_path  = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_path = out_path / "invoice_classifier.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    train_ds = InvoiceDataset(data_root, split="train")
    val_ds   = InvoiceDataset(data_root, split="val")

    if len(train_ds) == 0:
        print("[Train] ERROR: No training images found. Check --data_root path.")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=True)

    model     = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)

        train_acc  = correct / max(total, 1)
        train_loss = total_loss / max(total, 1)

        # ── validate ──
        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_labels  = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total   += len(labels)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_acc = val_correct / max(val_total, 1)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_acc={val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Saved best model (val_acc={val_acc:.3f})")

    print(f"\n[Train] Best val accuracy: {best_val_acc:.3f}")
    print(f"[Train] Model saved → {model_path}")

    # Final classification report
    if all_labels:
        print("\n" + classification_report(all_labels, all_preds,
              target_names=["not_invoice", "invoice"]))


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train invoice classifier")
    parser.add_argument("--data_root",  default="./invoice_images",
                        help="Root folder containing language subfolders")
    parser.add_argument("--out_dir",    default="./models",
                        help="Where to save the trained model")
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--workers",    type=int, default=4)
    args = parser.parse_args()
    train(args)
