"""
train_gan.py  –  DCGAN for synthetic invoice image augmentation
Trains one GAN per language folder, generates synthetic invoices to augment
the dataset up to a target count per language.

Usage:
  # Train on all languages and generate to 200 images each:
  python train_gan.py --data_root ./invoice_images --target 200 --epochs 300

  # Single language:
  python train_gan.py --data_root ./invoice_images --lang french --target 200

Architecture:
  Generator     : 100-dim noise → 64×64 RGB image  (5 ConvTranspose2d layers)
  Discriminator : 64×64 RGB → real/fake score       (5 Conv2d layers)
  Loss          : BCEWithLogitsLoss + label smoothing
  Training      : Two-step alternating G/D updates
"""

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
SUPPORTED_LANGS = ["arabic", "french", "german", "japanese", "mandarin", "spanish", "english"]
Z_DIM       = 100
IMG_SIZE    = 64        # DCGAN standard; increase to 128 for higher quality (slower)
NGF         = 64        # Generator feature maps
NDF         = 64        # Discriminator feature maps
LR          = 2e-4
BETA1       = 0.5       # Adam β₁ (standard for GANs)
LABEL_REAL  = 0.9       # Label smoothing for real samples
LABEL_FAKE  = 0.0


# ── Dataset ───────────────────────────────────────────────────────────────────
class InvoiceLangDataset(Dataset):
    TRANSFORM = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    def __init__(self, lang_dir: Path):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        self.paths = [p for p in lang_dir.iterdir()
                      if p.suffix.lower() in exts]
        if not self.paths:
            raise ValueError(f"No images found in {lang_dir}")
        print(f"  [Dataset] {lang_dir.name}: {len(self.paths)} real images")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), 255)
        return self.TRANSFORM(img)


# ── Generator ─────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, z_dim: int = Z_DIM, ngf: int = NGF):
        super().__init__()
        self.net = nn.Sequential(
            # 1×1 → 4×4
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            # 4×4 → 8×8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            # 8×8 → 16×16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            # 16×16 → 32×32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            # 32×32 → 64×64
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        return self.net(z)


# ── Discriminator ─────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self, ndf: int = NDF):
        super().__init__()
        self.net = nn.Sequential(
            # 64×64 → 32×32
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32×32 → 16×16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
            # 16×16 → 8×8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),
            # 8×8 → 4×4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True),
            # 4×4 → 1×1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x).view(-1)


# ── Training ──────────────────────────────────────────────────────────────────
def train_one_lang(
    lang_dir: Path,
    out_root: Path,
    target: int = 200,
    epochs: int  = 300,
    batch_size: int = 16,
    save_every: int = 50,
    device = None,
):
    lang = lang_dir.name
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'─'*55}")
    print(f"  GAN training: {lang.upper()}   device={device}")
    print(f"{'─'*55}")

    # Directories
    model_dir   = out_root / "gan_models"   / lang
    synth_dir   = lang_dir                                   # Save back into lang folder
    sample_dir  = out_root / "gan_samples"  / lang
    for d in [model_dir, sample_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Count existing images
    existing = len([p for p in lang_dir.iterdir()
                    if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    to_generate = max(0, target - existing)
    if to_generate <= 0:
        print(f"  Already have {existing} images (target={target}). Skipping.")
        return

    print(f"  Will generate {to_generate} synthetic images (have {existing}, want {target})")

    # Dataset & loader
    dataset = InvoiceLangDataset(lang_dir)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=True, num_workers=2, pin_memory=True,
                         drop_last=True)

    # Models
    G = Generator().to(device)
    D = Discriminator().to(device)

    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    # Fixed noise for monitoring
    fixed_noise = torch.randn(16, Z_DIM, 1, 1, device=device)

    g_losses, d_losses = [], []
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_g, epoch_d = 0.0, 0.0

        for real_imgs in loader:
            real_imgs = real_imgs.to(device)
            bs        = real_imgs.size(0)

            # ── Train Discriminator ──
            D.zero_grad()
            real_labels = torch.full((bs,), LABEL_REAL, device=device)
            fake_labels = torch.full((bs,), LABEL_FAKE, device=device)

            d_real  = D(real_imgs)
            loss_dr = criterion(d_real, real_labels)

            noise  = torch.randn(bs, Z_DIM, 1, 1, device=device)
            fake   = G(noise).detach()
            d_fake = D(fake)
            loss_df = criterion(d_fake, fake_labels)

            loss_D = loss_dr + loss_df
            loss_D.backward()
            opt_D.step()
            epoch_d += loss_D.item()

            # ── Train Generator ──
            G.zero_grad()
            noise    = torch.randn(bs, Z_DIM, 1, 1, device=device)
            fake     = G(noise)
            d_fake2  = D(fake)
            # Generator wants D to output REAL for its fakes
            loss_G   = criterion(d_fake2, real_labels)
            loss_G.backward()
            opt_G.step()
            epoch_g += loss_G.item()

        n_batches = max(len(loader), 1)
        g_losses.append(epoch_g / n_batches)
        d_losses.append(epoch_d / n_batches)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.perf_counter() - t0
            print(f"  Epoch {epoch:4d}/{epochs}  "
                  f"G={g_losses[-1]:.4f}  D={d_losses[-1]:.4f}  "
                  f"({elapsed:.0f}s)")

        # Save sample grid
        if epoch % save_every == 0 or epoch == epochs:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_noise)
            save_image(samples * 0.5 + 0.5,
                       sample_dir / f"epoch_{epoch:04d}.png",
                       nrow=4)
            G.train()

    # ── Save model ──
    torch.save({"G": G.state_dict(), "D": D.state_dict()},
               model_dir / "checkpoint.pth")
    print(f"  Model saved → {model_dir}/checkpoint.pth")

    # ── Generate synthetic images ──
    print(f"  Generating {to_generate} synthetic images…")
    G.eval()
    generated = 0
    batch_gen  = 32
    with torch.no_grad():
        while generated < to_generate:
            n     = min(batch_gen, to_generate - generated)
            noise = torch.randn(n, Z_DIM, 1, 1, device=device)
            imgs  = (G(noise) * 0.5 + 0.5).clamp(0, 1)
            for i, img_t in enumerate(imgs):
                fname = synth_dir / f"synthetic_{lang}_{generated + i:04d}.jpg"
                save_image(img_t, fname)
            generated += n

    total_now = len([p for p in lang_dir.iterdir()
                     if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    print(f"  ✓  {lang}: {total_now} total images (added {generated} synthetic)")

    # Save loss curves
    _save_loss_plot(g_losses, d_losses, lang, sample_dir)


def _save_loss_plot(g_losses, d_losses, lang, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(g_losses, label="Generator",     color="#f0c040")
        ax.plot(d_losses, label="Discriminator", color="#3af0a0")
        ax.set_title(f"GAN Losses – {lang}", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / "loss_curve.png", dpi=120)
        plt.close()
    except Exception:
        pass


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DCGAN per invoice language")
    parser.add_argument("--data_root", default="./invoice_images",
                        help="Root folder with language subfolders")
    parser.add_argument("--out_root",  default="./invoice_images",
                        help="Where to save GAN models/samples (default: same as data_root)")
    parser.add_argument("--target",    type=int, default=200,
                        help="Target images per language after GAN augmentation (default: 200)")
    parser.add_argument("--epochs",    type=int, default=300,
                        help="Training epochs per language (default: 300)")
    parser.add_argument("--batch",     type=int, default=16)
    parser.add_argument("--lang",      default=None,
                        help="Train only this language (e.g. 'french'). Default: all.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root  = Path(args.out_root)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.lang:
        langs = [args.lang]
    else:
        langs = [d.name for d in data_root.iterdir()
                 if d.is_dir() and d.name.lower() in SUPPORTED_LANGS]

    print(f"Languages to process: {langs}")

    for lang in langs:
        lang_dir = data_root / lang
        if not lang_dir.exists():
            print(f"  SKIP: {lang_dir} not found")
            continue
        train_one_lang(
            lang_dir   = lang_dir,
            out_root   = out_root,
            target     = args.target,
            epochs     = args.epochs,
            batch_size = args.batch,
            device     = device,
        )

    print("\n✅  All languages done.")
