"""
Non-Contrast CT Segmentation — Training Only (Phase 1 + Phase 2)
=================================================================
Trains a SegResNet model to segment Aorta + Pulmonary Artery on non-contrast CT.

Phase 1: Joint training on contrast + non-contrast data with contrast-dampening augmentation.
Phase 2: Fine-tuning exclusively on non-contrast data at 10x lower learning rate.

Saves checkpoints to WORK_DIR after each phase.

Usage:
    Kaggle:  !python train_phases.py
    Colab:   !python train_phases.py
    Local:   python train_phases.py
"""

# =============================================================
# Environment detection & dependencies
# =============================================================

import os
import subprocess
import sys
from pathlib import Path
import json

IS_KAGGLE = (
    "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    or "KAGGLE_URL_BASE" in os.environ
    or Path("/kaggle/input").exists()
    or Path("/kaggle/working").exists()
)
IS_COLAB = (
    not IS_KAGGLE
    and (
        "COLAB_GPU" in os.environ
        or "COLAB_RELEASE_TAG" in os.environ
        or Path("/var/colab/hostname").exists()
    )
)
IS_LOCAL = not (IS_COLAB or IS_KAGGLE)

if IS_COLAB or IS_KAGGLE:
    env_name = "Kaggle" if IS_KAGGLE else "Google Colab"
    print(f"Running on {env_name} -- installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "monai[nibabel,tqdm,itk]", "nibabel", "simpleitk",
        "scikit-image", "scikit-learn", "matplotlib", "tqdm", "pandas",
    ])
    print("[OK] Dependencies ready.\n")

# =============================================================
# Imports
# =============================================================

import numpy as np
import torch
from torch.utils.data import DataLoader

import monai
from monai.utils import set_determinism
from monai.data import Dataset, list_data_collate, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandShiftIntensityd, RandGaussianNoised,
    RandAdjustContrastd, RandAffined, MapLabelValued, EnsureTyped,
    AsDiscrete, MapTransform,
)
from monai.networks.nets import SegResNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

try:
    from monai.losses import HausdorffDTLoss
    HAS_HAUSDORFF = True
    print("[OK] HausdorffDTLoss available.")
except ImportError:
    HAS_HAUSDORFF = False
    print("[WARN] HausdorffDTLoss not available - using DiceCELoss only.")

# =============================================================
# Paths
# =============================================================

if IS_KAGGLE:
    WORK_DIR = Path("/kaggle/working/data")
    SEARCH_ROOTS = [Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for p in kaggle_input.iterdir():
            if p.is_dir():
                SEARCH_ROOTS.append(p)
                try:
                    for sub in p.iterdir():
                        if sub.is_dir():
                            SEARCH_ROOTS.append(sub)
                            for sub2 in sub.iterdir():
                                if sub2.is_dir():
                                    SEARCH_ROOTS.append(sub2)
                except Exception:
                    pass
elif IS_COLAB:
    try:
        from google.colab import drive
        if Path("/var/colab/hostname").exists() or "COLAB_RELEASE_TAG" in os.environ:
            drive.mount("/content/drive", force_remount=False)
    except Exception as e:
        print(f"  [WARN] Drive mount skipped: {e}")
    PROJECT_ROOT = Path("/content/drive/MyDrive/ai_healthcarehackathon")
    WORK_DIR = PROJECT_ROOT / "data"
    SEARCH_ROOTS = [PROJECT_ROOT, Path("/content/drive/MyDrive"), Path("/content"), Path.cwd()]
else:
    PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
    WORK_DIR = PROJECT_ROOT / "data"
    SEARCH_ROOTS = [PROJECT_ROOT, Path.cwd()]

WORK_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================
# Configuration
# =============================================================

SPACING = (2.5, 2.5, 2.5)
HU_MIN, HU_MAX = -200.0, 400.0
PATCH_SIZE = (64, 64, 64)
NUM_CLASSES = 3  # background, pulmonary_artery, aorta
LABEL_VALUE_MAP = {"pulmonary_artery_raw_value": 1, "aorta_raw_value": 2}

BATCH_SIZE = 4 if torch.cuda.is_available() else 2
NUM_WORKERS = 2 if ((IS_COLAB or IS_KAGGLE) and torch.cuda.is_available()) else 0
VAL_INTERVAL = 5
RANDOM_SEED = 42

# Phase 1: Joint training
PHASE1_EPOCHS = 100
PHASE1_LR = 1e-4

# Phase 2: NC-only fine-tuning
PHASE2_EPOCHS = 50
PHASE2_LR = 1e-5
EARLY_STOP_PATIENCE = 20

set_determinism(seed=RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"MONAI {monai.__version__} | Torch {torch.__version__} | Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
print()

# =============================================================
# Section 5A — Contrast-dampening augmentation
# =============================================================

class RandContrastDampend(MapTransform):
    """With prob p, dampen HU inside vessel labels toward soft-tissue range,
    simulating non-contrast appearance on contrast-CT training data.
    Must run AFTER Spacingd and BEFORE ScaleIntensityRanged."""

    def __init__(self, keys, label_key="label", prob=0.4,
                 target_hu=50.0, blend_range=(0.5, 1.0)):
        super().__init__(keys)
        self.label_key = label_key
        self.prob = prob
        self.target_hu = target_hu
        self.blend_range = blend_range

    def __call__(self, data):
        d = dict(data)
        if np.random.rand() < self.prob:
            img = d[self.keys[0]]
            lbl = d[self.label_key]
            mask = (lbl > 0)
            alpha = np.random.uniform(*self.blend_range)
            img_np = img.clone() if hasattr(img, "clone") else img.copy()
            img_np[mask] = (1 - alpha) * img_np[mask] + alpha * self.target_hu
            d[self.keys[0]] = img_np
        return d


# =============================================================
# Folder & file discovery
# =============================================================

def find_matching_dirs(pattern, roots=None):
    if roots is None:
        roots = SEARCH_ROOTS
    found, seen = [], set()
    for r in roots:
        r = Path(r)
        if not r.exists():
            continue
        try:
            if pattern.lower() in r.name.lower() and r.is_dir():
                res = r.resolve()
                if res not in seen:
                    seen.add(res); found.append(r)
            direct = r / pattern
            if direct.is_dir():
                res = direct.resolve()
                if res not in seen:
                    seen.add(res); found.append(direct)
            for c in sorted(r.glob(f"*{pattern}*")):
                if c.is_dir():
                    inner = c / pattern
                    target = inner if inner.is_dir() else c
                    res = target.resolve()
                    if res not in seen:
                        seen.add(res); found.append(target)
        except Exception:
            pass
    return found


def find_baseline_checkpoint(roots=None):
    if roots is None:
        roots = SEARCH_ROOTS
    for r in roots:
        r = Path(r)
        if not r.exists():
            continue
        for c in [r / "data" / "baseline_model.pt", r / "baseline_model.pt"]:
            if c.is_file():
                return c
    for r in roots:
        r = Path(r)
        if not r.exists():
            continue
        try:
            for pt in r.rglob("*.pt"):
                if "baseline" in pt.name.lower():
                    return pt
        except Exception:
            pass
    return None


def extract_case_id(filename, label_suffix="_label"):
    name = filename
    for ext in [".nii.gz", ".nii", ".gz"]:
        if name.lower().endswith(ext):
            name = name[:-len(ext)]; break
    name = name.replace("(1)", "").replace("(2)", "")
    if name.lower().endswith(label_suffix.lower()):
        name = name[:-len(label_suffix)]
    return name.strip()


def build_file_list(images_dirs, labels_dirs, label_suffix="_label"):
    if isinstance(images_dirs, (str, Path)):
        images_dirs = [Path(images_dirs)]
    if isinstance(labels_dirs, (str, Path)):
        labels_dirs = [Path(labels_dirs)]
    images_dirs = [Path(d) for d in images_dirs if d and Path(d).exists()]
    labels_dirs = [Path(d) for d in labels_dirs if d and Path(d).exists()]

    all_images = {}
    for idir in images_dirs:
        for p in sorted(idir.rglob("*")):
            if p.is_file() and ".nii" in p.name.lower() and "_label" not in p.name.lower():
                all_images[extract_case_id(p.name, label_suffix)] = p

    all_labels = {}
    for ldir in labels_dirs:
        for p in sorted(ldir.rglob("*")):
            if p.is_file() and ".nii" in p.name.lower():
                all_labels[extract_case_id(p.name, label_suffix)] = p

    print(f"  [Scan] {len(all_images)} image(s), {len(all_labels)} label(s)")
    pairs = []
    for cid, ip in sorted(all_images.items()):
        if cid in all_labels:
            pairs.append({"image": str(ip), "label": str(all_labels[cid])})
    return pairs


# =============================================================
# Transforms
# =============================================================

_label_map_args = dict(
    keys=["label"],
    orig_labels=[0, LABEL_VALUE_MAP["pulmonary_artery_raw_value"], LABEL_VALUE_MAP["aorta_raw_value"]],
    target_labels=[0, 1, 2],
    dtype=np.uint8,
)

train_transforms = Compose([
    LoadImaged(keys=["image", "label"], image_only=True),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=SPACING, mode=("bilinear", "nearest")),
    MapLabelValued(**_label_map_args),
    # Section 5A: contrast dampening — AFTER Spacingd, BEFORE ScaleIntensityRanged
    RandContrastDampend(keys=["image"], label_key="label", prob=0.4, target_hu=50.0, blend_range=(0.5, 1.0)),
    ScaleIntensityRanged(keys=["image"], a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=PATCH_SIZE,
                           pos=2, neg=1, num_samples=2, image_key="image", image_threshold=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    RandGaussianNoised(keys=["image"], prob=0.3, std=0.03),
    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.3)),
    RandAffined(keys=["image", "label"], prob=0.3, rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1), mode=("bilinear", "nearest")),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"], image_only=True),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=SPACING, mode=("bilinear", "nearest")),
    MapLabelValued(**_label_map_args),
    ScaleIntensityRanged(keys=["image"], a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    EnsureTyped(keys=["image", "label"]),
])

# =============================================================
# Model + Loss
# =============================================================

def build_model():
    return SegResNet(spatial_dims=3, in_channels=1, out_channels=NUM_CLASSES,
                     init_filters=16, dropout_prob=0.2).to(DEVICE)

def build_loss():
    dice_ce = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    hausdorff = None
    if HAS_HAUSDORFF:
        hausdorff = HausdorffDTLoss(to_onehot_y=True, softmax=True, include_background=False)
    return dice_ce, hausdorff

dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)])
post_label = Compose([AsDiscrete(to_onehot=NUM_CLASSES)])

# =============================================================
# Training loop
# =============================================================

def run_training(model, train_loader, val_loader, epochs, checkpoint_path,
                 lr=1e-4, tag="", early_stop_patience=None):
    dice_ce_loss, hausdorff_loss = build_loss()
    hausdorff_weight = 0.3

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_metric = -1.0
    no_improve_count = 0
    train_loss_history = []
    val_dice_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            inputs = batch["image"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = dice_ce_loss(outputs, labels)
                if hausdorff_loss is not None:
                    try:
                        loss = loss + hausdorff_weight * hausdorff_loss(outputs, labels)
                    except Exception:
                        pass

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        epoch_loss /= max(len(train_loader), 1)
        scheduler.step()
        train_loss_history.append(epoch_loss)

        # Validation
        if (epoch + 1) % VAL_INTERVAL == 0 or epoch == epochs - 1:
            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(DEVICE)
                    labels_batch = batch["label"].to(DEVICE)
                    outputs = sliding_window_inference(images, PATCH_SIZE, sw_batch_size=2,
                                                      predictor=model, overlap=0.5)
                    outputs = [post_pred(x) for x in decollate_batch(outputs)]
                    labels_batch = [post_label(x) for x in decollate_batch(labels_batch)]
                    dice_metric(y_pred=outputs, y=labels_batch)

            per_class = dice_metric.aggregate()
            mean_dice = per_class.mean().item()
            pa_dice = per_class[0].item()
            aorta_dice = per_class[1].item()
            val_dice_history.append((epoch + 1, mean_dice, pa_dice, aorta_dice))

            print(f"  [{tag}] Epoch {epoch+1:03d}/{epochs} | "
                  f"Loss: {epoch_loss:.4f} | "
                  f"Dice: {mean_dice:.4f} (PA: {pa_dice:.4f}, Aorta: {aorta_dice:.4f})")

            if mean_dice > best_metric:
                best_metric = mean_dice
                torch.save(model.state_dict(), checkpoint_path)
                no_improve_count = 0
                print(f"  >> New best! Saved to {checkpoint_path}")
            else:
                no_improve_count += VAL_INTERVAL

            if early_stop_patience and no_improve_count >= early_stop_patience:
                print(f"  [STOP] No improvement for {no_improve_count} epochs.")
                break
        else:
            print(f"  [{tag}] Epoch {epoch+1:03d}/{epochs} | Loss: {epoch_loss:.4f}", end="\r")

    print()
    return train_loss_history, val_dice_history, best_metric


# =============================================================
# Main
# =============================================================

def main():
    print("=" * 60)
    print("  PHASE 1 + PHASE 2 TRAINING")
    print("=" * 60)

    # --- Locate data ---
    print("\nLocating data...")
    contrast_img_dirs = find_matching_dirs("unused_contrast_images")
    contrast_lbl_dirs = find_matching_dirs("unused_contrast_labels")
    nc_img_dirs = find_matching_dirs("non_contrast_images")
    nc_lbl_dirs = find_matching_dirs("non_contrast_labels")

    print(f"  Contrast image dirs: {[str(d) for d in contrast_img_dirs]}")
    print(f"  Contrast label dirs: {[str(d) for d in contrast_lbl_dirs]}")
    print(f"  NC image dirs:       {[str(d) for d in nc_img_dirs]}")
    print(f"  NC label dirs:       {[str(d) for d in nc_lbl_dirs]}")

    baseline_ckpt = find_baseline_checkpoint()
    print(f"  Baseline model:      {baseline_ckpt}")
    print()

    # --- Build file lists ---
    print("Building file lists...")
    contrast_files = []
    if contrast_img_dirs and contrast_lbl_dirs:
        contrast_files = build_file_list(contrast_img_dirs, contrast_lbl_dirs)
        print(f"  Contrast pairs: {len(contrast_files)}")

    nc_files = build_file_list(nc_img_dirs, nc_lbl_dirs)
    print(f"  NC pairs:       {len(nc_files)}")

    if len(nc_files) == 0:
        print("ERROR: No non-contrast pairs found.")
        sys.exit(1)

    # --- Split NC data ---
    rng = np.random.default_rng(RANDOM_SEED)
    nc_indices = np.arange(len(nc_files))
    rng.shuffle(nc_indices)
    n_val = max(1, int(len(nc_files) * 0.22))
    nc_train_files = [nc_files[i] for i in nc_indices[n_val:]]
    nc_val_files = [nc_files[i] for i in nc_indices[:n_val]]
    print(f"\n  NC train: {len(nc_train_files)}, NC val: {len(nc_val_files)}")

    phase1_train_files = contrast_files + nc_train_files
    print(f"  Phase 1 train: {len(phase1_train_files)} ({len(contrast_files)} contrast + {len(nc_train_files)} NC)")

    # --- Model ---
    model = build_model()
    if baseline_ckpt:
        model.load_state_dict(torch.load(str(baseline_ckpt), map_location=DEVICE, weights_only=True))
        print(f"\n  [OK] Loaded baseline: {baseline_ckpt.name}")

    # --- Phase 1 ---
    print("\n" + "=" * 60)
    print(f"  PHASE 1: Joint Training | {PHASE1_EPOCHS} epochs | LR={PHASE1_LR}")
    print("=" * 60)

    p1_train_ds = Dataset(data=phase1_train_files, transform=train_transforms)
    nc_val_ds = Dataset(data=nc_val_files, transform=val_transforms)
    p1_loader = DataLoader(p1_train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, collate_fn=list_data_collate,
                           pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(nc_val_ds, batch_size=1, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=list_data_collate,
                            pin_memory=torch.cuda.is_available())

    p1_ckpt = WORK_DIR / "phase1_best.pt"
    _, p1_history, p1_best = run_training(model, p1_loader, val_loader,
                                          epochs=PHASE1_EPOCHS, checkpoint_path=p1_ckpt,
                                          lr=PHASE1_LR, tag="Phase1")
    print(f"  Phase 1 best NC Dice: {p1_best:.4f}")

    # --- Phase 2 ---
    print("\n" + "=" * 60)
    print(f"  PHASE 2: NC Fine-Tuning | {PHASE2_EPOCHS} epochs | LR={PHASE2_LR}")
    print("=" * 60)

    model.load_state_dict(torch.load(str(p1_ckpt), map_location=DEVICE, weights_only=True))
    print("  [OK] Loaded Phase 1 best checkpoint.")

    p2_train_ds = Dataset(data=nc_train_files, transform=train_transforms)
    p2_loader = DataLoader(p2_train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, collate_fn=list_data_collate,
                           pin_memory=torch.cuda.is_available())

    p2_ckpt = WORK_DIR / "phase2_nc_finetuned.pt"
    _, p2_history, p2_best = run_training(model, p2_loader, val_loader,
                                          epochs=PHASE2_EPOCHS, checkpoint_path=p2_ckpt,
                                          lr=PHASE2_LR, tag="Phase2",
                                          early_stop_patience=EARLY_STOP_PATIENCE)
    print(f"  Phase 2 best NC Dice: {p2_best:.4f}")

    # --- Save results ---
    results = {
        "phase1_best_dice": float(p1_best),
        "phase2_best_dice": float(p2_best),
        "best_phase": "Phase 2" if p2_best >= p1_best else "Phase 1",
        "best_dice": float(max(p1_best, p2_best)),
        "phase1_val_history": p1_history,
        "phase2_val_history": p2_history,
    }
    results_path = WORK_DIR / "nc_training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Phase 1 best: {p1_best:.4f}")
    print(f"  Phase 2 best: {p2_best:.4f}")
    print(f"  Checkpoints:  {p1_ckpt}")
    print(f"                {p2_ckpt}")
    print(f"  Results:      {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
