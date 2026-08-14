"""
Non-Contrast CT Segmentation Training Script (Colab + Local)
=============================================================
Implements the strategy from non_contrast_ct_segmentation_strategy.md:
  A. Contrast-dampening augmentation (RandContrastDampend)
  B. Joint training on contrast + non-contrast data, then NC fine-tuning
  C. Separate non-contrast validation with per-class Dice
  D. Combined DiceCELoss + HausdorffDTLoss

Starts from the pre-trained baseline_model.pt (Dice=0.8756 on contrast CT).
Resolution reduced (spacing 2.5mm, patch 64^3) for faster processing.

Usage:
    Google Colab: Copy cells into a notebook, or upload and run:
                  !python train_noncontrast.py
    Local:        python train_noncontrast.py
"""

# =============================================================
# Step 0: Environment detection & package installation
# =============================================================

import os
import subprocess
import sys
from pathlib import Path

# Detect runtime environment: Kaggle, Google Colab, or Local PC
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
    print("=" * 60)
    print(f"  Running on {env_name} -- installing / checking dependencies...")
    print("=" * 60)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "monai[nibabel,tqdm,itk]", "nibabel", "simpleitk",
        "scikit-image", "scikit-learn", "matplotlib", "tqdm", "pandas",
    ])
    print("  [OK] Dependencies ready.\n")

# =============================================================
# Step 1: Imports
# =============================================================

import json
import glob

import numpy as np
import matplotlib
if IS_LOCAL:
    matplotlib.use("Agg")  # Non-interactive backend for saving PNGs locally
import matplotlib.pyplot as plt
from tqdm import tqdm

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

# Try to import HausdorffDTLoss for boundary-aware training (Section 5D)
try:
    from monai.losses import HausdorffDTLoss
    HAS_HAUSDORFF = True
    print("[OK] HausdorffDTLoss available - will use combined loss.")
except ImportError:
    HAS_HAUSDORFF = False
    print("[WARN] HausdorffDTLoss not available - using DiceCELoss only.")


# =============================================================
# Step 2: Environment-specific path setup
# =============================================================

if IS_KAGGLE:
    # On Kaggle, /kaggle/input is read-only dataset storage.
    # Output checkpoints and predictions MUST be written to /kaggle/working/data
    WORK_DIR = Path("/kaggle/working/data")
    SEARCH_ROOTS = [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path.cwd(),
    ]
    # Automatically scan all datasets mounted in /kaggle/input
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

    DRIVE_DATA_ROOT = Path("/content/drive/MyDrive/ai_healthcarehackathon")
    PROJECT_ROOT = DRIVE_DATA_ROOT
    WORK_DIR = PROJECT_ROOT / "data"
    SEARCH_ROOTS = [
        PROJECT_ROOT,
        Path("/content/drive/MyDrive"),
        Path("/content"),
        Path.cwd(),
    ]
else:
    PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
    WORK_DIR = PROJECT_ROOT / "data"
    SEARCH_ROOTS = [
        PROJECT_ROOT,
        Path.cwd(),
    ]

WORK_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================
# Step 3: Configuration
# =============================================================

# Reduced resolution for faster processing (original: 1.5mm, 96^3)
SPACING = (2.5, 2.5, 2.5)
HU_MIN, HU_MAX = -200.0, 400.0
PATCH_SIZE = (64, 64, 64)

CLASS_NAMES = ["background", "pulmonary_artery", "aorta"]
NUM_CLASSES = len(CLASS_NAMES)
LABEL_VALUE_MAP = {"pulmonary_artery_raw_value": 1, "aorta_raw_value": 2}

# Colab/Kaggle GPU has 16GB VRAM; on CPU / local PC use smaller batch size
BATCH_SIZE = 4 if torch.cuda.is_available() else 2
NUM_WORKERS = 2 if ((IS_COLAB or IS_KAGGLE) and torch.cuda.is_available()) else 0
VAL_INTERVAL = 5
RANDOM_SEED = 42

# Phase 1: Joint training (contrast + NC)
PHASE1_EPOCHS = 100
PHASE1_LR = 1e-4

# Phase 2: NC-only fine-tuning
PHASE2_EPOCHS = 50
PHASE2_LR = 1e-5
EARLY_STOP_PATIENCE = 20

set_determinism(seed=RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nMONAI version: {monai.__version__}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU memory: {gpu_mem:.1f} GB")
print(f"Device: {DEVICE}")
print(f"Resolution: spacing={SPACING}, patch={PATCH_SIZE}")
print(f"Batch size: {BATCH_SIZE}, Workers: {NUM_WORKERS}")
print()


# =============================================================
# Section 5A - Contrast-dampening augmentation
# =============================================================

class RandContrastDampend(MapTransform):
    """With prob p, dampen HU inside the vessel labels toward soft-tissue range,
    simulating a non-contrast appearance on contrast-CT training data.

    Must run AFTER Spacingd and BEFORE ScaleIntensityRanged (needs raw HU
    values and the label mask).
    """
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
# Auto-detect extracted folder paths across search roots
# =============================================================

def find_matching_dirs(pattern: str, roots=None):
    """Find all directories matching pattern across search roots,
    handling clean names, timestamped zips (-001, -002), and nested subfolders.
    Returns a list of unique Path objects."""
    if roots is None:
        roots = SEARCH_ROOTS

    found = []
    seen = set()

    for r in roots:
        r = Path(r)
        if not r.exists():
            continue

        try:
            # Check if root itself matches
            if pattern.lower() in r.name.lower() and r.is_dir():
                res = r.resolve()
                if res not in seen:
                    seen.add(res)
                    found.append(r)

            # Check direct subfolder: r / pattern
            direct = r / pattern
            if direct.is_dir():
                res = direct.resolve()
                if res not in seen:
                    seen.add(res)
                    found.append(direct)

            # Check glob match in r
            for candidate in sorted(r.glob(f"*{pattern}*")):
                if candidate.is_dir():
                    inner = candidate / pattern
                    target = inner if inner.is_dir() else candidate
                    res = target.resolve()
                    if res not in seen:
                        seen.add(res)
                        found.append(target)
                    # Also include candidate if inner was chosen
                    if inner.is_dir() and candidate.resolve() not in seen:
                        seen.add(candidate.resolve())
                        found.append(candidate)
        except Exception:
            pass

    return found


def find_folder(pattern: str, root=None):
    """Backwards-compatible helper to find the first matching directory."""
    roots = [root] if root else SEARCH_ROOTS
    dirs = find_matching_dirs(pattern, roots=roots)
    if dirs:
        return dirs[0]
    raise FileNotFoundError(
        f"Could not find folder matching '{pattern}' in {[str(r) for r in roots]}."
    )


def find_baseline_checkpoint(roots=None):
    """Find the pre-trained baseline_model.pt across search roots."""
    if roots is None:
        roots = SEARCH_ROOTS

    # Priority 1: Check direct data/ or root locations
    for r in roots:
        r = Path(r)
        if not r.exists():
            continue
        for candidate in [r / "data" / "baseline_model.pt", r / "baseline_model.pt"]:
            if candidate.is_file():
                return candidate

    # Priority 2: Recursive search across all roots
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


# =============================================================
# Data loading (supports multi-part & merged directory lists)
# =============================================================

def extract_case_id(filename: str, label_suffix: str = "_label") -> str:
    """Extract normalized case ID from image or label filename."""
    name = filename
    # Strip extensions
    for ext in [".nii.gz", ".nii", ".gz", ".mha", ".nrrd"]:
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
            break
    # Strip duplicate indicators like (1)
    name = name.replace("(1)", "").replace("(2)", "")
    # Strip label suffixes
    if name.lower().endswith(label_suffix.lower()):
        name = name[:-len(label_suffix)]
    elif name.lower().endswith("_label"):
        name = name[:-6]
    elif name.lower().endswith("-label"):
        name = name[:-6]
    return name.strip()


def build_file_list(images_dirs, labels_dirs, label_suffix: str = "_label"):
    """Pair up image files with label files by case ID.
    Accepts single Path, list of Paths, or directory generators.
    Recursively scans image and label directories to handle split archives (-001, -002).
    """
    if isinstance(images_dirs, (str, Path)):
        images_dirs = [Path(images_dirs)]
    if isinstance(labels_dirs, (str, Path)):
        labels_dirs = [Path(labels_dirs)]

    images_dirs = [Path(d) for d in images_dirs if d and Path(d).exists()]
    label_dirs = [Path(d) for d in labels_dirs if d and Path(d).exists()]

    # Collect image files: case_id -> Path
    all_images = {}
    for idir in images_dirs:
        for img_path in sorted(idir.rglob("*")):
            if not img_path.is_file():
                continue
            name_lower = img_path.name.lower()
            if not (name_lower.endswith(".nii.gz") or name_lower.endswith(".nii") or ".nii" in name_lower):
                continue
            # Ignore label files if they happen to be in the images folder
            if "_label" in name_lower or "-label" in name_lower:
                continue
            case_id = extract_case_id(img_path.name, label_suffix=label_suffix)
            all_images[case_id] = img_path

    # Collect label files: case_id -> Path
    all_labels = {}
    for ldir in label_dirs:
        for lbl_path in sorted(ldir.rglob("*")):
            if not lbl_path.is_file():
                continue
            name_lower = lbl_path.name.lower()
            if not (name_lower.endswith(".nii.gz") or name_lower.endswith(".nii") or ".nii" in name_lower):
                continue
            case_id = extract_case_id(lbl_path.name, label_suffix=label_suffix)
            all_labels[case_id] = lbl_path

    print(f"  [Scan] Found {len(all_images)} image(s) and {len(all_labels)} label(s).")

    file_list, missing = [], []
    for case_id, img_path in sorted(all_images.items()):
        if case_id in all_labels:
            file_list.append({"image": str(img_path), "label": str(all_labels[case_id])})
        else:
            missing.append(case_id)

    if missing:
        sample_missing = missing[:5]
        print(f"  [WARN] {len(missing)} image(s) have no matching label: {sample_missing}{'...' if len(missing) > 5 else ''}")

    if len(file_list) == 0:
        print("  [DEBUG] Sample image IDs:", list(all_images.keys())[:5])
        print("  [DEBUG] Sample label IDs:", list(all_labels.keys())[:5])

    return file_list


# =============================================================
# Transforms (with contrast dampening for training)
# =============================================================

train_transforms = Compose([
    LoadImaged(keys=["image", "label"], image_only=True),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),

    # Downsample early to reduce memory by >16x before subsequent operations
    Spacingd(
        keys=["image", "label"],
        pixdim=SPACING,
        mode=("bilinear", "nearest"),
    ),

    # Map labels on downsampled volume with uint8 to save memory
    MapLabelValued(
        keys=["label"],
        orig_labels=[
            0,
            LABEL_VALUE_MAP["pulmonary_artery_raw_value"],
            LABEL_VALUE_MAP["aorta_raw_value"],
        ],
        target_labels=[0, 1, 2],
        dtype=np.uint8,
    ),

    # Section 5A: Contrast-dampening augmentation - AFTER Spacingd, BEFORE ScaleIntensityRanged
    # Simulates non-contrast appearance on contrast training data
    RandContrastDampend(
        keys=["image"],
        label_key="label",
        prob=0.4,
        target_hu=50.0,
        blend_range=(0.5, 1.0),
    ),

    ScaleIntensityRanged(
        keys=["image"],
        a_min=HU_MIN,
        a_max=HU_MAX,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    ),

    CropForegroundd(
        keys=["image", "label"],
        source_key="image",
    ),

    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=PATCH_SIZE,
        pos=2,
        neg=1,
        num_samples=2,
        image_key="image",
        image_threshold=0,
    ),

    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),

    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    RandGaussianNoised(keys=["image"], prob=0.3, std=0.03),
    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.3)),

    RandAffined(
        keys=["image", "label"],
        prob=0.3,
        rotate_range=(0.1, 0.1, 0.1),
        scale_range=(0.1, 0.1, 0.1),
        mode=("bilinear", "nearest"),
    ),

    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"], image_only=True),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),

    Spacingd(
        keys=["image", "label"],
        pixdim=SPACING,
        mode=("bilinear", "nearest"),
    ),

    MapLabelValued(
        keys=["label"],
        orig_labels=[
            0,
            LABEL_VALUE_MAP["pulmonary_artery_raw_value"],
            LABEL_VALUE_MAP["aorta_raw_value"],
        ],
        target_labels=[0, 1, 2],
        dtype=np.uint8,
    ),

    ScaleIntensityRanged(
        keys=["image"],
        a_min=HU_MIN,
        a_max=HU_MAX,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    ),

    CropForegroundd(
        keys=["image", "label"],
        source_key="image",
    ),

    EnsureTyped(keys=["image", "label"]),
])


# =============================================================
# Model + Loss
# =============================================================

def build_model():
    return SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=NUM_CLASSES,
        init_filters=16,
        dropout_prob=0.2,
    ).to(DEVICE)


def build_loss():
    """Section 5D: Combined DiceCELoss + HausdorffDTLoss for better tubular vessel segmentation."""
    dice_ce = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
    )
    if HAS_HAUSDORFF:
        hausdorff = HausdorffDTLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
        )
        return dice_ce, hausdorff
    return dice_ce, None


dice_metric = DiceMetric(
    include_background=False,
    reduction="mean_batch",
)

post_pred = Compose([AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)])
post_label = Compose([AsDiscrete(to_onehot=NUM_CLASSES)])


# =============================================================
# Training loop
# =============================================================

def run_training(model, train_loader, val_loader, epochs, checkpoint_path,
                 lr=1e-4, tag="", early_stop_patience=None):
    """Train with combined loss, per-class NC validation, optional early stopping."""
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
                        pass  # Graceful fallback if HausdorffDT fails on a batch

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

                    outputs = sliding_window_inference(
                        images,
                        PATCH_SIZE,
                        sw_batch_size=2,
                        predictor=model,
                        overlap=0.5,
                    )

                    outputs = [post_pred(x) for x in decollate_batch(outputs)]
                    labels_batch = [post_label(x) for x in decollate_batch(labels_batch)]
                    dice_metric(y_pred=outputs, y=labels_batch)

            per_class = dice_metric.aggregate()
            mean_dice = per_class.mean().item()
            pa_dice = per_class[0].item()
            aorta_dice = per_class[1].item()

            val_dice_history.append((epoch + 1, mean_dice, pa_dice, aorta_dice))

            print(
                f"  [{tag}] Epoch {epoch+1:03d}/{epochs} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Dice: {mean_dice:.4f} (PA: {pa_dice:.4f}, Aorta: {aorta_dice:.4f})"
            )

            if mean_dice > best_metric:
                best_metric = mean_dice
                torch.save(model.state_dict(), checkpoint_path)
                no_improve_count = 0
                print(f"  >> New best! Saved to {checkpoint_path}")
            else:
                no_improve_count += VAL_INTERVAL

            # Early stopping
            if early_stop_patience and no_improve_count >= early_stop_patience:
                print(f"  [WARN] Early stopping: no improvement for {no_improve_count} epochs.")
                break
        else:
            print(
                f"  [{tag}] Epoch {epoch+1:03d}/{epochs} | Loss: {epoch_loss:.4f}",
                end="\r",
            )

    print()
    return train_loss_history, val_dice_history, best_metric


# =============================================================
# Qualitative prediction viewer
# =============================================================

def save_predictions(model, val_files, output_dir):
    """Generate and save side-by-side prediction PNGs for NC validation cases."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    for i, sample_file in enumerate(val_files):
        print(f"  Generating prediction for NC case {i+1}/{len(val_files)}...")
        sample = val_transforms(sample_file)
        inp = sample["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = sliding_window_inference(
                inp, PATCH_SIZE, sw_batch_size=2,
                predictor=model, overlap=0.5,
            )

        img = sample["image"][0].cpu().numpy()
        gt = sample["label"][0].cpu().numpy()
        pred_seg = torch.argmax(pred, dim=1)[0].cpu().numpy()

        # Pick the slice with the most ground-truth labels
        label_counts = [(gt[:, :, s] > 0).sum() for s in range(gt.shape[2])]
        best_slice = int(np.argmax(label_counts))

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # CT image
        axes[0].imshow(img[:, :, best_slice], cmap="gray")
        axes[0].set_title("Non-Contrast CT")
        axes[0].axis("off")

        # Ground truth overlay
        axes[1].imshow(img[:, :, best_slice], cmap="gray")
        axes[1].imshow(gt[:, :, best_slice], cmap="viridis", alpha=0.45, vmin=0, vmax=2)
        axes[1].set_title("CT + Ground Truth")
        axes[1].axis("off")

        # Prediction overlay
        axes[2].imshow(img[:, :, best_slice], cmap="gray")
        axes[2].imshow(pred_seg[:, :, best_slice], cmap="viridis", alpha=0.45, vmin=0, vmax=2)
        axes[2].set_title("CT + Prediction")
        axes[2].axis("off")

        case_name = Path(sample_file["image"]).stem.replace(".nii", "")
        fig.suptitle(f"NC Case: {case_name} - Slice {best_slice}", fontsize=14)
        plt.tight_layout()

        out_path = output_dir / f"nc_pred_{case_name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

        if IS_COLAB or IS_KAGGLE:
            plt.show()  # Display inline in Colab/Kaggle notebook
        plt.close(fig)
        print(f"  >> Saved: {out_path}")


# =============================================================
# Main
# =============================================================

def main():
    print("=" * 60)
    print("  NON-CONTRAST CT SEGMENTATION TRAINING")
    print("  Strategy: Section 5A Contrast dampening + Section 5B Joint/fine-tune")
    print("            Section 5C NC validation + Section 5D Hausdorff loss")
    if IS_COLAB:
        env_str = "Google Colab (GPU)" if torch.cuda.is_available() else "Google Colab (CPU)"
    elif IS_KAGGLE:
        env_str = "Kaggle (GPU)" if torch.cuda.is_available() else "Kaggle (CPU)"
    else:
        env_str = "Local PC"
    print(f"  Environment: {env_str}")
    print(f"  Output Dir:  {WORK_DIR}")
    print("=" * 60)
    print()

    # ---------------------------------------------------------
    # Locate data folders
    # ---------------------------------------------------------
    print("Locating data folders...")

    contrast_images_dirs = find_matching_dirs("unused_contrast_images")
    contrast_labels_dirs = find_matching_dirs("unused_contrast_labels")
    nc_images_dirs = find_matching_dirs("non_contrast_images")
    nc_labels_dirs = find_matching_dirs("non_contrast_labels")

    if contrast_images_dirs:
        print(f"  Contrast image dirs: {[str(d) for d in contrast_images_dirs]}")
    else:
        print("  [WARN] No unused_contrast_images folders found.")

    if contrast_labels_dirs:
        print(f"  Contrast label dirs: {[str(d) for d in contrast_labels_dirs]}")
    else:
        print("  [WARN] No unused_contrast_labels folders found.")

    print(f"  NC image dirs:       {[str(d) for d in nc_images_dirs]}")
    print(f"  NC label dirs:       {[str(d) for d in nc_labels_dirs]}")

    baseline_ckpt = find_baseline_checkpoint()
    if baseline_ckpt:
        print(f"  Baseline model:      {baseline_ckpt}")
    else:
        print("  [WARN] No baseline_model.pt found - will train from scratch.")
    print()

    # ---------------------------------------------------------
    # Build file lists
    # ---------------------------------------------------------
    print("Building file lists...")

    contrast_files = []
    if contrast_images_dirs and contrast_labels_dirs:
        contrast_files = build_file_list(contrast_images_dirs, contrast_labels_dirs)
        print(f"  Contrast pairs:      {len(contrast_files)}")

    nc_files = build_file_list(nc_images_dirs, nc_labels_dirs)
    print(f"  NC pairs:            {len(nc_files)}")

    if len(nc_files) == 0:
        print("ERROR: No non-contrast image/label pairs found. Exiting.")
        sys.exit(1)

    # ---------------------------------------------------------
    # Split NC data (train/val)
    # ---------------------------------------------------------
    print("\nSplitting NC data...")

    rng = np.random.default_rng(RANDOM_SEED)
    nc_indices = np.arange(len(nc_files))
    rng.shuffle(nc_indices)

    n_nc_val = max(1, int(len(nc_files) * 0.22))  # ~2 out of 9
    nc_val_idx = nc_indices[:n_nc_val]
    nc_train_idx = nc_indices[n_nc_val:]

    nc_train_files = [nc_files[i] for i in nc_train_idx]
    nc_val_files = [nc_files[i] for i in nc_val_idx]

    print(f"  NC train: {len(nc_train_files)}")
    print(f"  NC val:   {len(nc_val_files)}")

    # Phase 1 training set: contrast + NC train
    phase1_train_files = contrast_files + nc_train_files
    print(f"  Phase 1 total train: {len(phase1_train_files)} "
          f"({len(contrast_files)} contrast + {len(nc_train_files)} NC)")

    # ---------------------------------------------------------
    # Build model & load baseline
    # ---------------------------------------------------------
    print("\nBuilding model...")
    model = build_model()

    if baseline_ckpt:
        print(f"  Loading pre-trained baseline from {baseline_ckpt}...")
        state_dict = torch.load(baseline_ckpt, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print("  [OK] Baseline loaded (Dice=0.8756 on contrast CT).")
    print()

    # ---------------------------------------------------------
    # Phase 1: Joint training (contrast + NC)
    # ---------------------------------------------------------
    print("=" * 60)
    print("  PHASE 1: Joint Training (Contrast + Non-Contrast)")
    print(f"  Epochs: {PHASE1_EPOCHS}, LR: {PHASE1_LR}")
    print(f"  Train cases: {len(phase1_train_files)}, Val cases: {len(nc_val_files)}")
    print("=" * 60)

    phase1_train_ds = Dataset(data=phase1_train_files, transform=train_transforms)
    nc_val_ds = Dataset(data=nc_val_files, transform=val_transforms)

    phase1_train_loader = DataLoader(
        phase1_train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    nc_val_loader = DataLoader(
        nc_val_ds, batch_size=1, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    phase1_ckpt = WORK_DIR / "phase1_best.pt"
    _, phase1_history, phase1_best = run_training(
        model, phase1_train_loader, nc_val_loader,
        epochs=PHASE1_EPOCHS,
        checkpoint_path=phase1_ckpt,
        lr=PHASE1_LR,
        tag="Phase1-Joint",
    )

    print(f"\n  Phase 1 best NC Dice: {phase1_best:.4f}")

    # ---------------------------------------------------------
    # Phase 2: NC-only fine-tuning
    # ---------------------------------------------------------
    print()
    print("=" * 60)
    print("  PHASE 2: Non-Contrast Fine-Tuning")
    print(f"  Epochs: {PHASE2_EPOCHS}, LR: {PHASE2_LR}")
    print(f"  Train cases: {len(nc_train_files)}, Val cases: {len(nc_val_files)}")
    print(f"  Early stopping patience: {EARLY_STOP_PATIENCE} epochs")
    print("=" * 60)

    # Load Phase 1 best checkpoint
    model.load_state_dict(torch.load(phase1_ckpt, map_location=DEVICE, weights_only=True))
    print("  [OK] Loaded Phase 1 best checkpoint.")

    phase2_train_ds = Dataset(data=nc_train_files, transform=train_transforms)
    phase2_train_loader = DataLoader(
        phase2_train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    phase2_ckpt = WORK_DIR / "phase2_nc_finetuned.pt"
    _, phase2_history, phase2_best = run_training(
        model, phase2_train_loader, nc_val_loader,
        epochs=PHASE2_EPOCHS,
        checkpoint_path=phase2_ckpt,
        lr=PHASE2_LR,
        tag="Phase2-NC",
        early_stop_patience=EARLY_STOP_PATIENCE,
    )

    print(f"\n  Phase 2 best NC Dice: {phase2_best:.4f}")

    # ---------------------------------------------------------
    # Final results
    # ---------------------------------------------------------
    print()
    print("=" * 60)
    print("        TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Phase 1 (joint) best NC Dice:     {phase1_best:.4f}")
    print(f"  Phase 2 (NC fine-tune) best Dice:  {phase2_best:.4f}")
    print(f"  Target Dice:                       >= 0.8000")
    print(f"  Phase 1 checkpoint:  {phase1_ckpt}")
    print(f"  Phase 2 checkpoint:  {phase2_ckpt}")
    print("=" * 60)

    # Determine which checkpoint is best overall
    if phase2_best >= phase1_best:
        best_ckpt = phase2_ckpt
        best_dice = phase2_best
        best_phase = "Phase 2 (NC fine-tuned)"
    else:
        best_ckpt = phase1_ckpt
        best_dice = phase1_best
        best_phase = "Phase 1 (joint)"

    print(f"\n  Best overall: {best_phase} - Dice: {best_dice:.4f}")

    # Save results
    results = {
        "phase1_best_dice": float(phase1_best),
        "phase2_best_dice": float(phase2_best),
        "best_phase": best_phase,
        "best_dice": float(best_dice),
        "phase1_epochs": PHASE1_EPOCHS,
        "phase2_epochs": PHASE2_EPOCHS,
        "n_contrast_train": len(contrast_files),
        "n_nc_train": len(nc_train_files),
        "n_nc_val": len(nc_val_files),
        "spacing": list(SPACING),
        "patch_size": list(PATCH_SIZE),
        "phase1_val_history": phase1_history,
        "phase2_val_history": phase2_history,
    }
    results_path = WORK_DIR / "nc_training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {results_path}")

    # ---------------------------------------------------------
    # Qualitative predictions
    # ---------------------------------------------------------
    print("\nGenerating qualitative predictions on NC validation cases...")
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE, weights_only=True))
    save_predictions(model, nc_val_files, WORK_DIR / "nc_predictions")

    print("\n[DONE] All done!")
    if IS_COLAB:
        print(f"  Checkpoints saved to Google Drive: {WORK_DIR}")
        print(f"  Predictions saved to: {WORK_DIR / 'nc_predictions'}")
    elif IS_KAGGLE:
        print(f"  Checkpoints saved to: {WORK_DIR}")
        print(f"  Predictions saved to: {WORK_DIR / 'nc_predictions'}")
        print(f"  (Artifacts will be available under Kaggle Notebook /kaggle/working/data)")


if __name__ == "__main__":
    main()
