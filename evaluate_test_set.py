"""
Non-Contrast CT Test Set Evaluation Script
============================================
Loads the Phase 2 fine-tuned model and evaluates on the held-out test set.
Produces:
  - Per-case & per-class Dice scores (JSON + CSV)
  - Side-by-side prediction PNGs for every test case
  - Summary statistics printed to console

Usage:
    Kaggle:  !python evaluate_test_set.py
    Local:   python evaluate_test_set.py
"""

# =============================================================
# Step 0: Environment detection & dependencies
# =============================================================

import os
import subprocess
import sys
from pathlib import Path

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
    print(f"Running on {env_name} -- checking dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "monai[nibabel,tqdm,itk]", "nibabel", "simpleitk",
        "scikit-image", "scikit-learn", "matplotlib", "tqdm", "pandas",
    ])
    print("[OK] Dependencies ready.\n")

# =============================================================
# Step 1: Imports
# =============================================================

import json
import csv
import numpy as np
import matplotlib
if IS_LOCAL:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import monai
from monai.data import Dataset, DataLoader, list_data_collate, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, MapLabelValued, EnsureTyped,
    AsDiscrete,
)
from monai.networks.nets import SegResNet
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference

# =============================================================
# Step 2: Configuration
# =============================================================

SPACING = (2.5, 2.5, 2.5)
HU_MIN, HU_MAX = -200.0, 400.0
PATCH_SIZE = (64, 64, 64)

CLASS_NAMES = ["background", "pulmonary_artery", "aorta"]
NUM_CLASSES = len(CLASS_NAMES)
LABEL_VALUE_MAP = {"pulmonary_artery_raw_value": 1, "aorta_raw_value": 2}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"MONAI version: {monai.__version__}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Device: {DEVICE}")
print()

# =============================================================
# Step 3: Locate files (Kaggle / Colab / Local)
# =============================================================

def find_dir(candidates):
    """Return the first directory path that exists from a list of candidates."""
    for p in candidates:
        p = Path(p)
        if p.exists() and p.is_dir():
            return p
    return None


def find_file(candidates):
    """Return the first file path that exists from a list of candidates."""
    for p in candidates:
        p = Path(p)
        if p.exists() and p.is_file():
            return p
    return None


def find_file_recursive(roots, filename):
    """Search recursively across roots for a file by name."""
    for r in roots:
        r = Path(r)
        if not r.exists():
            continue
        try:
            for f in r.rglob(filename):
                if f.is_file():
                    return f
        except Exception:
            pass
    return None


# --- Test images directory ---
test_images_dir = find_dir([
    # Kaggle: user may upload as a separate dataset or inside ctscan
    "/kaggle/input/non-contrast_test_set_images",
    "/kaggle/input/non-contrast-test-set-images",
    "/kaggle/input/datasets/rafaprapti/ctscan/non-contrast_test_set_images",
    "/kaggle/input/datasets/rafaprapti/ctscan/non-contrast-test-set-images",
    # Colab
    "/content/drive/MyDrive/ai_healthcarehackathon/non-contrast_test_set_images",
    # Local
    Path.cwd() / "non-contrast_test_set_images",
])

# If not found by clean name, search recursively in /kaggle/input
if test_images_dir is None and IS_KAGGLE:
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for candidate in kaggle_input.rglob("*test*image*"):
            if candidate.is_dir() and any(f.name.endswith((".nii.gz", ".nii")) for f in candidate.iterdir() if f.is_file()):
                test_images_dir = candidate
                break

# --- Test labels directory ---
test_labels_dir = find_dir([
    "/kaggle/input/non-contrast_test_set_labels",
    "/kaggle/input/non-contrast-test-set-labels",
    "/kaggle/input/datasets/rafaprapti/ctscan/non-contrast_test_set_labels",
    "/kaggle/input/datasets/rafaprapti/ctscan/non-contrast-test-set-labels",
    "/content/drive/MyDrive/ai_healthcarehackathon/non-contrast_test_set_labels",
    Path.cwd() / "non-contrast_test_set_labels",
])

if test_labels_dir is None and IS_KAGGLE:
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for candidate in kaggle_input.rglob("*test*label*"):
            if candidate.is_dir() and any(f.name.endswith((".nii.gz", ".nii")) for f in candidate.iterdir() if f.is_file()):
                test_labels_dir = candidate
                break

# --- Checkpoint file ---
search_roots = []
if IS_KAGGLE:
    search_roots = [Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()]
elif IS_COLAB:
    search_roots = [Path("/content/drive/MyDrive/ai_healthcarehackathon"), Path("/content"), Path.cwd()]
else:
    search_roots = [Path.cwd(), Path.cwd() / "data", Path.cwd() / "results (1)" / "data"]

checkpoint_path = find_file([
    # Kaggle: user's actual notebook output paths
    Path("/kaggle/input/notebooks/rafaprapti/ai-hackathon/data/phase2_nc_finetuned.pt"),
    Path("/kaggle/input/notebooks/rafaprapti/ai-hackathon/data/phase1_best.pt"),
    # Kaggle: other common locations
    Path("/kaggle/input/datasets/rafaprapti/ctscan/data/phase2_nc_finetuned.pt"),
    Path("/kaggle/working/data/phase2_nc_finetuned.pt"),
    # Colab
    Path("/content/drive/MyDrive/ai_healthcarehackathon/data/phase2_nc_finetuned.pt"),
    # Local
    Path.cwd() / "data" / "phase2_nc_finetuned.pt",
    Path.cwd() / "results (1)" / "data" / "phase2_nc_finetuned.pt",
])

if checkpoint_path is None:
    checkpoint_path = find_file_recursive(search_roots, "phase2_nc_finetuned.pt")

if checkpoint_path is None:
    # Fallback to phase1_best.pt
    checkpoint_path = find_file_recursive(search_roots, "phase1_best.pt")

# --- Output directory ---
if IS_KAGGLE:
    OUTPUT_DIR = Path("/kaggle/working/test_results")
elif IS_COLAB:
    OUTPUT_DIR = Path("/content/drive/MyDrive/ai_healthcarehackathon/test_results")
else:
    OUTPUT_DIR = Path.cwd() / "test_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR = OUTPUT_DIR / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

# --- Print summary ---
print("=" * 60)
print("  TEST SET EVALUATION")
print("=" * 60)
print(f"  Test images: {test_images_dir}")
print(f"  Test labels: {test_labels_dir}")
print(f"  Checkpoint:  {checkpoint_path}")
print(f"  Output dir:  {OUTPUT_DIR}")
print("=" * 60)
print()

if test_images_dir is None:
    print("ERROR: Could not find test images directory. Exiting.")
    sys.exit(1)
if test_labels_dir is None:
    print("ERROR: Could not find test labels directory. Exiting.")
    sys.exit(1)
if checkpoint_path is None:
    print("ERROR: Could not find model checkpoint. Exiting.")
    sys.exit(1)


# =============================================================
# Step 4: Build test file list
# =============================================================

def extract_case_id(filename, label_suffix="_label"):
    name = filename
    for ext in [".nii.gz", ".nii", ".gz"]:
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
            break
    if name.lower().endswith(label_suffix.lower()):
        name = name[:-len(label_suffix)]
    return name.strip()


# Collect images
all_images = {}
for img_path in sorted(test_images_dir.rglob("*")):
    if img_path.is_file() and (".nii" in img_path.name.lower()):
        if "_label" not in img_path.name.lower():
            case_id = extract_case_id(img_path.name)
            all_images[case_id] = img_path

# Collect labels
all_labels = {}
for lbl_path in sorted(test_labels_dir.rglob("*")):
    if lbl_path.is_file() and (".nii" in lbl_path.name.lower()):
        case_id = extract_case_id(lbl_path.name)
        all_labels[case_id] = lbl_path

# Pair them
test_files = []
for case_id in sorted(all_images.keys()):
    if case_id in all_labels:
        test_files.append({
            "image": str(all_images[case_id]),
            "label": str(all_labels[case_id]),
            "case_id": case_id,
        })

print(f"Found {len(all_images)} test images, {len(all_labels)} test labels.")
print(f"Matched {len(test_files)} test pairs:")
for f in test_files:
    print(f"  {f['case_id']}: {Path(f['image']).name} <-> {Path(f['label']).name}")
print()

if len(test_files) == 0:
    print("ERROR: No test pairs found. Exiting.")
    sys.exit(1)


# =============================================================
# Step 5: Transforms (validation / inference mode)
# =============================================================

test_transforms = Compose([
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
# Step 6: Build model and load checkpoint
# =============================================================

print("Building model and loading checkpoint...")
model = SegResNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=NUM_CLASSES,
    init_filters=16,
    dropout_prob=0.2,
).to(DEVICE)

state_dict = torch.load(str(checkpoint_path), map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.eval()
print(f"[OK] Loaded checkpoint: {checkpoint_path.name}")
print()

post_pred = Compose([AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)])
post_label = Compose([AsDiscrete(to_onehot=NUM_CLASSES)])


# =============================================================
# Step 7: Run evaluation
# =============================================================

print("=" * 60)
print("  RUNNING TEST SET EVALUATION")
print(f"  Cases: {len(test_files)}")
print("=" * 60)
print()

dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

# Try to use Hausdorff distance metric too
try:
    hd_metric = HausdorffDistanceMetric(include_background=False, reduction="mean_batch", percentile=95)
    HAS_HD = True
except Exception:
    HAS_HD = False

per_case_results = []

for i, test_case in enumerate(test_files):
    case_id = test_case["case_id"]
    print(f"[{i+1}/{len(test_files)}] Processing {case_id}...")

    # Load and transform
    sample = test_transforms(test_case)
    inp = sample["image"].unsqueeze(0).to(DEVICE)
    lbl = sample["label"].unsqueeze(0).to(DEVICE)

    # Sliding window inference
    with torch.no_grad():
        pred = sliding_window_inference(
            inp, PATCH_SIZE, sw_batch_size=2,
            predictor=model, overlap=0.5,
        )

    # Post-process (ensure both on same device)
    pred_onehot = [post_pred(x) for x in decollate_batch(pred)]
    label_onehot = [post_label(x) for x in decollate_batch(lbl)]

    # Compute per-case Dice
    dice_metric.reset()
    dice_metric(y_pred=pred_onehot, y=label_onehot)
    per_class_dice = dice_metric.aggregate()
    pa_dice = per_class_dice[0].item()
    aorta_dice = per_class_dice[1].item()
    mean_dice = (pa_dice + aorta_dice) / 2.0

    # Compute Hausdorff distance if available
    pa_hd, aorta_hd = -1.0, -1.0
    if HAS_HD:
        try:
            hd_metric.reset()
            hd_metric(y_pred=pred_onehot, y=label_onehot)
            per_class_hd = hd_metric.aggregate()
            pa_hd = per_class_hd[0].item()
            aorta_hd = per_class_hd[1].item()
        except Exception:
            pass

    case_result = {
        "case_id": case_id,
        "mean_dice": round(mean_dice, 4),
        "pulmonary_artery_dice": round(pa_dice, 4),
        "aorta_dice": round(aorta_dice, 4),
    }
    if HAS_HD and pa_hd >= 0:
        case_result["pulmonary_artery_hd95"] = round(pa_hd, 2)
        case_result["aorta_hd95"] = round(aorta_hd, 2)

    per_case_results.append(case_result)

    print(f"  Mean Dice: {mean_dice:.4f} | PA: {pa_dice:.4f} | Aorta: {aorta_dice:.4f}", end="")
    if HAS_HD and pa_hd >= 0:
        print(f" | PA HD95: {pa_hd:.2f}mm | Aorta HD95: {aorta_hd:.2f}mm", end="")
    print()

    # -------------------------------------------------------
    # Generate side-by-side visualization (3 slices per case)
    # -------------------------------------------------------
    img_np = sample["image"][0].cpu().numpy()
    gt_np = sample["label"][0].cpu().numpy()
    pred_seg = torch.argmax(pred, dim=1)[0].cpu().numpy()

    # Pick the slice with the most ground-truth labels
    label_counts = [(gt_np[:, :, s] > 0).sum() for s in range(gt_np.shape[2])]
    best_slice = int(np.argmax(label_counts))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # CT image
    axes[0].imshow(img_np[:, :, best_slice], cmap="gray")
    axes[0].set_title("Non-Contrast CT")
    axes[0].axis("off")

    # Ground truth overlay
    axes[1].imshow(img_np[:, :, best_slice], cmap="gray")
    axes[1].imshow(gt_np[:, :, best_slice], cmap="viridis", alpha=0.45, vmin=0, vmax=2)
    axes[1].set_title("CT + Ground Truth")
    axes[1].axis("off")

    # Prediction overlay
    axes[2].imshow(img_np[:, :, best_slice], cmap="gray")
    axes[2].imshow(pred_seg[:, :, best_slice], cmap="viridis", alpha=0.45, vmin=0, vmax=2)
    axes[2].set_title("CT + Prediction")
    axes[2].axis("off")

    fig.suptitle(
        f"Test Case: {case_id} — Slice {best_slice}\n"
        f"Dice: {mean_dice:.4f} (PA: {pa_dice:.4f}, Aorta: {aorta_dice:.4f})",
        fontsize=14,
    )
    plt.tight_layout()

    out_path = PRED_DIR / f"test_pred_{case_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    if IS_COLAB or IS_KAGGLE:
        plt.show()
    plt.close(fig)
    print(f"  >> Saved: {out_path}")
    print()


# =============================================================
# Step 8: Compute summary statistics & save results
# =============================================================

all_mean = np.mean([r["mean_dice"] for r in per_case_results])
all_pa = np.mean([r["pulmonary_artery_dice"] for r in per_case_results])
all_aorta = np.mean([r["aorta_dice"] for r in per_case_results])

summary = {
    "checkpoint": str(checkpoint_path),
    "num_test_cases": len(test_files),
    "overall_mean_dice": round(float(all_mean), 4),
    "overall_pulmonary_artery_dice": round(float(all_pa), 4),
    "overall_aorta_dice": round(float(all_aorta), 4),
    "target_dice": 0.8000,
    "target_met": bool(all_mean >= 0.80),
    "per_case_results": per_case_results,
}

# Save JSON
json_path = OUTPUT_DIR / "test_results.json"
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)

# Save CSV
csv_path = OUTPUT_DIR / "test_results.csv"
with open(csv_path, "w", newline="") as f:
    fieldnames = list(per_case_results[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(per_case_results)

# Print final summary
print()
print("=" * 60)
print("       TEST SET EVALUATION COMPLETE")
print("=" * 60)
print(f"  Test Cases:                   {len(test_files)}")
print(f"  Overall Mean Dice:            {all_mean:.4f}")
print(f"  Pulmonary Artery Dice:        {all_pa:.4f}")
print(f"  Aorta Dice:                   {all_aorta:.4f}")
print(f"  Target (>= 0.80):             {'PASSED' if all_mean >= 0.80 else 'NOT MET'}")
print()
print("  Per-Case Breakdown:")
print(f"  {'Case ID':<25} {'Mean':>8} {'PA':>8} {'Aorta':>8}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
for r in per_case_results:
    print(f"  {r['case_id']:<25} {r['mean_dice']:>8.4f} {r['pulmonary_artery_dice']:>8.4f} {r['aorta_dice']:>8.4f}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
print(f"  {'AVERAGE':<25} {all_mean:>8.4f} {all_pa:>8.4f} {all_aorta:>8.4f}")
print()
print(f"  Results JSON:  {json_path}")
print(f"  Results CSV:   {csv_path}")
print(f"  Predictions:   {PRED_DIR}")
print("=" * 60)

if IS_KAGGLE:
    print(f"\n  All outputs saved under /kaggle/working/test_results/")
    print(f"  Download from Kaggle Notebook -> Output tab.")
