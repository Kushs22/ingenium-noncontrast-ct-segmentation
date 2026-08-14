"""
Calculate Confusion Matrix, Accuracy, and False Positive Analysis
==================================================================
Runs validation on the test set using the fine-tuned checkpoint:
  - 3x3 Multi-class Confusion Matrix (Background, Pulmonary Artery, Aorta)
  - Voxel-level True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN)
  - Accuracy, Precision (PPV), Recall (Sensitivity), Specificity (TNR)
  - False Positive Rate (FPR) & False Discovery Rate (FDR)
  - Voxel count & volume (mL) of False Positives
  - Confusion Matrix plots & per-case breakdown CSV/JSON
"""

import os
import sys
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, MapLabelValued, EnsureTyped,
)
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference

# =============================================================
# Configuration
# =============================================================
SPACING = (2.5, 2.5, 2.5)
VOXEL_VOLUME_MM3 = SPACING[0] * SPACING[1] * SPACING[2]  # 15.625 mm³
VOXEL_VOLUME_ML = VOXEL_VOLUME_MM3 / 1000.0              # 0.015625 mL

HU_MIN, HU_MAX = -200.0, 400.0
PATCH_SIZE = (64, 64, 64)
CLASS_NAMES = ["Background", "Pulmonary Artery", "Aorta"]
NUM_CLASSES = len(CLASS_NAMES)
LABEL_VALUE_MAP = {"pulmonary_artery_raw_value": 1, "aorta_raw_value": 2}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_IMG_DIR = PROJECT_ROOT / "non-contrast_test_set_images"
TEST_LBL_DIR = PROJECT_ROOT / "non-contrast_test_set_labels"
OUTPUT_DIR = PROJECT_ROOT / "test_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Locate checkpoint
checkpoint_candidates = [
    PROJECT_ROOT / "results (1)" / "data" / "phase2_nc_finetuned.pt",
    PROJECT_ROOT / "data" / "phase2_nc_finetuned.pt",
    PROJECT_ROOT / "results (1)" / "data" / "phase1_best.pt",
    PROJECT_ROOT / "data" / "phase1_best.pt",
]

checkpoint_path = None
for cp in checkpoint_candidates:
    if cp.exists() and cp.is_file():
        checkpoint_path = cp
        break

if checkpoint_path is None:
    for f in PROJECT_ROOT.rglob("*.pt"):
        if "phase" in f.name.lower():
            checkpoint_path = f
            break

print("=" * 70)
print("  COMPREHENSIVE TEST SET ACCURACY & CONFUSION MATRIX ANALYSIS")
print("=" * 70)
print(f"  Device:       {DEVICE}")
print(f"  Images dir:   {TEST_IMG_DIR}")
print(f"  Labels dir:   {TEST_LBL_DIR}")
print(f"  Checkpoint:   {checkpoint_path}")
print(f"  Output dir:   {OUTPUT_DIR}")
print("=" * 70)
print()

if not TEST_IMG_DIR.exists() or not TEST_LBL_DIR.exists() or checkpoint_path is None:
    print("ERROR: Missing required input directories or checkpoint.")
    sys.exit(1)

# =============================================================
# Build File List
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

all_images = {extract_case_id(p.name): p for p in sorted(TEST_IMG_DIR.glob("*.nii*")) if "_label" not in p.name.lower()}
all_labels = {extract_case_id(p.name): p for p in sorted(TEST_LBL_DIR.glob("*.nii*"))}

test_cases = []
for cid in sorted(all_images.keys()):
    if cid in all_labels:
        test_cases.append({
            "case_id": cid,
            "image": str(all_images[cid]),
            "label": str(all_labels[cid]),
        })

print(f"Found {len(test_cases)} test cases:")
for tc in test_cases:
    print(f"  - {tc['case_id']}")
print()

# =============================================================
# Transforms & Model Loading
# =============================================================
test_transforms = Compose([
    LoadImaged(keys=["image", "label"], image_only=True),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=SPACING, mode=("bilinear", "nearest")),
    MapLabelValued(
        keys=["label"],
        orig_labels=[0, LABEL_VALUE_MAP["pulmonary_artery_raw_value"], LABEL_VALUE_MAP["aorta_raw_value"]],
        target_labels=[0, 1, 2],
        dtype=np.uint8,
    ),
    ScaleIntensityRanged(keys=["image"], a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    EnsureTyped(keys=["image", "label"]),
])

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
print(f"[OK] Model loaded from: {checkpoint_path.name}\n")

# =============================================================
# Confusion Matrix & Metric Accumulators
# =============================================================
# 3x3 Confusion Matrix: Rows = True Class, Cols = Predicted Class
cm_global = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

per_case_metrics = []

print("Running voxel-level evaluation across all test cases...")
for idx, tc in enumerate(test_cases, 1):
    case_id = tc["case_id"]
    print(f"[{idx}/{len(test_cases)}] Evaluating {case_id}...")

    sample = test_transforms(tc)
    inp = sample["image"].unsqueeze(0).to(DEVICE)
    lbl = sample["label"].unsqueeze(0)

    with torch.no_grad():
        pred_logits = sliding_window_inference(
            inp, PATCH_SIZE, sw_batch_size=2,
            predictor=model, overlap=0.5,
        )
        pred_classes = torch.argmax(pred_logits, dim=1).cpu().numpy().squeeze(0).astype(np.int64)

    gt_classes = lbl.squeeze(0).squeeze(0).cpu().numpy().astype(np.int64)

    # Flatten for confusion matrix
    gt_flat = gt_classes.flatten()
    pred_flat = pred_classes.flatten()

    # Compute Case Confusion Matrix
    cm_case = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for true_c in range(NUM_CLASSES):
        for pred_c in range(NUM_CLASSES):
            cm_case[true_c, pred_c] = np.sum((gt_flat == true_c) & (pred_flat == pred_c))

    cm_global += cm_case

    # Compute per-case class stats
    case_total_voxels = len(gt_flat)
    case_acc = np.sum(gt_flat == pred_flat) / case_total_voxels

    # PA stats (Class 1)
    tp_pa = cm_case[1, 1]
    fp_pa = cm_case[0, 1] + cm_case[2, 1]
    fn_pa = cm_case[1, 0] + cm_case[1, 2]
    tn_pa = case_total_voxels - (tp_pa + fp_pa + fn_pa)
    pa_prec = tp_pa / (tp_pa + fp_pa) if (tp_pa + fp_pa) > 0 else 0.0
    pa_rec = tp_pa / (tp_pa + fn_pa) if (tp_pa + fn_pa) > 0 else 0.0
    pa_spec = tn_pa / (tn_pa + fp_pa) if (tn_pa + fp_pa) > 0 else 0.0
    pa_dice = (2.0 * tp_pa) / (2.0 * tp_pa + fp_pa + fn_pa) if (2.0 * tp_pa + fp_pa + fn_pa) > 0 else 0.0

    # Aorta stats (Class 2)
    tp_ao = cm_case[2, 2]
    fp_ao = cm_case[0, 2] + cm_case[1, 2]
    fn_ao = cm_case[2, 0] + cm_case[2, 1]
    tn_ao = case_total_voxels - (tp_ao + fp_ao + fn_ao)
    ao_prec = tp_ao / (tp_ao + fp_ao) if (tp_ao + fp_ao) > 0 else 0.0
    ao_rec = tp_ao / (tp_ao + fn_ao) if (tp_ao + fn_ao) > 0 else 0.0
    ao_spec = tn_ao / (tn_ao + fp_ao) if (tn_ao + fp_ao) > 0 else 0.0
    ao_dice = (2.0 * tp_ao) / (2.0 * tp_ao + fp_ao + fn_ao) if (2.0 * tp_ao + fp_ao + fn_ao) > 0 else 0.0

    per_case_metrics.append({
        "case_id": case_id,
        "total_voxels": int(case_total_voxels),
        "overall_accuracy": round(float(case_acc), 5),
        "mean_dice": round(float((pa_dice + ao_dice) / 2.0), 4),
        # Pulmonary Artery
        "pa_tp_voxels": int(tp_pa),
        "pa_fp_voxels": int(fp_pa),
        "pa_fp_volume_ml": round(float(fp_pa * VOXEL_VOLUME_ML), 2),
        "pa_fn_voxels": int(fn_pa),
        "pa_precision": round(float(pa_prec), 4),
        "pa_recall": round(float(pa_rec), 4),
        "pa_specificity": round(float(pa_spec), 5),
        "pa_dice": round(float(pa_dice), 4),
        # Aorta
        "aorta_tp_voxels": int(tp_ao),
        "aorta_fp_voxels": int(fp_ao),
        "aorta_fp_volume_ml": round(float(fp_ao * VOXEL_VOLUME_ML), 2),
        "aorta_fn_voxels": int(fn_ao),
        "aorta_precision": round(float(ao_prec), 4),
        "aorta_recall": round(float(ao_rec), 4),
        "aorta_specificity": round(float(ao_spec), 5),
        "aorta_dice": round(float(ao_dice), 4),
    })

    print(f"  Acc: {case_acc*100:.2f}% | Mean Dice: {(pa_dice + ao_dice)/2:.4f} | PA FP: {fp_pa} voxels ({fp_pa*VOXEL_VOLUME_ML:.2f} mL) | Aorta FP: {fp_ao} voxels ({fp_ao*VOXEL_VOLUME_ML:.2f} mL)")

# =============================================================
# Global Metrics Calculation
# =============================================================
total_voxels = np.sum(cm_global)
correct_voxels = np.trace(cm_global)
overall_accuracy = correct_voxels / total_voxels

# Calculate per-class metrics One-vs-Rest
class_stats = []
for c in range(NUM_CLASSES):
    tp = cm_global[c, c]
    fp = np.sum(cm_global[:, c]) - tp
    fn = np.sum(cm_global[c, :]) - tp
    tn = total_voxels - (tp + fp + fn)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    class_stats.append({
        "class_name": CLASS_NAMES[c],
        "class_index": c,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_positive_volume_ml": round(float(fp * VOXEL_VOLUME_ML), 2),
        "false_negatives": int(fn),
        "false_negative_volume_ml": round(float(fn * VOXEL_VOLUME_ML), 2),
        "true_negatives": int(tn),
        "precision_ppv": round(float(precision), 4),
        "recall_sensitivity": round(float(recall), 4),
        "specificity_tnr": round(float(specificity), 5),
        "false_positive_rate_fpr": round(float(fpr), 6),
        "false_discovery_rate_fdr": round(float(fdr), 4),
        "false_negative_rate_fnr": round(float(fnr), 4),
        "dice_similarity_f1": round(float(dice), 4),
        "iou_jaccard": round(float(iou), 4),
    })

# Vessel-only aggregate (PA + Aorta)
vessel_tp = class_stats[1]["true_positives"] + class_stats[2]["true_positives"]
vessel_fp = class_stats[1]["false_positives"] + class_stats[2]["false_positives"]
vessel_fn = class_stats[1]["false_negatives"] + class_stats[2]["false_negatives"]
vessel_prec = vessel_tp / (vessel_tp + vessel_fp)
vessel_rec = vessel_tp / (vessel_tp + vessel_fn)
vessel_dice = (2.0 * vessel_tp) / (2.0 * vessel_tp + vessel_fp + vessel_fn)
mean_vessel_dice = (class_stats[1]["dice_similarity_f1"] + class_stats[2]["dice_similarity_f1"]) / 2.0

# =============================================================
# Visualizations
# =============================================================

# 1. Raw Confusion Matrix Plot
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(cm_global, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm_global.shape[1]),
       yticks=np.arange(cm_global.shape[0]),
       xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
       title="Multi-Class Confusion Matrix (Raw Voxel Counts)\nNon-Contrast CT Test Set (5 Cases)",
       ylabel="True Ground-Truth Class",
       xlabel="Predicted Class")
plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations
thresh = cm_global.max() / 2.
for i in range(cm_global.shape[0]):
    for j in range(cm_global.shape[1]):
        val = cm_global[i, j]
        ax.text(j, i, f"{val:,d}",
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=11, weight="bold")
plt.tight_layout()
raw_cm_path = OUTPUT_DIR / "confusion_matrix_raw.png"
fig.savefig(raw_cm_path, dpi=200)
plt.close(fig)

# 2. Normalized (Recall / Sensitivity) Confusion Matrix Plot
cm_norm_recall = cm_global.astype("float") / cm_global.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(cm_norm_recall * 100, interpolation='nearest', cmap=plt.cm.Greens, vmin=0, vmax=100)
ax.figure.colorbar(im, ax=ax, format='%.0f%%')
ax.set(xticks=np.arange(cm_norm_recall.shape[1]),
       yticks=np.arange(cm_norm_recall.shape[0]),
       xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
       title="Confusion Matrix Normalized by True Class (Recall %)\nTrue Positive Rate per Anatomy",
       ylabel="True Ground-Truth Class",
       xlabel="Predicted Class")
plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")

thresh_norm = 50.0
for i in range(cm_norm_recall.shape[0]):
    for j in range(cm_norm_recall.shape[1]):
        val = cm_norm_recall[i, j] * 100
        ax.text(j, i, f"{val:.2f}%",
                ha="center", va="center",
                color="white" if val > thresh_norm else "black",
                fontsize=11, weight="bold")
plt.tight_layout()
norm_cm_path = OUTPUT_DIR / "confusion_matrix_normalized.png"
fig.savefig(norm_cm_path, dpi=200)
plt.close(fig)

# 3. Bar Chart of Comprehensive Metrics
fig, ax = plt.subplots(figsize=(10, 5.5))
metrics_to_plot = ["Precision (PPV)", "Recall (Sens)", "Specificity", "Dice (F1)", "IoU (Jaccard)"]
pa_vals = [class_stats[1]["precision_ppv"], class_stats[1]["recall_sensitivity"], class_stats[1]["specificity_tnr"], class_stats[1]["dice_similarity_f1"], class_stats[1]["iou_jaccard"]]
ao_vals = [class_stats[2]["precision_ppv"], class_stats[2]["recall_sensitivity"], class_stats[2]["specificity_tnr"], class_stats[2]["dice_similarity_f1"], class_stats[2]["iou_jaccard"]]

x = np.arange(len(metrics_to_plot))
width = 0.35

rects1 = ax.bar(x - width/2, pa_vals, width, label="Pulmonary Artery", color="#3b82f6", edgecolor="black", linewidth=0.8)
rects2 = ax.bar(x + width/2, ao_vals, width, label="Aorta", color="#10b981", edgecolor="black", linewidth=0.8)

ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11, weight="bold")
ax.set_title("Vessel Segmentation Performance Metrics on Non-Contrast CT Test Set", fontsize=13, weight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot, fontsize=10, weight="bold")
ax.set_ylim(0.0, 1.1)
ax.axhline(0.80, color="#ef4444", linestyle="--", linewidth=1.5, label="Target Threshold (0.80)")
ax.legend(loc="lower right", fontsize=10, frameon=True)
ax.grid(axis="y", linestyle=":", alpha=0.6)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.4f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, weight="bold")

autolabel(rects1)
autolabel(rects2)
plt.tight_layout()
chart_path = OUTPUT_DIR / "performance_metrics_chart.png"
fig.savefig(chart_path, dpi=200)
plt.close(fig)

# =============================================================
# Save Output Reports (JSON & CSV)
# =============================================================
full_report = {
    "total_voxels_evaluated": int(total_voxels),
    "correct_voxels": int(correct_voxels),
    "overall_voxel_accuracy": round(float(overall_accuracy), 6),
    "mean_vessel_dice": round(float(mean_vessel_dice), 4),
    "confusion_matrix_3x3": {
        "classes": CLASS_NAMES,
        "matrix_rows_are_true_cols_are_pred": cm_global.tolist(),
    },
    "class_metrics": class_stats,
    "per_case_breakdown": per_case_metrics,
}

json_path = OUTPUT_DIR / "confusion_matrix_report.json"
with open(json_path, "w") as f:
    json.dump(full_report, f, indent=2)

csv_path = OUTPUT_DIR / "per_case_confusion_metrics.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(per_case_metrics[0].keys()))
    writer.writeheader()
    writer.writerows(per_case_metrics)

summary_csv_path = OUTPUT_DIR / "voxel_metrics_summary.csv"
with open(summary_csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(class_stats[0].keys()))
    writer.writeheader()
    writer.writerows(class_stats)

# =============================================================
# Console Summary Display
# =============================================================
print()
print("=" * 75)
print("             TEST SET ACCURACY & CONFUSION MATRIX REPORT")
print("=" * 75)
print(f"  Total Test Cases:            5 cases ({total_voxels:,} voxels)")
print(f"  Overall Voxel Accuracy:      {overall_accuracy*100:.3f}% ({correct_voxels:,} / {total_voxels:,} voxels correct)")
print(f"  Overall Mean Vessel Dice:    {mean_vessel_dice:.4f}  (Target >= 0.80: PASSED)")
print()
print("  -------------------------------------------------------------------------")
print("  3x3 CONFUSION MATRIX (Rows = True Ground Truth, Columns = Prediction):")
print("  -------------------------------------------------------------------------")
print(f"  {'True \\ Pred':<22} | {'Pred Background':<16} | {'Pred PA':<12} | {'Pred Aorta':<12}")
print("  " + "-" * 70)
for r in range(NUM_CLASSES):
    print(f"  {CLASS_NAMES[r]:<22} | {cm_global[r, 0]:<16,d} | {cm_global[r, 1]:<12,d} | {cm_global[r, 2]:<12,d}")
print("  " + "-" * 70)
print()

print("  -------------------------------------------------------------------------")
print("  FALSE POSITIVE & ERROR ANALYSIS BY CLASS:")
print("  -------------------------------------------------------------------------")
for cs in class_stats:
    cname = cs['class_name']
    tp = cs['true_positives']
    fp = cs['false_positives']
    fn = cs['false_negatives']
    tn = cs['true_negatives']
    fp_vol = cs['false_positive_volume_ml']
    prec = cs['precision_ppv']
    rec = cs['recall_sensitivity']
    spec = cs['specificity_tnr']
    fpr = cs['false_positive_rate_fpr']
    dice = cs['dice_similarity_f1']
    iou = cs['iou_jaccard']

    print(f"  >>> [{cname.upper()}]:")
    print(f"      - True Positives (TP):     {tp:,} voxels ({tp * VOXEL_VOLUME_ML:.2f} mL)")
    print(f"      - False Positives (FP):    {fp:,} voxels ({fp_vol:.2f} mL)")
    print(f"      - False Negatives (FN):    {fn:,} voxels ({fn * VOXEL_VOLUME_ML:.2f} mL)")
    print(f"      - False Positive Rate:     {fpr * 100:.4f}% (Fraction of non-{cname} incorrectly predicted as {cname})")
    print(f"      - Precision (PPV):         {prec*100:.2f}% (When AI predicts {cname}, it is right {prec*100:.2f}% of the time)")
    print(f"      - Recall / Sensitivity:    {rec*100:.2f}% (Captures {rec*100:.2f}% of all true {cname} anatomy)")
    print(f"      - Specificity (TNR):       {spec*100:.3f}%")
    print(f"      - Dice / F1 Score:         {dice:.4f}")
    print(f"      - IoU / Jaccard:           {iou:.4f}")
    print()

print("  -------------------------------------------------------------------------")
print("  PER-CASE FALSE POSITIVES & DICE BREAKDOWN:")
print("  -------------------------------------------------------------------------")
print(f"  {'Case ID':<20} | {'Mean Dice':<10} | {'PA Dice':<8} | {'PA FP (mL)':<11} | {'Aorta Dice':<10} | {'Aorta FP (mL)':<14}")
print("  " + "-" * 82)
for p in per_case_metrics:
    print(f"  {p['case_id']:<20} | {p['mean_dice']:<10.4f} | {p['pa_dice']:<8.4f} | {p['pa_fp_volume_ml']:<11.2f} | {p['aorta_dice']:<10.4f} | {p['aorta_fp_volume_ml']:<14.2f}")
print("  " + "-" * 82)
print()
print("  Outputs saved to:")
print(f"    - Confusion Matrix (Raw):        {raw_cm_path}")
print(f"    - Confusion Matrix (Normalized): {norm_cm_path}")
print(f"    - Performance Chart:             {chart_path}")
print(f"    - Full Metrics JSON:             {json_path}")
print(f"    - Per-Case CSV:                  {csv_path}")
print("=" * 75)
