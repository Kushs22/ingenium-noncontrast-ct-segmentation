#!/usr/bin/env python3
"""Non-contrast aorta + PA segmentation — IngeniumAI hackathon pipeline.

Domain adaptation (not plain supervised fine-tuning on the target):
  Source = unused contrast CTPA. During training, vessel HU is randomly
  *removed* (flattened to soft-tissue) or left intact, so the network cannot
  rely on iodinated lumen brightness.
  Target-train = held-in non-contrast cases with the same HU randomization.
  Target-val  = held-out non-contrast cases (never used for training).

Outputs Dice, MPA:aorta diameters/ratio, overlay figures, JSON, and a PDF.
"""
from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    EnsureType,
    LoadImage,
    Orientation,
    ScaleIntensityRange,
    Spacing,
)
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
DRIVE = ROOT / "Hackathon Challenge"
CKPT_BASE = DRIVE / "data" / "baseline_model.pt"
if not CKPT_BASE.exists():
    CKPT_BASE = ROOT / "data" / "baseline_model.pt"
CKPT_ADAPT = ROOT / "data" / "adapted_model.pt"
RESULTS = ROOT / "results"
FIGDIR = RESULTS / "figures"
CACHE = RESULTS / "cache"

SPACING = (3.0, 3.0, 3.0)
PATCH = (64, 64, 64)
HU_MIN, HU_MAX = -200.0, 400.0
NUM_CLASSES = 3
CLASS_NAMES = ["background", "pulmonary_artery", "aorta"]
N_VAL = 3
EPOCHS = 80
STEPS_PER_EPOCH = 32
LR = 1e-4
SEED = 42
# Positive-patch sampling: most crops are vessel-centred, and most of those
# are centred on pulmonary_artery (class 1) because PA was completely missed.
POS_PROB = 0.90
PA_POS_FRAC = 0.70
# CE+Dice class weights: [background, pulmonary_artery, aorta]
CLASS_WEIGHT = (0.4, 2.5, 1.0)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pair_cases(img_dir: Path, lab_dir: Path) -> list[tuple[Path, Path]]:
    out = []
    if not img_dir.exists():
        return out
    for img in sorted(img_dir.glob("*.nii.gz")):
        if img.name.endswith("_label.nii.gz"):
            continue
        case = img.name[: -len(".nii.gz")]
        lab = lab_dir / f"{case}_label.nii.gz"
        if lab.exists():
            out.append((img, lab))
    return out


def crop_both(img: np.ndarray, lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = img > 0
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img, lab
    mn, mx = coords.min(0), coords.max(0) + 1
    sl = tuple(slice(int(a), int(b)) for a, b in zip(mn, mx))
    return img[sl], lab[sl]


def load_resampled(image_path: Path, label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    key = f"{image_path.stem}.npz"
    cache_path = CACHE / key
    if cache_path.exists():
        z = np.load(cache_path)
        return z["image"], z["label"]
    img_tf = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=SPACING, mode="bilinear"),
            ScaleIntensityRange(a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
            EnsureType(data_type="numpy"),
        ]
    )
    lab_tf = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=SPACING, mode="nearest"),
            EnsureType(data_type="numpy"),
        ]
    )
    image = np.asarray(img_tf(str(image_path)))[0]
    label = np.asarray(lab_tf(str(label_path)))[0].astype(np.int16)
    # align shapes if spacing rounding differs by 1 voxel
    m = tuple(min(a, b) for a, b in zip(image.shape, label.shape))
    image, label = image[tuple(slice(0, s) for s in m)], label[tuple(slice(0, s) for s in m)]
    image, label = crop_both(image, label)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, image=image.astype(np.float16), label=label)
    return image.astype(np.float32), label


def build_model(dev: torch.device) -> SegResNet:
    return SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=NUM_CLASSES,
        init_filters=16,
        dropout_prob=0.2,
    ).to(dev)


def load_weights(path: Path, dev: torch.device) -> SegResNet:
    model = build_model(dev)
    model.load_state_dict(torch.load(str(path), map_location=dev))
    return model


def contrast_invariant(image: torch.Tensor, label: torch.Tensor, force_remove: bool = False) -> torch.Tensor:
    """Flatten vessel HU (contrast removal) or boost it (synthetic contrast)."""
    img = image.clone()
    vessel = label > 0
    if int(vessel.sum()) < 20:
        return img
    if force_remove:
        mode = "flatten"
    else:
        mode = random.choice(["flatten", "flatten", "contrast", "identity"])
    if mode == "flatten":
        tissue_mask = (label == 0) & (img > 0.12) & (img < 0.55)
        tissue = img[tissue_mask]
        mu = float(tissue.mean()) if tissue.numel() else 0.35
        img[vessel] = mu + torch.randn_like(img[vessel]) * 0.025
    elif mode == "contrast":
        img[vessel] = (img[vessel] + random.uniform(0.18, 0.48)).clamp(0, 1)
    return img.clamp(0, 1)


def random_patch(image: np.ndarray, label: np.ndarray, force_remove: bool):
    d, h, w = image.shape
    pd, ph, pw = PATCH
    pos = None
    if random.random() < POS_PROB:
        pa = np.argwhere(label == 1)
        if len(pa) and random.random() < PA_POS_FRAC:
            pos = pa
        else:
            pos = np.argwhere(label > 0)
        if len(pos) == 0:
            pos = None
    if pos is not None:
        z, y, x = pos[random.randrange(len(pos))]
        z0 = min(max(int(z) - pd // 2, 0), max(d - pd, 0))
        y0 = min(max(int(y) - ph // 2, 0), max(h - ph, 0))
        x0 = min(max(int(x) - pw // 2, 0), max(w - pw, 0))
    else:
        z0 = random.randint(0, max(d - pd, 0))
        y0 = random.randint(0, max(h - ph, 0))
        x0 = random.randint(0, max(w - pw, 0))
    img_p = image[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw]
    lab_p = label[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw]
    img_t = torch.from_numpy(np.ascontiguousarray(img_p, dtype=np.float32))[None]
    lab_t = torch.from_numpy(np.ascontiguousarray(lab_p, dtype=np.int64))[None]
    pad = []
    for size, need in zip(img_t.shape[1:][::-1], PATCH[::-1]):
        extra = max(need - size, 0)
        pad.extend([0, extra])
    if any(pad):
        img_t = F.pad(img_t, pad)
        lab_t = F.pad(lab_t, pad)
    if random.random() < 0.5:
        img_t = torch.flip(img_t, [1])
        lab_t = torch.flip(lab_t, [1])
    if random.random() < 0.5:
        img_t = torch.flip(img_t, [2])
        lab_t = torch.flip(lab_t, [2])
    img_t = contrast_invariant(img_t, lab_t, force_remove=force_remove)
    return img_t, lab_t


@torch.no_grad()
def predict_volume(model: torch.nn.Module, image: np.ndarray, dev: torch.device) -> np.ndarray:
    model.eval()
    inp = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None].to(dev)
    pred = sliding_window_inference(inp, PATCH, sw_batch_size=1, predictor=model, overlap=0.25)
    return torch.argmax(pred, dim=1)[0].cpu().numpy()


def dice_binary(pred: np.ndarray, gt: np.ndarray, klass: int) -> float:
    p = pred == klass
    g = gt == klass
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0
    return float(2 * inter / denom)


def keep_largest(pred: np.ndarray, klass: int) -> np.ndarray:
    out = pred.copy()
    blob = pred == klass
    if blob.sum() == 0:
        return out
    labeled, n = ndimage.label(blob)
    if n <= 1:
        return out
    sizes = ndimage.sum(blob, labeled, range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    out[(pred == klass) & (labeled != keep)] = 0
    return out


def equivalent_diameter_mm(mask2d: np.ndarray, spacing_xy=(3.0, 3.0)) -> float:
    area_mm2 = float(mask2d.sum()) * spacing_xy[0] * spacing_xy[1]
    if area_mm2 <= 0:
        return 0.0
    return 2.0 * math.sqrt(area_mm2 / math.pi)


def clinical_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Equivalent diameters at the axial slice of maximum GT PA area.

    Arrays are (X, Y, Z) in RAS; axial slices are vol[:, :, z].
    """
    pa_area = (gt == 1).reshape(-1, gt.shape[2]).sum(0)
    z = int(pa_area.argmax()) if pa_area.max() > 0 else gt.shape[2] // 2
    out = {"reference_slice_z": z}
    for name, klass in [("pa", 1), ("aorta", 2)]:
        out[f"{name}_gt_diam_mm"] = equivalent_diameter_mm(gt[:, :, z] == klass)
        out[f"{name}_pred_diam_mm"] = equivalent_diameter_mm(pred[:, :, z] == klass)
    out["mpa_aorta_ratio_gt"] = (
        out["pa_gt_diam_mm"] / out["aorta_gt_diam_mm"] if out["aorta_gt_diam_mm"] else None
    )
    out["mpa_aorta_ratio_pred"] = (
        out["pa_pred_diam_mm"] / out["aorta_pred_diam_mm"] if out["aorta_pred_diam_mm"] else None
    )
    return out


def overlay_figure(image, gt, pred, z, title, path: Path) -> None:
    sl = image[:, :, z]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, mask, name in zip(axes, [gt[:, :, z] if gt is not None else None, pred[:, :, z], None], ["Ground truth", "Prediction", "CT only"]):
        ax.imshow(np.rot90(sl), cmap="gray", vmin=0, vmax=1)
        if mask is not None:
            overlay = np.zeros((*sl.shape, 4))
            overlay[mask == 1] = (0.95, 0.85, 0.15, 0.45)
            overlay[mask == 2] = (0.15, 0.75, 0.75, 0.45)
            ax.imshow(np.rot90(overlay))
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_pdf(metrics: dict, fig_paths: list[Path], pdf_path: Path) -> None:
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        b, a = metrics["baseline_mean"], metrics["adapted_mean"]
        cg, cb, ca = metrics["clinical_mean_gt"], metrics["clinical_mean_baseline"], metrics["clinical_mean_adapted"]
        lines = [
            "IngeniumAI Hackathon — Non-contrast aorta & PA segmentation",
            f"Compiled {metrics['timestamp']}",
            "",
            "Task: segment pulmonary artery and aorta on non-contrast chest CT",
            "so PHAST can run earlier than contrast CTPA.",
            "",
            "Method: contrast-removal domain adaptation (not plain supervised).",
            "Source = unused contrast CTPA. Vessel HU is flattened to soft-tissue",
            "range so SegResNet must use shape and location. Mixed with held-in",
            "non-contrast cases under the same HU randomization.",
            f"Val = {metrics['n_val']} held-out non-contrast cases (never trained on).",
            "",
            f"Device: {metrics['device']}",
            f"Non-contrast paired: {metrics['n_noncontrast']}   Contrast paired: {metrics['n_contrast']}",
            f"Epochs: {metrics['epochs']}   steps/epoch: {metrics.get('steps_per_epoch', STEPS_PER_EPOCH)}",
            f"PA-focused sampling: pos={metrics.get('pos_prob', POS_PROB)}  PA frac={metrics.get('pa_pos_frac', PA_POS_FRAC)}",
            f"Class weights (bg, PA, aorta): {metrics.get('class_weight', CLASS_WEIGHT)}",
            "",
            "Mean hold-out Dice (non-contrast val)",
            f"  Baseline  PA {b['dice_pa']:.3f}   aorta {b['dice_aorta']:.3f}   mean {b['dice_mean']:.3f}",
            f"  Adapted   PA {a['dice_pa']:.3f}   aorta {a['dice_aorta']:.3f}   mean {a['dice_mean']:.3f}",
            "",
            "Clinical mean equivalent diameter at max-PA GT slice",
            f"  GT     MPA {cg['pa_mm']:.1f} mm   aorta {cg['aorta_mm']:.1f} mm   ratio {cg['ratio']}",
            f"  Base   MPA {cb['pa_mm']:.1f} mm   aorta {cb['aorta_mm']:.1f} mm   ratio {cb['ratio']}",
            f"  Adapt  MPA {ca['pa_mm']:.1f} mm   aorta {ca['aorta_mm']:.1f} mm   ratio {ca['ratio']}",
            "",
            "Licence: CT-RATE is CC BY-NC-SA 4.0 (demonstration only).",
            "A production PHAST model would be retrained on licensed/clinical data.",
            "FUMPE / CAD-PE contrast sets are CC BY 4.0.",
        ]
        ax.text(0.04, 0.96, "\n".join(lines), va="top", family="monospace", fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)
        for fp in fig_paths:
            if not fp.exists():
                continue
            img = plt.imread(fp)
            f, ax = plt.subplots(figsize=(11, 6))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(fp.stem.replace("_", " "))
            pdf.savefig(f, bbox_inches="tight")
            plt.close(f)


def mean_dict(items: list[dict], keys: list[str]) -> dict:
    out = {}
    for k in keys:
        vals = [x[k] for x in items if x.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


def main() -> None:
    set_seed()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    dev = device()
    print("device", dev)

    extra_nc = pair_cases(ROOT / "non_contrast_images", ROOT / "non_contrast_labels")
    nc = pair_cases(DRIVE / "non_contrast_images", DRIVE / "non_contrast_labels")
    # de-duplicate by case id
    seen = {p[0].name for p in nc}
    for pair in extra_nc:
        if pair[0].name not in seen:
            nc.append(pair)
    contrast = pair_cases(DRIVE / "unused_contrast_images", DRIVE / "unused_contrast_labels")
    print(f"non-contrast paired {len(nc)}  contrast paired {len(contrast)}")
    if len(nc) < 2:
        sys.exit("Need at least 2 non-contrast labelled cases from the Drive zip.")

    rng = np.random.default_rng(SEED)
    order = np.arange(len(nc))
    rng.shuffle(order)
    n_val = min(N_VAL, max(1, len(nc) // 3))
    val_idx = set(order[:n_val].tolist())
    val_cases = [nc[i] for i in range(len(nc)) if i in val_idx]
    train_nc = [nc[i] for i in range(len(nc)) if i not in val_idx]
    print("VAL", [p.name for p, _ in val_cases])
    print("TRAIN_NC", [p.name for p, _ in train_nc])
    print("SOURCE_CONTRAST", [p.name for p, _ in contrast])

    print("caching resampled volumes...")
    train_store = []  # (img, lab, force_remove)
    for p, l in train_nc:
        print(" ", p.name)
        img, lab = load_resampled(p, l)
        train_store.append((img.astype(np.float32), lab, False))
    for p, l in contrast:
        print(" ", p.name, "(contrast source)")
        img, lab = load_resampled(p, l)
        train_store.append((img.astype(np.float32), lab, True))
    val_store = []
    for p, l in val_cases:
        print("  val", p.name)
        img, lab = load_resampled(p, l)
        val_store.append((p.stem, img.astype(np.float32), lab))

    print("baseline inference on val...")
    baseline = load_weights(CKPT_BASE, dev).eval()
    base_rows = []
    adapted_rows = []  # filled later
    figs: list[Path] = []

    def eval_model(model, tag: str):
        rows = []
        for i, (name, img, lab) in enumerate(val_store):
            pred = keep_largest(keep_largest(predict_volume(model, img, dev), 1), 2)
            row = {
                "case": name,
                "dice_pa": dice_binary(pred, lab, 1),
                "dice_aorta": dice_binary(pred, lab, 2),
            }
            row["dice_mean"] = (row["dice_pa"] + row["dice_aorta"]) / 2
            clin = clinical_metrics(pred, lab)
            row.update(clin)
            rows.append(row)
            z = clin["reference_slice_z"]
            fp = FIGDIR / f"{tag}_{name}.png"
            overlay_figure(img, lab, pred, z, f"{tag}  {name}  PA Dice {row['dice_pa']:.2f}  aorta {row['dice_aorta']:.2f}", fp)
            figs.append(fp)
            print(f"  {tag} {name} PA {row['dice_pa']:.3f} aorta {row['dice_aorta']:.3f} ratio_pred {clin['mpa_aorta_ratio_pred']}")
        return rows

    base_rows = eval_model(baseline, "baseline")

    print("training contrast-invariant adaptation...")
    adapted = load_weights(CKPT_BASE, dev)
    adapted.train()
    opt = torch.optim.AdamW(adapted.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    class_w = torch.tensor(CLASS_WEIGHT, dtype=torch.float32, device=dev)
    loss_fn = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        weight=class_w,
    )
    history = []
    t0 = datetime.now(timezone.utc)
    for epoch in range(EPOCHS):
        running = 0.0
        for _ in range(STEPS_PER_EPOCH):
            img, lab, force = random.choice(train_store)
            img_p, lab_p = random_patch(img, lab, force_remove=force)
            img_p = img_p.unsqueeze(0).to(dev)
            lab_p = lab_p.unsqueeze(0).float().to(dev)
            opt.zero_grad(set_to_none=True)
            out = adapted(img_p)
            loss = loss_fn(out, lab_p)
            loss.backward()
            opt.step()
            running += float(loss.item())
        sched.step()
        avg = running / STEPS_PER_EPOCH
        history.append(avg)
        elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
        print(
            f"  epoch {epoch+1}/{EPOCHS} loss {avg:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
            f"elapsed {elapsed/60:.1f} min",
            flush=True,
        )

    CKPT_ADAPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapted.state_dict(), str(CKPT_ADAPT))
    adapted.eval()
    adapted_rows = eval_model(adapted, "adapted")

    def summarise(rows):
        return {
            "dice_pa": float(np.mean([r["dice_pa"] for r in rows])),
            "dice_aorta": float(np.mean([r["dice_aorta"] for r in rows])),
            "dice_mean": float(np.mean([r["dice_mean"] for r in rows])),
        }

    def clin_mean(rows, pred_key=True):
        pa = np.mean([r["pa_gt_diam_mm"] if not pred_key else r["pa_pred_diam_mm"] for r in rows])
        ao = np.mean([r["aorta_gt_diam_mm"] if not pred_key else r["aorta_pred_diam_mm"] for r in rows])
        ratios = [r["mpa_aorta_ratio_gt"] if not pred_key else r["mpa_aorta_ratio_pred"] for r in rows]
        ratios = [x for x in ratios if x]
        return {
            "pa_mm": float(pa),
            "aorta_mm": float(ao),
            "ratio": float(np.mean(ratios)) if ratios else None,
        }

    metrics = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "device": str(dev),
        "method": "contrast-removal domain adaptation from unused contrast CTPA + HU-randomized non-contrast train cases; PA-focused patch sampling and class-weighted DiceCE",
        "n_noncontrast": len(nc),
        "n_contrast": len(contrast),
        "n_train_nc": len(train_nc),
        "n_val": len(val_cases),
        "val_cases": [p.name for p, _ in val_cases],
        "train_nc_cases": [p.name for p, _ in train_nc],
        "contrast_source_cases": [p.name for p, _ in contrast],
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "pos_prob": POS_PROB,
        "pa_pos_frac": PA_POS_FRAC,
        "class_weight": list(CLASS_WEIGHT),
        "train_loss": history,
        "baseline_per_case": base_rows,
        "adapted_per_case": adapted_rows,
        "baseline_mean": summarise(base_rows),
        "adapted_mean": summarise(adapted_rows),
        "clinical_mean_gt": clin_mean(base_rows, pred_key=False),
        "clinical_mean_baseline": clin_mean(base_rows, pred_key=True),
        "clinical_mean_adapted": clin_mean(adapted_rows, pred_key=True),
        "licence_note": "CT-RATE CC BY-NC-SA 4.0 is demonstration-only; production PHAST would retrain on licensed/clinical data.",
        "commercialisation": "Pitch the method and PHAST integration. Do not commercialise CT-RATE-trained weights. Retrain on hospital or licensed data before product use.",
        "zip_source": "Hackathon Challenge-20260813T145659Z-1-001.zip",
        "figures": [str(p.relative_to(ROOT)) for p in figs],
    }
    (RESULTS / "challenge_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    write_pdf(metrics, figs, RESULTS / "Ingenium_Hackathon_Results.pdf")
    print("MEAN baseline", metrics["baseline_mean"])
    print("MEAN adapted", metrics["adapted_mean"])
    print("wrote", RESULTS / "challenge_metrics.json")
    print("wrote", RESULTS / "Ingenium_Hackathon_Results.pdf")


if __name__ == "__main__":
    main()
