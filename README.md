# Non-contrast aorta & pulmonary-artery segmentation

Hackathon work for **IngeniumAI / PHAST**: move anatomical segmentation from contrast CTPA onto **non-contrast** chest CT so pulmonary-hypertension signs (MPA diameter and MPA:aorta ratio) can be measured earlier in the pathway.

This is **not** a disease detector. It is the missing component: reliable aorta + pulmonary-artery masks on non-contrast CT.

## Method

The Ingenium baseline SegResNet is trained on contrast CT and fails on non-contrast scans (vessel lumen no longer lights up). We do **not** only supervised-fine-tune on the target labels.

**Contrast-removal domain adaptation**

1. **Source domain** — unused contrast CTPA (`unused_contrast_images` / `unused_contrast_labels` from the Drive pack).
2. During training, vessel HU is randomly **flattened to soft-tissue range** (and sometimes boosted), so the network cannot use iodinated brightness.
3. Held-in **non-contrast** cases get the same HU randomization.
4. Held-out **non-contrast** cases are used only for evaluation.
5. **PA-focused patch sampling** (70% of positive crops centred on pulmonary artery) and class-weighted DiceCE (`bg/PA/aorta = 0.4/2.5/1.0`) so the thinner PA class is not ignored.

Architecture stays Ingenium’s SegResNet (`init_filters=16`, 3 classes: background / PA / aorta) so the head can slot into PHAST.

## Data

From `Hackathon Challenge-20260813T145659Z-1-001.zip` (Google Drive export the Colab notebook mounted at `/content/drive/MyDrive/Hackathon Challenge`):

| Split | Role | Typical contents in this export |
| --- | --- | --- |
| `non_contrast_images` + `non_contrast_labels` | **Target** (CT-RATE style) | Paired non-contrast volumes |
| `unused_contrast_images` + `unused_contrast_labels` | **Source** (CAD-PE / FUM-PE style) | Paired contrast CTPA |
| `data/baseline_model.pt` | Contrast-trained baseline | ~18 MB |

Raw `.nii.gz` volumes are **not** in git (too large). Place the unzipped `Hackathon Challenge/` folder next to this README.

CT-RATE is **CC BY-NC-SA 4.0**. FUMPE and CAD-PE are CC BY 4.0.

## Licence / commercialisation

Writing a commercialisation plan is not commercial use of CT-RATE. A judge may still ask.

**Answer:** CT-RATE only demonstrates the method. A production PHAST model would be retrained on licensed or hospital data. Do not ship CT-RATE-trained weights in a product.

## Setup (Python 3.12, macOS MPS or CUDA)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Unzip the Drive pack:

```bash
unzip "Hackathon Challenge-*.zip"
```

## Run

```bash
source .venv/bin/activate
python src/run_challenge.py
```

Score frozen checkpoints on an extra labelled non-contrast folder (no retraining):

```bash
python src/run_challenge.py --eval-only \
  --images "non-contrast_test_set_images 2" \
  --labels "non-contrast_test_set_labels"
```

Optional public downloads (CT-RATE is gated on Hugging Face):

```bash
python src/download_datasets.py --list
python src/download_datasets.py fumpe
```

## Results (held-out non-contrast, n=3)

Contrast-trained baseline vs contrast-removal adapted SegResNet (PA-focused sampling, class-weighted DiceCE). Val cases were never used in training.

| Model | Epochs | PA Dice | Aorta Dice | Mean Dice | Pred MPA:aorta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | — | 0.000 | 0.068 | 0.034 | — |
| Adapted (previous) | 5 | 0.000 | 0.349 | 0.175 | — (PA missed) |
| Adapted (this run) | **80** | **0.692** | **0.823** | **0.758** | **1.50** (GT 1.32) |

Ground-truth mean equivalent diameters at the max-PA slice: MPA **56.9 mm**, aorta **42.6 mm**, MPA:aorta ratio **1.32**. Adapted predictions: MPA **62.3 mm**, aorta **42.9 mm**, ratio **1.50**.

Full numbers: `results/challenge_metrics.json`. Overlays: `results/figures/`. Report: `results/Ingenium_Hackathon_Results.pdf`.

## External test set (n=5, never trained on)

Organisers provided a further labelled non-contrast set (`train_10035`–`train_10039`). The **same frozen 80-epoch checkpoint** was scored against the original baseline (no extra training).

| Model | PA Dice | Aorta Dice | Mean Dice | Pred MPA:aorta |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.000 | 0.045 | 0.023 | — |
| Adapted (80 ep., frozen) | **0.703** | **0.725** | **0.714** | **1.41** (GT **1.64**) |

GT mean diameters: MPA **61.0 mm**, aorta **39.6 mm**. Weakest case is `train_10036` (PA Dice 0.39); the other four PA Dice values are 0.65–0.85.

JSON: `results/testset_metrics.json`. PDF: `results/Ingenium_Hackathon_TestSet_Results.pdf`.


## Notebook

`Copy_of_Hackathon_Notebook.ipynb` is the starter notebook adapted to run locally (Colab Drive mounts removed). The challenge code path is `src/run_challenge.py`.
