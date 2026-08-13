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

Optional public downloads (CT-RATE is gated on Hugging Face):

```bash
python src/download_datasets.py --list
python src/download_datasets.py fumpe
```

## Results (held-out non-contrast, n=3)

Contrast-trained baseline vs contrast-removal adapted SegResNet. Val cases were never used in training.

| Model | PA Dice | Aorta Dice | Mean Dice |
| --- | ---: | ---: | ---: |
| Baseline | 0.000 | 0.068 | 0.034 |
| Adapted | 0.000 | **0.349** | **0.175** |

Ground-truth mean equivalent diameters at the max-PA slice: MPA **56.9 mm**, aorta **42.6 mm**, MPA:aorta ratio **1.32**. The adapted model recovers aorta (Dice 0.07 → 0.35) but still misses PA on this short run, so predicted ratio is not yet clinically usable. Next: more epochs, PA-specific sampling, and the remaining Drive zip volumes if Google split the export.

Full numbers: `results/challenge_metrics.json`. Overlays: `results/figures/`. Report: `results/Ingenium_Hackathon_Results.pdf`.


## Notebook

`Copy_of_Hackathon_Notebook.ipynb` is the starter notebook adapted to run locally (Colab Drive mounts removed). The challenge code path is `src/run_challenge.py`.
