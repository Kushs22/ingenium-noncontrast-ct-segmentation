# Life sciences challenge — IngeniumAI

**Read this before you touch a file.** AI Hackathon, 13–14 August 2026.

## The challenge

Extend anatomical segmentation to **non-contrast** chest CT.

IngeniumAI's product (PHAST) already detects signs of pulmonary hypertension from
**contrast** CT pulmonary angiograms, at or above the performance of a specialist
chest radiologist. The problem is timing: CTPAs are acquired midway through the
diagnostic pathway. Non-contrast CT is acquired far earlier and is not currently
exploited. Pulmonary hypertension takes roughly **2.5 years** to diagnose from first
GP presentation.

So the task is not "detect the disease". It is the component that unlocks doing so
earlier: **reliably segment the relevant anatomical structures in non-contrast CT**,
robust to variation in scan quality and patient anatomy, with output usable downstream.

Success, in IngeniumAI's words: an anatomical segmentation model that works on
non-contrast scans and shows clear potential to slot into their existing stack.

Contacts: Jeff Clark (CEO) jeff@ingeniumai.com · Aaron Mir (CTO) aaronm@ingeniumai.com.
A technical mentor is on site both days and you can ask clinical questions.

## The three datasets

| | CT-RATE | FUMPE | CAD-PE |
|---|---|---|---|
| Patients | 21,304 | 35 | 91 |
| Contrast | **Non-contrast** | Contrast CTPA | Contrast CTPA |
| Age range | 18–102 | 24–82 | Unknown |
| Slice thickness | 0.035–6 mm | ≤1 mm (2 cases @2 mm) | ≤1.5 mm |
| Content | Normal + 18 chest abnormalities | Contains PE | Contains PE |
| Licence | CC BY-NC-SA 4.0 | CC BY 4.0 | CC BY 4.0 |
| Access | Hugging Face, **gated** | Kaggle / figshare | IEEE DataPort, free account |

**CT-RATE is the target domain** — it is the non-contrast set. The two contrast sets
are where you have strong labels and where IngeniumAI's existing capability lives, so
they are your source domain. That asymmetry *is* the challenge.

Sources:
- CT-RATE — https://huggingface.co/datasets/ibrahimhamamci/CT-RATE · paper https://arxiv.org/pdf/2403.17834
- FUMPE — https://pmc.ncbi.nlm.nih.gov/articles/PMC6122162/ · figshare DOI 10.6084/m9.figshare.c.4107803
- CAD-PE — https://ieee-dataport.org/open-access/cad-pe · paper https://arxiv.org/abs/2003.13440

## Getting the data

`Data Examples - *` folders in here hold sample scans so you can start immediately
without downloading anything.

For more, run `download_datasets.py` (in this folder):

    pip install huggingface_hub kaggle
    python download_datasets.py --list
    python download_datasets.py ct-rate --volumes 20
    python download_datasets.py fumpe
    python download_datasets.py cad-pe

**Do before 8 August, not on the day:**
1. Hugging Face account, then **accept the CT-RATE terms on the dataset page** — it is
   gated and the accept is manual. Create a read token, `huggingface-cli login`.
2. Kaggle account, Settings → API → Create New Token, save `kaggle.json` to `~/.kaggle/`.
3. IEEE account for CAD-PE (free, membership not required).

Do not attempt the full CT-RATE. It is 50,188 volumes and over 14 million slices.
Twenty volumes is plenty for a two-day proof of concept.

## Two things that will catch you out

**CT-RATE is CC BY-NC-SA 4.0 — non-commercial.** You are asked to pitch a
commercialisation plan. Writing a plan is not commercial use, so this is not a
blocker, but a judge may well ask. Have the answer ready: a production system would
be trained on licensed or clinical data, and CT-RATE demonstrates the method.

**Reports are machine-translated.** CT-RATE was collected in Turkey and the radiology
reports were machine-translated into English then corrected by bilingual medical
students. Treat report text as good but not gold.

## Also in this folder

`IngeniumAI - Hackathon Problem Statement & Datasets.pptx` — Aaron's deck, including
the dataset comparison table this README is drawn from.

`IngeniumAI - abstract A322.full.pdf` — the British Thoracic Society abstract from
IngeniumAI's clinical collaborators (Rossdale et al., Royal United Hospitals Bath and
University of Bath). This is the clinical grounding: automated 3D main pulmonary artery
diameter and MPA:aorta ratio from CTPA, predictive of pulmonary hypertension at right
heart catheterisation. **Read it.** It tells you which structures matter and why.

IngeniumAI have said they will also provide labels, a trained segmentation model for
contrast scans, and a starter Python notebook.

## Handling rules

These are public research datasets, not confidential client data, so the rules are
lighter than the other two challenges — but the licences still bind you. Attribute the
source, do not re-publish the data, and respect the non-commercial clause on CT-RATE.
