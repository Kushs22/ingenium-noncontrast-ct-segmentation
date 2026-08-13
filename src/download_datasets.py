#!/usr/bin/env python3
"""Download public challenge datasets (CT-RATE, FUMPE, CAD-PE).

CT-RATE is gated on Hugging Face — accept the terms on
https://huggingface.co/datasets/ibrahimhamamci/CT-RATE then `huggingface-cli login`.

Usage:
    python src/download_datasets.py --list
    python src/download_datasets.py ct-rate --volumes 20
    python src/download_datasets.py fumpe
"""
from __future__ import annotations

import argparse
import json
import ssl
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public_data"

SOURCES = {
    "ct-rate": {
        "hf": "ibrahimhamamci/CT-RATE",
        "licence": "CC BY-NC-SA 4.0",
        "role": "target domain (non-contrast)",
    },
    "fumpe": {
        "figshare_collection": 4107803,
        "licence": "CC BY 4.0",
        "role": "source domain (contrast CTPA + PE)",
    },
    "cad-pe": {
        "url": "https://ieee-dataport.org/open-access/cad-pe",
        "licence": "CC BY 4.0",
        "role": "source domain (contrast CTPA + PE)",
    },
}


def _ssl_ctx():
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def list_sources() -> None:
    for name, meta in SOURCES.items():
        print(f"{name:8}  {meta['role']}  [{meta['licence']}]")


def download_ct_rate(n_volumes: int) -> None:
    dest = OUT / "ct-rate"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("Install huggingface_hub and log in after accepting CT-RATE terms.")
    print(f"Downloading up to {n_volumes} CT-RATE volumes (gated dataset)...")
    snapshot_download(
        repo_id=SOURCES["ct-rate"]["hf"],
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=["dataset/valid_fixed/*/*.nii.gz", "dataset/valid_fixed/*/*.csv"],
        max_workers=4,
    )
    print("Saved under", dest)


def download_fumpe() -> None:
    dest = OUT / "fumpe"
    dest.mkdir(parents=True, exist_ok=True)
    ctx = _ssl_ctx()
    url = "https://api.figshare.com/v2/collections/4107803/articles"
    with urllib.request.urlopen(url, timeout=30, context=ctx) as resp:
        articles = json.loads(resp.read())
    (dest / "manifest.json").write_text(json.dumps(articles, indent=2))
    print(f"FUMPE collection has {len(articles)} articles. Manifest -> {dest / 'manifest.json'}")
    print("Full DICOM volumes are large; download individual figshare files as needed.")
    print("Ground-truth article id 6265721.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", nargs="?", choices=list(SOURCES))
    p.add_argument("--list", action="store_true")
    p.add_argument("--volumes", type=int, default=20)
    args = p.parse_args()
    if args.list or not args.dataset:
        list_sources()
        return
    if args.dataset == "ct-rate":
        download_ct_rate(args.volumes)
    elif args.dataset == "fumpe":
        download_fumpe()
    elif args.dataset == "cad-pe":
        print("CAD-PE requires a free IEEE DataPort account:", SOURCES["cad-pe"]["url"])


if __name__ == "__main__":
    main()
