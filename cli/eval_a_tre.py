"""Evaluate a trained A-TRE head on the test split.

Run this after ``cli/train_a_tre.py``. By default reads the head from
``<output.dir>/tre_head.pth``; override with ``--tre-ckpt PATH``.

Usage::

    python cli/eval_a_tre.py configs/random.yaml
    python cli/eval_a_tre.py configs/random.yaml --tre-ckpt outputs/random/tre_head.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from audiocomp.cli_utils import get_view
from audiocomp.config import load_config
from audiocomp.data import ATREDataset
from audiocomp.models import get_model
from audiocomp.tasks import evaluate_a_tre


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="Path to model config YAML.")
    ap.add_argument("--tre-ckpt", default=None,
                    help="Path to trained TRE head (default: <output.dir>/tre_head.pth).")
    ap.add_argument("overrides", nargs="*",
                    help="Dot-notation config overrides (e.g. train.batch_size=128).")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tre_ckpt = args.tre_ckpt or os.path.join(cfg["output"]["dir"], "tre_head.pth")
    if not os.path.isfile(tre_ckpt):
        sys.exit(f"TRE head not found at {tre_ckpt}; run cli/train_a_tre.py first.")

    print(f"Model : {cfg['model']['name']}")
    print(f"Head  : {tre_ckpt}")
    print(f"Device: {device}")

    encoder = get_model(**{k: v for k, v in cfg["model"].items() if v is not None})

    test_ds = ATREDataset(split="test", target_sr=encoder.sample_rate)
    test_view = get_view(
        encoder, cfg, test_ds, "a-tre-10k", "test",
        embedding_shape=(encoder.embedding_dim,),
        device=device,
    )

    df = evaluate_a_tre(
        test_view, tre_ckpt, encoder.embedding_dim,
        batch_size=cfg["train"]["batch_size"],
        device=device,
    )

    score_mean = float(df["score"].mean())
    score_std = float(df["score"].std())
    print(f"\n=== A-TRE ({cfg['model']['name']}) ===")
    print(f"  score (↑): {score_mean:.4f} ± {score_std:.4f}")
    print(f"  N        : {len(df)}")

    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "a_tre_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": cfg["model"]["name"],
            "task": "a_tre",
            "n": len(df),
            "score_mean": score_mean,
            "score_std": score_std,
            "items": df.to_dict(orient="records"),
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
