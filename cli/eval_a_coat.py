"""Evaluate an audio encoder on the A-COAT compositionality benchmark.

Zero-shot — no training required. The encoder cache is built on first run
and reused thereafter (set ``cache.dir`` in config to share across runs).

Usage::

    python cli/eval_a_coat.py configs/random.yaml
    python cli/eval_a_coat.py configs/random.yaml train.batch_size=128
"""
from __future__ import annotations

import json
import os

from audiocomp.cli_utils import get_view, parse_args
from audiocomp.data import ACOATDataset
from audiocomp.models import get_model
from audiocomp.tasks import evaluate_a_coat


def main() -> None:
    config_path, cfg, device = parse_args(__doc__)
    print(f"Model : {cfg['model']['name']}")
    print(f"Config: {config_path}")
    print(f"Device: {device}")

    encoder = get_model(**{k: v for k, v in cfg["model"].items() if v is not None})

    ds = ACOATDataset(target_sr=encoder.sample_rate)
    view = get_view(
        encoder, cfg, ds, "a-coat-2k", "test",
        embedding_shape=(4, encoder.embedding_dim),
        device=device,
    )
    df = evaluate_a_coat(view, batch_size=cfg["train"]["batch_size"], device=device)

    score_mean = float(df["score"].mean())
    score_std = float(df["score"].std())
    print(f"\n=== A-COAT ({cfg['model']['name']}) ===")
    print(f"  score (↑): {score_mean:.4f} ± {score_std:.4f}")
    print(f"  N        : {len(df)}")

    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "a_coat_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": cfg["model"]["name"],
            "task": "a_coat",
            "n": len(df),
            "score_mean": score_mean,
            "score_std": score_std,
            "items": df.to_dict(orient="records"),
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
