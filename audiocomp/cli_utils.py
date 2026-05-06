"""Shared CLI helpers: config parsing + cache-aware dataset views."""

from __future__ import annotations

import os
import sys

import torch
from torch.utils.data import Dataset

from .cache import CachedDatasetView, EmbeddingCache, build_cache, make_fingerprint
from .config import load_config


def parse_args(doc: str | None = None) -> tuple[str, dict, str]:
    """Parse ``<config> [overrides...]`` from ``sys.argv``.

    Returns ``(config_path, cfg, device)``.
    """
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        print(doc or "Usage: <script> <config> [key.subkey=value ...]", file=sys.stderr)
        raise SystemExit(1)
    config_path, overrides = args[0], args[1:]
    cfg = load_config(config_path, overrides)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return config_path, cfg, device


def get_view(
    encoder,
    cfg: dict,
    dataset: Dataset,
    dataset_name: str,
    split: str,
    embedding_shape: tuple[int, ...],
    *,
    device: str,
) -> CachedDatasetView:
    """Return a :class:`CachedDatasetView` for *dataset*; auto-build cache on miss.

    Existing cache files are reused (idempotent). Fingerprint mismatches
    raise :class:`audiocomp.cache.CacheFingerprintMismatch` — delete the
    offending ``.pt`` and re-run.
    """
    model_cfg = cfg["model"]
    init_kwargs = {
        k: v for k, v in model_cfg.items() if k != "name" and v is not None
    }
    fp = make_fingerprint(
        model_name=model_cfg["name"],
        sample_rate=encoder.sample_rate,
        embedding_dim=encoder.embedding_dim,
        init_kwargs=init_kwargs,
    )
    path = os.path.join(cfg["cache"]["dir"], model_cfg["name"], f"{dataset_name}-{split}.pt")
    cache = EmbeddingCache(path, len(dataset), embedding_shape, fp)
    cache.load()
    if cache.is_complete:
        print(f"  cache hit:  {path}  ({cache.n_filled} entries)")
    else:
        print(f"  cache miss: {path}  (building...)")
        build_cache(
            encoder, dataset, cache,
            batch_size=cfg["train"]["batch_size"],
            device=device,
            num_workers=cfg.get("data", {}).get("num_workers", 4),
        )
    return CachedDatasetView(dataset, cache)
