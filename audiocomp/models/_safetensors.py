"""Load one sub-model's tensors from a sharded or single-file Hugging Face safetensors checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def load_prefixed(model_dir: str, prefix: str) -> dict[str, torch.Tensor]:
    """Return every tensor under ``prefix`` in ``model_dir``, with ``prefix`` stripped."""
    from safetensors import safe_open

    root = Path(model_dir)
    index = root / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        shards = sorted({f for k, f in weight_map.items() if k.startswith(prefix)})
    else:
        shards = ["model.safetensors"]
    state = {}
    for shard in shards:
        with safe_open(str(root / shard), framework="pt") as f:
            state.update({k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)})
    return state
