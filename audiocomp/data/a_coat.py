"""A-COAT-2k dataset adapter (HuggingFace ``chuyangchenn/a-coat-2k``).

Wraps :func:`datasets.load_dataset` and yields ``(waveforms_(4,T), metadata)``
torch items for the zero-shot A-COAT benchmark. The four waveforms are
``A``, ``B``, ``C``, ``D`` in order, where ``B = A ∪ T`` and ``D = C ∪ T``
share a transformation set ``T``.
"""

from __future__ import annotations

import torch
from datasets import Audio, load_dataset
from torch.utils.data import Dataset

from ._utils import pad_or_trim

REPO_ID = "chuyangchenn/a-coat-2k"

ROLES = ("A", "B", "C", "D")


class ACOATDataset(Dataset):
    """A-COAT benchmark dataset (zero-shot, single ``test`` split).

    Parameters
    ----------
    target_sr:
        Sample rate the encoder expects; HF auto-resamples on access.
    duration:
        Clip length in seconds; clips are padded or trimmed to
        ``duration * target_sr`` samples.
    cache_dir:
        Optional override for the HF datasets cache root.

    Each item is ``(waveforms, metadata)``:

    * ``waveforms`` — ``torch.Tensor`` of shape
      ``(4, target_sr * duration)``, mono float32; the four roles A/B/C/D
      are stacked in that order.
    * ``metadata`` — dict ``{"A": [...], "C": [...], "T": [...]}`` where each
      value is a list of per-source attribute dicts.
    """

    def __init__(
        self,
        target_sr: int = 32_000,
        duration: int = 10,
        cache_dir: str | None = None,
    ) -> None:
        self.target_samples = duration * target_sr
        ds = load_dataset(REPO_ID, split="test", cache_dir=cache_dir)
        # Precompute metadata as a Python list so CachedDatasetView can read
        # it without triggering audio decoding.
        self.metadata: list = list(ds["metadata"])
        for role in ROLES:
            ds = ds.cast_column(role, Audio(sampling_rate=target_sr))
        self.hf = ds

    def __len__(self) -> int:
        return len(self.hf)

    def __getitem__(self, idx: int):
        ex = self.hf[idx]
        waves = [
            pad_or_trim(ex[role].get_all_samples().data.squeeze(0), self.target_samples)
            for role in ROLES
        ]
        return torch.stack(waves), ex["metadata"]
