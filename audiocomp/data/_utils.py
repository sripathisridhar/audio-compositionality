from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate


def pad_or_trim(wave: torch.Tensor, n: int) -> torch.Tensor:
    """Right-pad or right-trim a 1-D waveform to exactly ``n`` samples."""
    if wave.shape[0] < n:
        return F.pad(wave, (0, n - wave.shape[0]))
    return wave[:n]


def collate_fn(batch):
    """Stack tensors; keep metadata as a tuple (variable-length structure).

    Works for both A-TRE items ``(waveform_(T,), metadata_list)`` and A-COAT
    items ``(waveforms_(4,T), metadata_dict)`` — only the tensor shape differs.
    Yields ``(stacked_tensor, tuple_of_metadata)``.
    """
    tensors, metadatas = zip(*batch)
    return default_collate(list(tensors)), tuple(metadatas)
