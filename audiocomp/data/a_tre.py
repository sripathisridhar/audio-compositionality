"""A-TRE-10k dataset adapter (HuggingFace ``chuyangchenn/a-tre-10k``).

Wraps :func:`datasets.load_dataset` and yields ``(waveform_(T,), metadata)``
torch items ready for the encoder + composition head used by the A-TRE
benchmark. Audio is auto-resampled to the encoder's ``target_sr`` and
padded/trimmed to ``duration * target_sr`` samples.
"""

from __future__ import annotations

from typing import Literal

from datasets import Audio, load_dataset
from torch.utils.data import Dataset

from ._utils import pad_or_trim

REPO_ID = "chuyangchenn/a-tre-10k"

TIMBRE_CLASSES: list[str] = [f"t{i}" for i in range(1, 9)]
PITCH_CLASSES: list[str] = [f"p{i}" for i in range(1, 9)]
RATE_CLASSES: list[str] = [f"r{i}" for i in range(1, 9)]
AMPLITUDE_CLASSES: list[str] = [f"a{i}" for i in range(1, 9)]

# Full ordered attribute vocabulary used by the A-TRE composition head.
# Index 0 is reserved for padding; attribute labels map to indices 1…32.
ATTRIBUTES: list[str] = TIMBRE_CLASSES + PITCH_CLASSES + RATE_CLASSES + AMPLITUDE_CLASSES


class ATREDataset(Dataset):
    """A-TRE benchmark dataset.

    Parameters
    ----------
    split:
        ``"train"``, ``"val"``, or ``"test"``.
    target_sr:
        Sample rate the encoder expects; HF auto-resamples on access.
    duration:
        Clip length in seconds; clips are padded or trimmed to
        ``duration * target_sr`` samples.
    cache_dir:
        Optional override for the HF datasets cache root.

    Each item is ``(waveform, metadata)``:

    * ``waveform`` — ``torch.Tensor`` of shape ``(target_sr * duration,)``,
      mono float32.
    * ``metadata`` — list of per-source dicts:
      ``{"timbre_label", "pitch_label", "rate_label", "amplitude_label"}``.
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        target_sr: int = 32_000,
        duration: int = 10,
        cache_dir: str | None = None,
    ) -> None:
        self.target_samples = duration * target_sr
        ds = load_dataset(REPO_ID, split=split, cache_dir=cache_dir)
        # Precompute metadata as a Python list so CachedDatasetView can read
        # it without triggering audio decoding. Cheap (~1 MB for 10k items).
        self.metadata: list = list(ds["metadata"])
        self.hf = ds.cast_column("audio", Audio(sampling_rate=target_sr))

    def __len__(self) -> int:
        return len(self.hf)

    def __getitem__(self, idx: int):
        ex = self.hf[idx]
        wave = ex["audio"].get_all_samples().data.squeeze(0)   # (1, T) -> (T,)
        wave = pad_or_trim(wave, self.target_samples)
        return wave, ex["metadata"]
