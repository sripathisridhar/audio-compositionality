"""Baseline encoders for sanity-checking benchmark scores.

``random``       — returns random Gaussian noise; sets a lower-bound floor.
                   Seeded for reproducibility so that the embedding cache
                   built on a fresh machine matches the one built earlier.
``downsample``   — linearly resamples the waveform to a fixed-length vector;
                   tests whether a trivial signal-level representation is
                   sufficient.
"""

from __future__ import annotations

import torch
import torchaudio.functional as AF

from .base import AudioEncoder


class RandomEncoder(AudioEncoder):
    """Returns random Gaussian noise — establishes the chance-level floor.

    When a *seed* is supplied at construction, embeddings are drawn from a
    private :class:`torch.Generator` and the per-call counter is advanced
    by the batch size, so the sequence of embeddings produced across a
    deterministic-order pass through the dataset (as performed by
    :func:`audiocomp.cache.build_cache`) is fully reproducible. Without a
    seed, the embeddings depend on global RNG state — fine for one-off
    debugging but not for cached benchmarking.
    """

    sample_rate: int = 32_000
    embedding_dim: int = 768

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self._counter = 0

    def embed(self, audio: torch.Tensor) -> torch.Tensor:
        """Returns (B, 768) standard-normal noise."""
        B = audio.shape[0]
        if self._seed is None:
            return torch.randn(B, self.embedding_dim, device=audio.device)
        # Generators tied to a CUDA device cannot be seeded from CPU state
        # without a device-aware constructor; build per-call to keep this simple.
        gen = torch.Generator(device=audio.device)
        gen.manual_seed(self._seed + self._counter)
        self._counter += B
        return torch.randn(
            B, self.embedding_dim, device=audio.device, generator=gen
        )


class DownsampleEncoder(AudioEncoder):
    """Linearly resamples each waveform to a 768-dim vector."""

    sample_rate: int = 32_000
    embedding_dim: int = 768

    def embed(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        audio : (B, T) float32

        Returns
        -------
        (B, 768) float32
        """
        T = audio.shape[1]
        return AF.resample(audio, orig_freq=T, new_freq=self.embedding_dim)
