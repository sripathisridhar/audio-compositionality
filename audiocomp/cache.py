"""Persistent per-item embedding cache.

Audio encoders are frozen during A-TRE training and A-COAT evaluation, so
each item's embedding is a deterministic function of ``(encoder, audio)``.
Recomputing the encoder forward pass every epoch / every re-run wastes GPU
time, so this module precomputes embeddings once and persists them on disk.

The cache also makes stochastic baselines (``random``) reproducible across
runs — without it, every fresh process would draw different noise.

Cache layout::

    <cache_dir>/<encoder_name>/<dataset>-<split>.pt   # torch.save'd dict

Each ``.pt`` is a wrapper dict::

    {
        "schema":      1,
        "fingerprint": {... encoder identity ...},
        "embeddings":  Tensor of shape (N, *embedding_shape),
        "filled":      bool Tensor of shape (N,),
    }

The fingerprint records *which* encoder produced the cached embeddings:
encoder name, sample rate, embedding dim, and the kwargs passed to the
encoder's ``__init__`` (e.g. ``seed`` for ``RandomEncoder``,
``checkpoint`` for a user-added encoder). Loading raises
:class:`CacheFingerprintMismatch` if the on-disk fingerprint disagrees
with the expected one — typically because the user changed a config key
behind the same encoder name. Delete the offending ``.pt`` and re-run.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .data._utils import collate_fn

CACHE_SCHEMA_VERSION = 1


class CacheFingerprintMismatch(RuntimeError):
    """On-disk cache's fingerprint does not match the expected one."""


def make_fingerprint(
    *,
    model_name: str,
    sample_rate: int,
    embedding_dim: int,
    init_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the identity record stored alongside a cache file.

    ``init_kwargs`` captures whatever was passed to the encoder's
    ``__init__`` (e.g. ``{"seed": 42}`` for ``RandomEncoder``, or
    ``{"checkpoint": "/path/to/weights.pt"}`` for a user-added encoder);
    changing any of them invalidates the cache.

    Note: kwargs whose values are file paths are stored as paths, not
    file-content hashes. If you swap weights at the same path, you must
    delete the cache yourself — the fingerprint won't catch it.
    """
    return {
        "model_name": model_name,
        "sample_rate": sample_rate,
        "embedding_dim": embedding_dim,
        "init_kwargs": dict(init_kwargs or {}),
    }


class EmbeddingCache:
    """Dense ``(N, *embedding_shape)`` tensor persisted as a single ``.pt``.

    Parameters
    ----------
    path:
        File where the cache lives on disk. Parent directory is created on
        :meth:`save`.
    n_items:
        Total number of items in the dataset (must match ``len(dataset)``).
    embedding_shape:
        Per-item embedding shape, e.g. ``(768,)`` for A-TRE or ``(4, 768)``
        for A-COAT (one embedding per role A/B/C/D).
    fingerprint:
        Encoder identity from :func:`make_fingerprint`.
    dtype:
        Tensor dtype; defaults to ``torch.float32``.
    """

    def __init__(
        self,
        path: str,
        n_items: int,
        embedding_shape: tuple[int, ...],
        fingerprint: dict[str, Any],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self.embedding_shape = tuple(embedding_shape)
        self._tensor = torch.zeros((n_items, *self.embedding_shape), dtype=dtype)
        self._filled = torch.zeros(n_items, dtype=torch.bool)

    # -- I/O ---------------------------------------------------------------

    def load(self) -> bool:
        """Populate from disk if a matching cache file exists.

        Returns ``True`` if loaded, ``False`` if the file is missing.
        Raises :class:`CacheFingerprintMismatch` if the file exists but its
        fingerprint or shape disagrees with what was expected.
        """
        if not os.path.isfile(self.path):
            return False
        blob = torch.load(self.path, map_location="cpu", weights_only=False)
        if not isinstance(blob, dict) or "embeddings" not in blob:
            raise CacheFingerprintMismatch(
                f"Cache file {self.path} is in an unrecognised format; delete and re-run."
            )
        on_disk = blob.get("fingerprint")
        if on_disk != self.fingerprint:
            raise CacheFingerprintMismatch(
                f"Cache fingerprint mismatch at {self.path}.\n"
                f"  on disk : {on_disk}\n"
                f"  expected: {self.fingerprint}\n"
                f"Delete the file (or its parent dir) and re-run to regenerate."
            )
        if blob["embeddings"].shape != self._tensor.shape:
            raise CacheFingerprintMismatch(
                f"Cache shape mismatch at {self.path}: on-disk "
                f"{tuple(blob['embeddings'].shape)} vs expected {tuple(self._tensor.shape)}."
            )
        self._tensor = blob["embeddings"]
        self._filled = blob["filled"]
        return True

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        blob = {
            "schema": CACHE_SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "embeddings": self._tensor,
            "filled": self._filled,
        }
        tmp = self.path + ".tmp"
        torch.save(blob, tmp)
        os.replace(tmp, self.path)

    # -- access ------------------------------------------------------------

    def __len__(self) -> int:
        return int(self._tensor.shape[0])

    def __contains__(self, idx: int) -> bool:
        return bool(self._filled[idx].item())

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._tensor[idx]

    def put(self, idx: int, embedding: torch.Tensor) -> None:
        self._tensor[idx] = embedding.detach().to("cpu")
        self._filled[idx] = True

    def put_batch(self, start: int, embeddings: torch.Tensor) -> None:
        """Fill ``[start, start + B)`` from a batched embedding tensor."""
        B = embeddings.shape[0]
        self._tensor[start : start + B] = embeddings.detach().to("cpu")
        self._filled[start : start + B] = True

    @property
    def is_complete(self) -> bool:
        return bool(self._filled.all().item())

    @property
    def n_filled(self) -> int:
        return int(self._filled.sum().item())


class CachedDatasetView(Dataset):
    """Wraps a dataset; yields ``(cache[idx], metadata[idx])`` instead of audio.

    Lets training/eval loops be model-agnostic: the same loop runs on raw
    audio (when no cache is built) or on cached embeddings (after running
    :func:`build_cache`), with the only change being which Dataset is passed
    to the DataLoader.
    """

    def __init__(self, base: Dataset, cache: EmbeddingCache) -> None:
        if len(base) != len(cache):
            raise ValueError(
                f"Dataset and cache have different lengths: "
                f"{len(base)} vs {len(cache)}."
            )
        self.base = base
        self.cache = cache

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        return self.cache[idx], self.base.metadata[idx]


def build_cache(
    encoder,
    dataset: Dataset,
    cache: EmbeddingCache,
    *,
    batch_size: int = 32,
    device: str = "cpu",
    num_workers: int = 0,
) -> EmbeddingCache:
    """Iterate *dataset* in order, run *encoder*, fill *cache*, and save it.

    Handles both A-TRE-style items (``waveform`` shape ``(T,)``) and
    A-COAT-style items (``waveform`` shape ``(K, T)``, e.g. K=4 roles)
    by detecting batch dimensionality at runtime.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
    )

    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    n_total = len(dataset)
    print(f"  cache: {cache.path}  ({cache.n_filled}/{n_total} filled)")

    idx = 0
    with torch.no_grad():
        for waves, _ in loader:
            B = waves.shape[0]
            if waves.dim() == 3:
                _, K, T = waves.shape
                flat = waves.reshape(B * K, T).to(device, non_blocking=True)
                embs = encoder.embed(flat).reshape(B, K, -1).cpu()
            else:
                embs = encoder.embed(waves.to(device, non_blocking=True)).cpu()
            cache.put_batch(idx, embs)
            idx += B

    cache.save()
    print(f"  cache: wrote {cache.n_filled}/{n_total} entries to {cache.path}")
    return cache
