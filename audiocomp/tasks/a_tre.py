"""A-TRE: Audio Tree Reconstruction Error.

A lightweight composition head ``g_θ`` is trained to reconstruct the
encoder's scene embedding from the scene's symbolic attribute metadata
alone. Each source ``s_n = [t_n, p_n, r_n, a_n]`` becomes the sum of four
learned attribute token vectors; a single-layer Transformer with a
prepended ``[CLS]`` aggregates the source representations. The ``[CLS]``
output is trained to align with the (frozen, cached) encoder embedding via
cosine similarity.

At test time the A-TRE score is

    A-TRE(X) = cos(z, ẑ)  ∈ [−1, 1]

This module is pure: it consumes a :class:`audiocomp.cache.CachedDatasetView`
that yields ``(z, metadata)`` items where ``z`` is shape ``(D,)``.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ..data import ATTRIBUTES, collate_fn


# ---------------------------------------------------------------------------
# Composition model
# ---------------------------------------------------------------------------

class TREModel(nn.Module):
    """Composition model g_θ for the A-TRE benchmark.

    Each source ``s_n = [t_n, p_n, r_n, a_n]`` is represented as the sum of
    four learnable attribute token vectors:

        E(s_n) = Q_{t_n} + Q_{p_n} + Q_{r_n} + Q_{a_n}

    A single-layer Transformer with a prepended ``[CLS]`` aggregates the
    source tokens; the ``[CLS]`` output is the predicted embedding ẑ.

    Parameters
    ----------
    embedding_dim:
        Dimensionality D of the attribute tokens and encoder output.
        Must match the upstream encoder's ``embedding_dim``.
    attributes:
        Ordered list of all attribute label strings (32 total: 8 timbre +
        8 pitch + 8 rate + 8 amplitude). Index 0 is reserved for padding.
    max_objects:
        Maximum number of sources per scene (4 in the paper).
    """

    def __init__(
        self,
        embedding_dim: int,
        attributes: list[str] = ATTRIBUTES,
        max_objects: int = 4,
    ) -> None:
        super().__init__()
        self._vocab = {label: i + 1 for i, label in enumerate(attributes)}
        self._pad_idx = 0
        self._max_obj = max_objects
        self.embedding_dim = embedding_dim

        self.emb = nn.Embedding(len(attributes) + 1, embedding_dim, padding_idx=0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=1, batch_first=True
        )

    def forward(self, batch_metadata: tuple[list[dict[str, Any]], ...]) -> torch.Tensor:
        """Return the ``[CLS]`` output ẑ of shape ``(B, D)``."""
        B = len(batch_metadata)
        device = self.cls_token.device

        token_ids = torch.full(
            (B, self._max_obj, 4), self._pad_idx, dtype=torch.long, device=device
        )
        key_padding_mask = torch.ones(
            B, self._max_obj + 1, dtype=torch.bool, device=device
        )
        key_padding_mask[:, 0] = False  # always attend to CLS

        for i, obj_list in enumerate(batch_metadata):
            for j, obj in enumerate(obj_list[: self._max_obj]):
                token_ids[i, j, 0] = self._vocab.get(obj["timbre_label"], self._pad_idx)
                token_ids[i, j, 1] = self._vocab.get(obj["pitch_label"], self._pad_idx)
                token_ids[i, j, 2] = self._vocab.get(obj["rate_label"], self._pad_idx)
                token_ids[i, j, 3] = self._vocab.get(obj["amplitude_label"], self._pad_idx)
                key_padding_mask[i, j + 1] = False

        obj_repr = self.emb(token_ids).sum(dim=2)               # (B, max_obj, D)
        cls = self.cls_token.expand(B, 1, self.embedding_dim)
        seq = torch.cat([cls, obj_repr], dim=1)                  # (B, max_obj+1, D)
        out = self.encoder_layer(seq, src_key_padding_mask=key_padding_mask)
        return out[:, 0]                                         # (B, D)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopper:
    """Patience-based early stopping with EMA-smoothed validation loss.

    Parameters
    ----------
    patience:
        Epochs without improvement before stopping.
    ema_alpha:
        Decay applied to the running estimate. Higher ⇒ smoother.
        ``ema = old * ema_alpha + new * (1 - ema_alpha)``.
    """

    def __init__(self, patience: int = 4, ema_alpha: float = 0.3) -> None:
        self.patience = patience
        self.ema_alpha = ema_alpha
        self.best = float("inf")
        self.count = 0
        self._ema: float | None = None

    def step(self, val: float) -> tuple[bool, bool]:
        self._ema = (
            val if self._ema is None
            else self._ema * self.ema_alpha + val * (1 - self.ema_alpha)
        )
        if self._ema < self.best:
            self.best = self._ema
            self.count = 0
            return True, False
        self.count += 1
        return False, self.count >= self.patience


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train(
    train_view: Dataset,
    val_view: Dataset,
    embedding_dim: int,
    *,
    output_dir: str,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    patience: int = 4,
    ema_alpha: float = 0.3,
    seed: int = 42,
    device: str = "cpu",
) -> str:
    """Train a :class:`TREModel` head against cached encoder embeddings.

    Parameters
    ----------
    train_view, val_view:
        :class:`audiocomp.cache.CachedDatasetView` instances wrapping the
        train/val splits of A-TRE; each item is ``(z, metadata)`` with
        ``z`` shape ``(D,)``.
    embedding_dim:
        Encoder output dim ``D`` — must match what's in the cache.
    output_dir:
        Directory to save the best checkpoint (``<dir>/tre_head.pth``).
    epochs, batch_size, lr, weight_decay, grad_clip:
        Standard training hyperparams.
    patience, ema_alpha:
        Early-stopping config (see :class:`EarlyStopper`).
    seed:
        Seeds ``torch.manual_seed`` for reproducible head init + DataLoader.
    device:
        ``cuda`` or ``cpu``.

    Returns
    -------
    str
        Absolute path to the best checkpoint saved.
    """
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "tre_head.pth")

    model = TREModel(embedding_dim=embedding_dim).to(device).train()

    train_loader = DataLoader(
        train_view, batch_size=batch_size, shuffle=True, drop_last=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        val_view, batch_size=batch_size, shuffle=False, drop_last=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=(device != "cpu"),
    )

    optimizer = optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.1,
    )
    stopper = EarlyStopper(patience=patience, ema_alpha=ema_alpha)

    best_val = float("inf")
    best_epoch = 1
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = _run_epoch(
            model, train_loader, optimizer, device, train=True, grad_clip=grad_clip,
        )
        scheduler.step()

        model.eval()
        val_loss = _run_epoch(
            model, val_loader, optimizer, device, train=False, grad_clip=grad_clip,
        )

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:3d}/{epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}")

        improved, stop = stopper.step(val_loss)
        if improved:
            best_val, best_epoch = val_loss, epoch
            torch.save({"model": model.state_dict(), "embedding_dim": embedding_dim}, ckpt_path)
            print(f"  ✓ saved (val={best_val:.4f})")
        if stop:
            print(f"early stop at epoch {epoch}; best epoch {best_epoch}, val={best_val:.4f}")
            break

    return ckpt_path


def _run_epoch(model, loader, optimizer, device, *, train: bool, grad_clip: float) -> float:
    total = 0.0
    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for emb_batch, metadata in loader:
            with torch.no_grad():
                enc_emb = F.normalize(emb_batch.to(device), dim=-1)
            tre_emb = F.normalize(model(metadata), dim=-1)
            loss = (1 - F.cosine_similarity(enc_emb, tre_emb, dim=-1)).mean()

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total += loss.item()
    return total / max(len(loader), 1)


def evaluate(
    test_view: Dataset,
    tre_ckpt: str,
    embedding_dim: int,
    *,
    batch_size: int = 64,
    device: str = "cpu",
) -> pd.DataFrame:
    """Score every test scene with a trained head and return the results.

    Parameters
    ----------
    test_view:
        :class:`audiocomp.cache.CachedDatasetView` over the A-TRE test split.
    tre_ckpt:
        Path to the checkpoint produced by :func:`train`.
    embedding_dim:
        Encoder output dim.
    batch_size:
        Eval batch size (no encoder forward, so memory-bound).
    device:
        ``cuda`` or ``cpu``.

    Returns
    -------
    pd.DataFrame
        One row per scene (dataset order); columns ``idx`` and ``score``
        (cosine similarity ∈ [−1, 1]).
    """
    model = TREModel(embedding_dim=embedding_dim).to(device).eval()
    ckpt = torch.load(tre_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])

    loader = DataLoader(
        test_view, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=(device != "cpu"),
    )

    all_scores: list[float] = []
    with torch.no_grad():
        for emb_batch, metadata in loader:
            enc_emb = F.normalize(emb_batch.to(device), dim=-1)
            tre_emb = F.normalize(model(metadata), dim=-1)
            scores = F.cosine_similarity(enc_emb, tre_emb, dim=-1).clamp(-1, 1)
            all_scores.extend(scores.tolist())

    return pd.DataFrame({
        "idx": list(range(len(test_view))),
        "score": all_scores,
    })
