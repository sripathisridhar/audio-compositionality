"""A-COAT: Audio Compositional Object Algebra Test (zero-shot).

Given a quadruple ``(A, B, C, D)`` where ``B = A ∪ T`` and ``D = C ∪ T`` for
a shared transformation set ``T``, a compositionally-consistent encoder
should produce residual vectors ``z_B − z_A`` and ``z_D − z_C`` that point in
the same direction — the "signature" of adding ``T`` should not depend on
the base scene.

**Metric (A-COAT score):**

    A-COAT(A,B,C,D) = cos(z_B − z_A,  z_D − z_C)  ∈ [−1, 1]

This module is pure: it consumes a :class:`audiocomp.cache.CachedDatasetView`
that yields ``(z, metadata)`` items where ``z`` is shape ``(4, D)``
(A/B/C/D embeddings stacked).
"""

from __future__ import annotations

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..data import collate_fn


def evaluate(
    view: Dataset,
    *,
    batch_size: int = 64,
    device: str = "cpu",
) -> pd.DataFrame:
    """Score every quadruple in *view* and return a results DataFrame.

    Parameters
    ----------
    view:
        A :class:`audiocomp.cache.CachedDatasetView` wrapping ``ACOATDataset``,
        so each item is ``(embeddings_(4, D), metadata)``.
    batch_size:
        Eval batch size (no GPU forward inside the loop, so this is just
        memory shape for vector arithmetic).
    device:
        ``cpu`` is usually fine — only cosine + norms run here.

    Returns
    -------
    pd.DataFrame
        One row per quadruple (in dataset order), columns ``idx`` and
        ``score`` (cosine similarity ∈ [−1, 1]).
    """
    loader = DataLoader(
        view, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=0, pin_memory=(device != "cpu"),
    )

    all_scores: list[float] = []
    with torch.no_grad():
        for embs, _ in loader:
            embs = embs.to(device)                            # (B, 4, D)
            zA, zB, zC, zD = embs[:, 0], embs[:, 1], embs[:, 2], embs[:, 3]
            u = zB - zA                                       # add-T from A
            v = zD - zC                                       # add-T from C
            scores = F.cosine_similarity(u, v, dim=-1).clamp(-1, 1)
            all_scores.extend(scores.tolist())

    return pd.DataFrame({
        "idx": list(range(len(view))),
        "score": all_scores,
    })
