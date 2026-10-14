"""M2D-CLAP (github.com/nttcslab/m2d) backbone tokens and text-aligned CLAP audio embedding."""

from __future__ import annotations

import importlib
import sys

import torch

from .base import AudioEncoder


def _portable_m2d(m2d_repo: str, checkpoint: str, flat_features: bool):
    """Build PortableM2D from an m2d checkout and a ``checkpoint-*.pth`` kept inside its release directory."""
    sys.path.insert(0, m2d_repo)
    portable_m2d = importlib.import_module("examples.portable_m2d")
    # A fresh Config is passed because PortableM2D mutates its shared default instance.
    cfg = portable_m2d.Config()
    return portable_m2d.PortableM2D(checkpoint, cfg=cfg, flat_features=flat_features).eval()


class M2DCLAPEncoder(AudioEncoder):
    """Backbone patch tokens averaged; takes an m2d checkout and a v0.5.0 ``m2d_clap_*`` checkpoint."""

    sample_rate: int = 16_000

    def __init__(self, m2d_repo: str, checkpoint: str) -> None:
        self._model = _portable_m2d(m2d_repo, checkpoint, flat_features=True)
        self.embedding_dim = self._model.cfg.feature_d

    def embed(self, audio: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._model.encode(audio).mean(1)


class M2DCLAPProjEncoder(AudioEncoder):
    """Text-aligned ``encode_clap_audio`` embedding; same arguments as :class:`M2DCLAPEncoder`."""

    sample_rate: int = 16_000

    def __init__(self, m2d_repo: str, checkpoint: str) -> None:
        self._model = _portable_m2d(m2d_repo, checkpoint, flat_features=False)
        backbone = self._model.backbone
        proj = backbone.audio_proj
        if hasattr(proj, "sem_token"):
            in_dim = self.embedding_dim = proj.sem_token.shape[-1]
        else:
            in_dim, self.embedding_dim = proj[0].in_features, proj[-1].out_features
        # The CLAP head gets the layout it was trained on: flat patch tokens or stacked frequency rows.
        self._model.cfg.flat_features = in_dim == backbone.pos_embed.shape[-1]

    def embed(self, audio: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._model.encode_clap_audio(audio)
