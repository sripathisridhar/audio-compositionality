"""Dasheng and MiDashengLM-7B audio encoders, scored on final-layer tokens averaged over time."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from ._safetensors import load_prefixed
from .base import AudioEncoder


class _DashengTokens(AudioEncoder):
    """Shared ``embed`` for a ``dasheng`` model stored as ``self._model``."""

    sample_rate: int = 16_000

    def embed(self, audio: torch.Tensor) -> torch.Tensor:
        m = self._model
        with torch.no_grad():
            # A (B, 1, T) input sets the -120 dB spectrogram floor per clip instead of per batch.
            x = m.init_bn(m.front_end(audio.unsqueeze(1)))
            return m.forward_spectrogram(x).mean(1)


class DashengEncoder(_DashengTokens):
    """Released Dasheng (``size`` is base, 0.6B or 1.2B), downloaded into ``$TORCH_HOME`` on first use."""

    def __init__(self, size: str) -> None:
        import dasheng

        ctor = {"base": dasheng.dasheng_base, "0.6B": dasheng.dasheng_06B, "1.2B": dasheng.dasheng_12B}
        self._model = ctor[size]().eval()
        self.embedding_dim = self._model.embed_dim


class MiDashengLMAudioEncoder(_DashengTokens):
    """MiDashengLM-7B-0804 audio encoder; ``model_dir`` holds its config.json, index and audio_encoder shard."""

    def __init__(self, model_dir: str) -> None:
        from dasheng.pretrained.pretrained import Dasheng

        cfg = json.loads((Path(model_dir) / "config.json").read_text())["audio_encoder_config"]
        self._model = Dasheng(embed_dim=cfg["embed_dim"], depth=cfg["depth"], num_heads=cfg["num_heads"])
        state = {
            re.sub(r"^init_bn\.", "init_bn.1.", k): v
            for k, v in load_prefixed(model_dir, "audio_encoder.").items()
        }
        # nn.Module's loader is called directly because dasheng's override drops the key report.
        missing, unexpected = torch.nn.Module.load_state_dict(self._model, state, strict=False)
        if unexpected or any(not k.startswith("front_end.") for k in missing):
            raise RuntimeError(f"audio_encoder key mismatch: missing={missing}, unexpected={unexpected}")
        self._model.eval()
        self.embedding_dim = self._model.embed_dim
