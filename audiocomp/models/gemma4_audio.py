"""Gemma 4 E4B audio tower (USM conformer), scored on encoder or ``embed_audio`` tokens averaged over valid frames."""

from __future__ import annotations

import torch

from ._safetensors import load_prefixed
from .base import AudioEncoder


class Gemma4AudioEncoder(AudioEncoder):
    """``model_dir`` holds google/gemma-4-E4B-it's config.json, processor_config.json and model.safetensors; ``tap`` is ``encoder`` or ``embed_audio``."""

    sample_rate: int = 16_000

    def __init__(self, model_dir: str, tap: str) -> None:
        from transformers import Gemma4Config
        from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
        from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel, Gemma4MultimodalEmbedder

        cfg = Gemma4Config.from_pretrained(model_dir)
        self._tower = Gemma4AudioModel._from_config(cfg.audio_config, dtype=torch.float32)
        self._tower.load_state_dict(load_prefixed(model_dir, "model.audio_tower."), strict=True)
        self._tower.eval()
        self._fe = Gemma4AudioFeatureExtractor.from_pretrained(model_dir)
        self._head = None
        if tap == "encoder":
            self.embedding_dim = cfg.audio_config.hidden_size
            self._tower.layers[-1].register_forward_hook(self._keep_tokens)
        elif tap == "embed_audio":
            self.embedding_dim = cfg.text_config.hidden_size
            self._head = Gemma4MultimodalEmbedder(cfg.audio_config, cfg.text_config)
            self._head.load_state_dict(load_prefixed(model_dir, "model.embed_audio."), strict=True)
            self._head.eval()
        else:
            raise ValueError(f"tap must be 'encoder' or 'embed_audio', got {tap!r}")

    def _keep_tokens(self, module, inputs, output) -> None:
        self._tokens = output

    def embed(self, audio: torch.Tensor) -> torch.Tensor:
        feats = self._fe(
            [a.cpu().numpy() for a in audio],
            sampling_rate=self.sample_rate,
            truncation=False,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = self._tower(feats["input_features"].to(audio.device), feats["input_features_mask"].to(audio.device))
            tokens = self._tokens if self._head is None else self._head(out.last_hidden_state)
        valid = out.attention_mask.unsqueeze(-1).to(tokens.dtype)
        return (tokens * valid).sum(1) / valid.sum(1)
