"""Qwen3-Omni-30B-A3B audio tower (AuT), scored on ``ln_post`` or ``proj2`` tokens averaged over time."""

from __future__ import annotations

import torch

from ._safetensors import load_prefixed
from .base import AudioEncoder


class Qwen3OmniAuTEncoder(AudioEncoder):
    """``model_dir`` holds the HF repo's config.json, preprocessor_config.json, index and audio_tower shard; ``tap`` is ``ln_post`` (AuT space) or ``proj2`` (LLM input space)."""

    sample_rate: int = 16_000

    def __init__(self, model_dir: str, tap: str) -> None:
        from transformers import Qwen3OmniMoeConfig, WhisperFeatureExtractor
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoder,
            _get_feat_extract_output_lengths,
        )

        cfg = Qwen3OmniMoeConfig.from_pretrained(model_dir).thinker_config.audio_config
        self._model = Qwen3OmniMoeAudioEncoder._from_config(
            cfg, attn_implementation="sdpa", dtype=torch.float32
        )
        self._model.load_state_dict(load_prefixed(model_dir, "thinker.audio_tower."), strict=True)
        self._model.eval()
        self._fe = WhisperFeatureExtractor.from_pretrained(model_dir)
        self._out_lengths = _get_feat_extract_output_lengths
        self.embedding_dim = cfg.output_dim if tap == "proj2" else cfg.d_model
        getattr(self._model, tap).register_forward_hook(self._keep_tokens)

    def _keep_tokens(self, module, inputs, output) -> None:
        self._tokens = output

    def embed(self, audio: torch.Tensor) -> torch.Tensor:
        feats = self._fe(
            [a.cpu().numpy() for a in audio],
            sampling_rate=self.sample_rate,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
            device=str(audio.device),
        )
        mask = feats["attention_mask"].to(audio.device).bool()
        lens = mask.sum(1)
        # Valid frames are packed as (128, total_frames), as the thinker's get_audio_features does.
        packed = feats["input_features"].to(audio.device).transpose(1, 2)[mask].T
        with torch.no_grad():
            self._model(packed, feature_lens=lens)
        per_clip = self._tokens.split(self._out_lengths(lens, self._model.n_window).tolist())
        return torch.stack([t.mean(0) for t in per_clip])
