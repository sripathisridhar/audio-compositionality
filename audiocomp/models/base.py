"""Base class for all audio encoders in the benchmark.

To add a new model
------------------
1. Create ``audiocomp/models/my_model.py`` with a class subclassing
   :class:`AudioEncoder`. Set ``sample_rate`` and ``embedding_dim`` as
   class attributes and implement :meth:`embed`. Take any construction
   args (checkpoints, hyperparameters) via ``__init__``.
2. In ``audiocomp/models/__init__.py`` import the class and add it to
   ``_REGISTRY``::

       from .my_model import MyModel
       _REGISTRY["my_model"] = MyModel

3. Create ``configs/my_model.yaml`` with at least ``model.name: my_model``
   plus any keyword args your ``__init__`` expects (e.g. ``checkpoint:``).

The CLI passes everything under ``model:`` (except ``name``) as keyword
arguments to your constructor.

Example::

    import torch
    from audiocomp.models.base import AudioEncoder

    class MyModel(AudioEncoder):
        sample_rate = 16000
        embedding_dim = 512

        def __init__(self, checkpoint: str | None = None) -> None:
            self._model = ...
            if checkpoint:
                self._model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            self._model.eval()

        def embed(self, audio: torch.Tensor) -> torch.Tensor:
            # audio: (B, T) float32 at self.sample_rate
            with torch.no_grad():
                return self._model(audio)  # (B, D)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class AudioEncoder(ABC):
    """Abstract base class for all audio encoders.

    Every encoder must declare two class-level attributes and implement two
    methods.  The dataset loader uses ``sample_rate`` to resample audio
    automatically, so the encoder always receives waveforms at the correct
    rate without any extra plumbing in the training script.

    Class attributes
    ----------------
    sample_rate : int
        The sample rate (Hz) expected by this encoder.  The dataset loader
        resamples audio to this value automatically.
    embedding_dim : int
        Dimensionality of the vectors returned by :meth:`embed`.
    """

    sample_rate: int
    embedding_dim: int

    @abstractmethod
    def embed(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode a batch of audio waveforms.

        All preprocessing (mel spectrogram, padding, normalisation, …) must
        happen *inside* this method so that the training script stays
        model-agnostic.

        Parameters
        ----------
        audio:
            ``(B, T)`` float32 tensor of waveforms already resampled to
            ``self.sample_rate``.

        Returns
        -------
        torch.Tensor
            ``(B, D)`` float32 embedding tensor where ``D == self.embedding_dim``.
        """

    # ------------------------------------------------------------------
    # Convenience helpers (not abstract — subclasses inherit for free)
    # ------------------------------------------------------------------

    def to(self, device: str | torch.device) -> "AudioEncoder":
        """Move all sub-modules to *device* and return self."""
        for attr in vars(self).values():
            if isinstance(attr, torch.nn.Module):
                attr.to(device)
        self._device = device
        return self

    def eval(self) -> "AudioEncoder":
        """Set all sub-modules to eval mode and return self."""
        for attr in vars(self).values():
            if isinstance(attr, torch.nn.Module):
                attr.eval()
        return self

    @property
    def device(self) -> torch.device:
        return torch.device(getattr(self, "_device", "cpu"))

    def parameters(self):
        """Yield parameters from all sub-modules (for compatibility with torch utilities)."""
        for attr in vars(self).values():
            if isinstance(attr, torch.nn.Module):
                yield from attr.parameters()
