"""Audio encoder registry.

Usage
-----
List available models::

    from audiocomp.models import list_models
    print(list_models())          # ['downsample', 'random']

Instantiate a model by name::

    from audiocomp.models import get_model
    encoder = get_model("random", seed=0)

Add a new model
---------------
1. Create ``audiocomp/models/my_model.py`` with a class subclassing
   :class:`audiocomp.models.base.AudioEncoder`.
2. Import the class here and add it to ``_REGISTRY`` below.
"""

from __future__ import annotations

from .base import AudioEncoder
from .baselines import DownsampleEncoder, RandomEncoder
from .dasheng import DashengEncoder, MiDashengLMAudioEncoder
from .m2d_clap import M2DCLAPEncoder, M2DCLAPProjEncoder
from .qwen3_omni_aut import Qwen3OmniAuTEncoder

# One class may serve several names; the name keys the embedding cache and output dir.
_REGISTRY: dict[str, type[AudioEncoder]] = {
    "random": RandomEncoder,
    "downsample": DownsampleEncoder,
    "dasheng_base": DashengEncoder,
    "dasheng_06b": DashengEncoder,
    "dasheng_12b": DashengEncoder,
    "midashenglm_7b_audio": MiDashengLMAudioEncoder,
    "m2d_clap": M2DCLAPEncoder,
    "m2d_clap_proj": M2DCLAPProjEncoder,
    "qwen3_omni_aut_instruct": Qwen3OmniAuTEncoder,
    "qwen3_omni_aut_captioner": Qwen3OmniAuTEncoder,
    "qwen3_omni_aut_proj": Qwen3OmniAuTEncoder,
}


def get_model(name: str, **kwargs) -> AudioEncoder:
    """Instantiate an encoder by registry name.

    All keyword arguments are forwarded to the encoder's ``__init__``, so
    encoder-specific construction args (e.g. ``seed=42`` for
    ``RandomEncoder``, ``checkpoint=...`` for a user-added encoder) live
    in ``configs/<model>.yaml`` and the CLI just passes the whole
    ``model:`` section through.

    Parameters
    ----------
    name:
        Registry key, e.g. ``"random"``. Use :func:`list_models` to see all.
    **kwargs:
        Forwarded verbatim to the encoder's ``__init__``.

    Raises
    ------
    ValueError
        If *name* is not in the registry.
    TypeError
        If *kwargs* contains a key the encoder's ``__init__`` does not accept.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list_models()}")
    return _REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Return the sorted list of registered model names."""
    return sorted(_REGISTRY)
