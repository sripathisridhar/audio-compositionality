"""YAML config loader for the benchmark CLIs.

A model config (e.g. ``configs/random.yaml``) inherits defaults from a
sibling ``base.yaml`` via deep merge, so per-model files only need to
declare the keys they override (``model.name``, etc. — anything else is
picked up from base).

Two further conveniences:

* ``${section.key}`` tokens inside string values are replaced with the
  matching dotted path from the fully-merged config (e.g.
  ``${model.name}``). Missing references raise :class:`KeyError`.
* Dot-notation overrides on the command line (e.g. ``train.batch_size=32``)
  are applied last and parsed via :func:`ast.literal_eval`.
"""

from __future__ import annotations

import ast
import os
import re
from typing import Any

import yaml


def load_config(config_path: str, overrides: list[str] | None = None) -> dict:
    """Load *config_path*, deep-merged onto its sibling ``base.yaml``.

    Merge order: ``base.yaml`` → *config_path* → *overrides*.
    """
    overrides = overrides or []
    cfg: dict[str, Any] = {}

    base_path = os.path.join(os.path.dirname(config_path) or ".", "base.yaml")
    if (
        os.path.isfile(base_path)
        and os.path.abspath(base_path) != os.path.abspath(config_path)
    ):
        with open(base_path) as f:
            cfg = yaml.safe_load(f) or {}

    with open(config_path) as f:
        per_model = yaml.safe_load(f) or {}
    cfg = _deep_merge(cfg, per_model)

    cfg = _apply_overrides(cfg, overrides)
    cfg = _interpolate(cfg, cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*; *override* wins on conflicts."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}' — expected key.subkey=value")
        key_path, raw_value = override.split("=", 1)
        try:
            value = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            value = raw_value
        keys = key_path.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return cfg


_TOKEN_RE = re.compile(r"\$\{([^}]+)\}")


def _interpolate(obj, root: dict):
    """Replace ``${section.key}`` placeholders with the dotted-path value
    from the fully-merged *root* config. Missing references raise KeyError.
    """
    if isinstance(obj, dict):
        return {k: _interpolate(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(v, root) for v in obj]
    if isinstance(obj, str):
        def _replace(m: re.Match) -> str:
            token = m.group(1)
            node = root
            for part in token.split("."):
                if not isinstance(node, dict) or part not in node:
                    raise KeyError(
                        f"Config reference '${{{token}}}' could not be "
                        f"resolved against the merged config."
                    )
                node = node[part]
            return str(node)
        return _TOKEN_RE.sub(_replace, obj)
    return obj
