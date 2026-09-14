from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    return config


def resolve_paths(config: dict[str, Any], root: str | Path = ".") -> dict[str, Any]:
    """Return a copy with pipeline paths resolved against the repository root."""
    result = dict(config)
    paths = dict(result["paths"])
    base = Path(root)
    for key, value in paths.items():
        if key.endswith(("_dir", "_file")):
            paths[key] = str((base / value).resolve())
    result["paths"] = paths
    return result

