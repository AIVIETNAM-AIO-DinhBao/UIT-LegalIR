from __future__ import annotations

from pathlib import Path
from typing import Any


def model_source(model_spec: dict[str, Any]) -> str:
    """Return the local snapshot when configured, otherwise the Hub model ID."""
    local_path = model_spec.get("local_path")
    if local_path:
        path = Path(local_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Configured local model snapshot does not exist: {path}")
        return str(path)
    return model_spec["id"]


def model_load_kwargs(model_spec: dict[str, Any]) -> dict[str, Any]:
    """Arguments shared by Transformers and Sentence Transformers model loaders."""
    kwargs: dict[str, Any] = {}
    if model_spec.get("local_path"):
        kwargs["local_files_only"] = True
    elif model_spec.get("revision"):
        kwargs["revision"] = model_spec["revision"]
    if model_spec.get("local_files_only"):
        kwargs["local_files_only"] = True
    return kwargs