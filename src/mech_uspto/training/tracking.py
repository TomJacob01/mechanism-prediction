"""Tracker abstraction so engine.train() doesn't import W&B / TensorBoard directly.

Soft dependencies: if the backend isn't installed, the tracker silently
degrades to a no-op. Default backend is ``"none"``.
"""

from __future__ import annotations

from typing import Any


class Tracker:
    """Base class. No-op implementation — also used as the default."""

    def init(self, config: dict[str, Any], run_name: str | None = None) -> None:  # noqa: ARG002
        return None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:  # noqa: ARG002
        return None

    def finish(self) -> None:
        return None


class WandbTracker(Tracker):
    """Weights & Biases backend. Requires ``pip install wandb``."""

    def __init__(self, project: str = "mech-uspto", entity: str | None = None) -> None:
        try:
            import wandb  # noqa: F401
        except ImportError as e:
            raise ImportError("wandb not installed. Install with: pip install wandb") from e
        self._project = project
        self._entity = entity
        self._run = None

    def init(self, config: dict[str, Any], run_name: str | None = None) -> None:
        import wandb

        # Strip non-serializable fields (torch tensors etc.) before logging config.
        safe_config = {k: _to_jsonable(v) for k, v in config.items()}
        self._run = wandb.init(
            project=self._project,
            entity=self._entity,
            name=run_name,
            config=safe_config,
        )

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        import wandb

        if self._run is None:
            return
        # Flatten list-valued metrics (per-class) into scalar keys for W&B.
        flat = _flatten_per_class(metrics)
        wandb.log(flat, step=step)

    def finish(self) -> None:
        import wandb

        if self._run is not None:
            wandb.finish()
            self._run = None


def make_tracker(backend: str = "none", **kwargs: Any) -> Tracker:
    """Factory. ``backend`` is one of ``{"none", "wandb"}``."""
    backend = (backend or "none").lower()
    if backend == "none":
        return Tracker()
    if backend == "wandb":
        return WandbTracker(**kwargs)
    raise ValueError(f"Unknown tracker backend: {backend!r}")


def _to_jsonable(v: Any) -> Any:
    """Convert torch.Tensor / Path / etc. to JSON-friendly types for config logging."""
    try:
        import torch

        if isinstance(v, torch.Tensor):
            return v.detach().cpu().tolist()
    except ImportError:
        pass
    if hasattr(v, "__fspath__"):
        return str(v)
    return v


def _flatten_per_class(metrics: dict[str, Any]) -> dict[str, Any]:
    """Expand ``{"pr_auc_per_class": [a, b, c]}`` → ``{"pr_auc/c0": a, ...}``."""
    out: dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, list):
            stem = k.replace("_per_class", "")
            for i, val in enumerate(v):
                out[f"{stem}/c{i}"] = val
        else:
            out[k] = v
    return out
