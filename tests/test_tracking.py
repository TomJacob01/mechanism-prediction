"""Tests for the experiment-tracker abstraction."""

import pytest

from mech_uspto.training.tracking import (
    Tracker,
    _flatten_per_class,
    make_tracker,
)


def test_default_tracker_is_noop():
    t = make_tracker("none")
    # Calling any method should not raise and should not require any backend.
    t.init({"hidden_dim": 64}, run_name="test")
    t.log({"loss": 0.5}, step=1)
    t.finish()
    assert isinstance(t, Tracker)


def test_make_tracker_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown tracker"):
        make_tracker("mlflow")


def test_flatten_per_class_expands_lists():
    flat = _flatten_per_class(
        {
            "epoch": 5,
            "val/loss": 0.5,
            "val/pr_auc_per_class": [0.1, 0.2, 0.3],
            "val/n_preds_per_class": [10, 20, 30],
        }
    )
    assert flat["epoch"] == 5
    assert flat["val/loss"] == 0.5
    assert flat["val/pr_auc/c0"] == 0.1
    assert flat["val/pr_auc/c2"] == 0.3
    assert flat["val/n_preds/c1"] == 20
    # Original list keys should not survive.
    assert "val/pr_auc_per_class" not in flat


def test_wandb_tracker_raises_helpful_error_if_uninstalled():
    """If wandb isn't installed, instantiating WandbTracker should fail clearly.

    Skipped when wandb *is* installed (irrelevant case for this test).
    """
    try:
        import wandb  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="wandb not installed"):
            make_tracker("wandb")
    else:
        pytest.skip("wandb is installed; nothing to test")
