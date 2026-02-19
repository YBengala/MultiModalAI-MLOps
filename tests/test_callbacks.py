from __future__ import annotations

import torch
import torch.nn as nn

from multimodal_ai.training.callbacks import EarlyStopping


def _make_model() -> nn.Linear:
    """Simple model for testing state save/restore."""
    torch.manual_seed(0)
    return nn.Linear(4, 2)


def test_initialization_defaults():
    """Verifies default parameters."""
    es = EarlyStopping()
    assert es.patience == 7
    assert es.min_delta == 1e-4
    assert es.counter == 0
    assert es.best_loss == float("inf")
    assert es.early_stop is False


def test_initialization_custom():
    """Verifies custom parameters."""
    es = EarlyStopping(patience=10, min_delta=0.01)
    assert es.patience == 10
    assert es.min_delta == 0.01


def test_first_call_sets_best_loss():
    """First call should always save as best."""
    es = EarlyStopping(patience=3)
    model = _make_model()

    es(val_loss=1.0, model=model)

    assert es.best_loss == 1.0
    assert es.counter == 0
    assert es.early_stop is False


def test_improvement_resets_counter():
    """When loss improves, counter should reset."""
    es = EarlyStopping(patience=3, min_delta=0.0)
    model = _make_model()

    es(val_loss=1.0, model=model)
    es(val_loss=0.9, model=model)
    es(val_loss=0.8, model=model)

    assert es.counter == 0
    assert es.best_loss == 0.8
    assert es.early_stop is False
