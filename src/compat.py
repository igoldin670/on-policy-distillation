"""Explicit generation settings across the two pinned Transformers releases."""

from __future__ import annotations

import inspect


def generation_options(model, use_model_defaults=False):
    """Transformers 4 supports this switch. Transformers 5 removed it."""
    if "use_model_defaults" in inspect.signature(model.generate).parameters:
        return {"use_model_defaults": use_model_defaults}
    return {}
