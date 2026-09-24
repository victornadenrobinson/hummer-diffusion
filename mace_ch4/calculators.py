"""Resolve an ASE calculator for a MACE-OFF model version."""
from __future__ import annotations

import contextlib
import functools

from ase.calculators.calculator import Calculator

_OFF23_SIZES = {"off23-small": "small", "off23-medium": "medium", "off23-large": "large"}


@contextlib.contextmanager
def _full_torch_load():
    """Let torch.load unpickle full MACE checkpoints while a calculator is built.

    torch >= 2.6 defaults to weights_only=True, which rejects MACE model
    files. Patched only for the duration of the block, not process-wide.
    """
    import torch

    original = torch.load
    torch.load = functools.partial(original, weights_only=False)
    try:
        yield
    finally:
        torch.load = original


def get_calculator(
    model: str = "off23-medium",
    device: str = "cpu",
    model_path: str | None = None,
    default_dtype: str = "float64",
) -> Calculator:
    """Resolve a MACE-OFF ASE calculator.

    model: one of "off23-small", "off23-medium", "off23-large", or "off24-medium".
    model_path: explicit local checkpoint path/URL. Required for "off24-medium",
        since that checkpoint is only published as a GitHub Release asset
        (ACEsuit/mace-off, tag v0.2) and is not wired into mace.calculators.mace_off().
        If given, it always takes precedence over `model`.
    """
    from mace.calculators import MACECalculator, mace_off

    if model_path is not None:
        with _full_torch_load():
            return MACECalculator(
                model_paths=model_path, device=device, default_dtype=default_dtype
            )

    if model in _OFF23_SIZES:
        with _full_torch_load():
            return mace_off(
                model=_OFF23_SIZES[model], device=device, default_dtype=default_dtype
            )

    if model == "off24-medium":
        raise ValueError(
            "MACE-OFF24(M) is only distributed as a GitHub Release asset "
            "(ACEsuit/mace-off, tag v0.2) and is not wired into mace_off(). "
            "Download the checkpoint yourself and pass its path via model_path."
        )

    raise ValueError(
        f"Unknown model {model!r}; expected one of {sorted(_OFF23_SIZES)} or 'off24-medium'"
    )
