"""Gaussian measurement noise for simulated NMR data."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def add_gaussian_noise(
    data: pd.DataFrame,
    sd: float,
    columns: Sequence[str] | None = None,
    relative: bool = False,
    seed: int | None = None,
) -> pd.DataFrame:
    """Return a copy of ``data`` with N(0, sd^2) noise added to ``columns``.

    ``relative=True`` scales the noise by each value (sd is then a fraction,
    e.g. 0.05 for 5 %).  With ``relative=False`` sd is in the column's units
    (e.g. umol/g for ``*_conc`` columns, or FE units for ``*_FE``).  Columns
    default to all ``*_FE`` and ``*_conc`` columns.  Pass ``seed`` for
    reproducible noise.
    """
    rng = np.random.default_rng(seed)
    out = data.copy()
    if columns is None:
        columns = [c for c in data.columns if c.endswith(("_FE", "_conc"))]
    for c in columns:
        values = out[c].to_numpy(dtype=float)
        scale = sd * np.abs(values) if relative else sd
        out[c] = values + rng.normal(0.0, 1.0, size=values.shape) * scale
    return out
