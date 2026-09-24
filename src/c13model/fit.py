"""Least-squares estimation of PDH or LDH activity from (noisy) labeling data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .model import Parameters, _suffixes, simulate, with_activity


@dataclass
class ActivityFit:
    enzyme: str
    factor: float  # fitted activity relative to ``base``
    factor_se: float  # asymptotic standard error
    params: Parameters  # parameters at the fitted activity
    residual_sd: float
    success: bool


PdhFit = ActivityFit


def _upper_bound(base: Parameters, enzyme: str, mode: str | None, compartment: str,
                 compensate_with: str) -> float:
    if enzyme == "pdh" and (mode or "compensated") == "compensated":
        # the compensating flux must stay non-negative
        bounds = []
        for c in _suffixes(compartment):
            sink = "V_ac" if (c == "g" and compensate_with == "acetate") else f"V_dil_{c}"
            pdh = getattr(base, f"V_pdh_{c}")
            bounds.append((pdh + getattr(base, sink)) / pdh)
        return min(bounds)
    return 10.0


def fit_activity(
    data: pd.DataFrame,
    columns: Sequence[str],
    enzyme: str,
    base: Parameters | None = None,
    mode: str | None = None,
    compartment: str = "both",
    compensate_with: str = "dil",
    initial_factor: float = 0.8,
    **simulate_kwargs,
) -> ActivityFit:
    """Fit the ``enzyme`` ("pdh" or "ldh") activity factor to ``data``
    (a ``time_min`` column plus the observable ``columns``), all columns
    jointly with equal weight.

    Only the activity factor is estimated; every other parameter is held at
    ``base``.  ``simulate_kwargs`` (e.g. ``glucose_fe``, ``acetate_fe``) are
    passed to :func:`simulate`.  Note that an activity is only identifiable
    from observables it affects: e.g. with unlabeled blood lactate and no
    lactate data, LDH mainly shifts glutamate labeling through pyruvate
    dilution.  ``compartment`` is "both", "neuron" or "astrocyte".
    """
    base = base or Parameters()
    extra = {"compensate_with": compensate_with} if enzyme == "pdh" else {}
    t = np.sort(data["time_min"].unique())
    obs = data.set_index("time_min").loc[t, list(columns)].to_numpy()
    upper = _upper_bound(base, enzyme, mode, compartment, compensate_with)

    def params_at(f: float) -> Parameters:
        return with_activity(base, enzyme, f, mode, compartment, **extra)

    def residuals(x: np.ndarray) -> np.ndarray:
        pred = simulate(params_at(float(x[0])), t_eval=t, **simulate_kwargs).observables()
        return np.nan_to_num(pred[list(columns)].to_numpy() - obs).ravel()

    x0 = min(max(initial_factor, 1e-3), upper)
    sol = least_squares(residuals, [x0], bounds=([1e-3], [upper]), diff_step=1e-3)
    dof = max(sol.fun.size - 1, 1)
    s2 = float(sol.fun @ sol.fun) / dof
    jtj = float(sol.jac[:, 0] @ sol.jac[:, 0])
    se = float(np.sqrt(s2 / jtj)) if jtj > 0 else float("nan")
    factor = float(sol.x[0])
    return ActivityFit(enzyme, factor, se, params_at(factor), float(np.sqrt(s2)), bool(sol.success))


def fit_pdh(data: pd.DataFrame, columns: Sequence[str], base: Parameters | None = None,
            mode: str = "compensated", **kwargs) -> ActivityFit:
    return fit_activity(data, columns, "pdh", base, mode, **kwargs)


def fit_ldh(data: pd.DataFrame, columns: Sequence[str], base: Parameters | None = None,
            mode: str = "both", **kwargs) -> ActivityFit:
    return fit_activity(data, columns, "ldh", base, mode, **kwargs)
