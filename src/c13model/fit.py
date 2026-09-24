"""Least-squares estimation of PDH activity from (noisy) labeling data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .model import Parameters, simulate, with_pdh_activity


@dataclass
class PdhFit:
    factor: float  # fitted PDH activity relative to ``base``
    factor_se: float  # asymptotic standard error
    params: Parameters  # parameters at the fitted PDH activity
    residual_sd: float
    success: bool


def fit_pdh(
    data: pd.DataFrame,
    columns: Sequence[str],
    base: Parameters | None = None,
    mode: str = "compensated",
    glucose_fe: Callable[[float], float] | None = None,
    initial_factor: float = 0.8,
) -> PdhFit:
    """Fit the PDH activity factor to ``data`` (``time_min`` column plus the
    observable ``columns``), all columns jointly with equal weight.

    Only the PDH factor is estimated; every other parameter is held at
    ``base``.  In ``compensated`` mode the factor cannot exceed
    V_TCA / V_PDH of ``base`` (V_dil would become negative).
    """
    base = base or Parameters()
    t = np.sort(data["time_min"].unique())
    obs = data.set_index("time_min").loc[t, list(columns)].to_numpy()
    upper = base.V_tca / base.V_pdh if mode == "compensated" else 5.0

    def residuals(x: np.ndarray) -> np.ndarray:
        p = with_pdh_activity(base, float(x[0]), mode)
        pred = simulate(p, t_eval=t, glucose_fe=glucose_fe).observables()[list(columns)].to_numpy()
        return np.nan_to_num(pred - obs).ravel()

    x0 = min(max(initial_factor, 1e-3), upper)
    sol = least_squares(residuals, [x0], bounds=([1e-3], [upper]), diff_step=1e-3)
    dof = max(sol.fun.size - 1, 1)
    s2 = float(sol.fun @ sol.fun) / dof
    jtj = float(sol.jac[:, 0] @ sol.jac[:, 0])
    se = float(np.sqrt(s2 / jtj)) if jtj > 0 else float("nan")
    factor = float(sol.x[0])
    return PdhFit(factor, se, with_pdh_activity(base, factor, mode), float(np.sqrt(s2)), bool(sol.success))
