"""Metabolic model of cerebral 13C labeling from 13C-glucose.

Pool structure follows Mason et al. (1992), J Cereb Blood Flow Metab
12:434-447 (glucose -> pyruvate/lactate -> acetyl-CoA -> TCA cycle, with
alpha-ketoglutarate/glutamate and oxaloacetate/aspartate exchange and a
glutamine pool).  Unlike the original [1-13C]glucose analysis, every pool
carries a full isotopomer distribution so multiply-labeled tracers such as
[U-13C]glucose and the resulting NMR multiplets are simulated exactly.

Pyruvate dehydrogenase (PDH) is an explicit flux.  Acetyl-CoA balance gives

    V_TCA = V_PDH + V_dil

where V_dil is unlabeled acetyl-CoA from other substrates (fatty acids,
ketone bodies, acetate...).  Changing PDH activity therefore changes either
V_TCA (``mode="uncompensated"``) or the glucose fraction of acetyl-CoA at
constant V_TCA (``mode="compensated"``); see :func:`pdh_scan`.

Units: pools in umol/g, fluxes in umol/g/min, time in minutes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from .isotopomers import (
    Transform,
    fractional_enrichment,
    multiplets,
    point,
    unlabeled,
)

# Carbon counts of the tracked pools, in state-vector order.
POOLS = {
    "Pyr": 3,  # pyruvate + lactate (single well-mixed pool)
    "AcCoA": 2,
    "OAA": 4,
    "aKG": 5,
    "Glu": 5,
    "Gln": 5,
    "Asp": 4,
}

# --- Carbon transitions -----------------------------------------------------
# Pyruvate: C1 carboxyl, C2 carbonyl, C3 methyl.
# PDH: acetyl-CoA C1 (carbonyl) <- Pyr C2, C2 (methyl) <- Pyr C3.
_PDH = Transform([3], [(0, 2), (0, 3)])
# Citrate synthase + aconitase + IDH (OAA C1 is lost as CO2):
# aKG C1..C5 <- OAA C4, OAA C3, OAA C2, AcCoA C2, AcCoA C1.
_CS = Transform([4, 2], [(0, 4), (0, 3), (0, 2), (1, 2), (1, 1)])
# aKG dehydrogenase (aKG C1 lost) -> symmetric succinate/fumarate -> OAA.
_OGDH_FWD = Transform([5], [(0, 2), (0, 3), (0, 4), (0, 5)])
_OGDH_REV = Transform([5], [(0, 5), (0, 4), (0, 3), (0, 2)])
# Pyruvate carboxylase: OAA C1..C3 <- Pyr C1..C3, OAA C4 <- CO2.
_PC_FWD = Transform([3, 1], [(0, 1), (0, 2), (0, 3), (1, 1)])
# ... and the same OAA after back-scrambling through fumarate.
_PC_REV = Transform([3, 1], [(1, 1), (0, 3), (0, 2), (0, 1)])

# Pyruvate isotopomers produced from each glucose tracer (per labeled glucose
# molecule, averaged over the two trioses).
TRACERS = {
    "U-13C": {(1, 2, 3): 1.0},
    "1-13C": {(3,): 0.5, (): 0.5},
    "2-13C": {(2,): 0.5, (): 0.5},
    "6-13C": {(3,): 0.5, (): 0.5},
}


@dataclass
class Parameters:
    """Model parameters.

    Defaults: V_PDH = 1.58 with V_dil = 0 gives V_TCA = 1.58 umol/g/min, the
    rat brain value reported by Mason et al. (1992).  All other defaults are
    illustrative placeholders; set them from your own references.
    """

    # Fluxes (umol/g/min)
    V_pdh: float = 1.58
    V_dil: float = 0.0  # unlabeled acetyl-CoA entry
    V_pc: float = 0.0  # pyruvate carboxylase (anaplerosis, balanced by aKG efflux)
    V_x: float = 57.0  # aKG <-> Glu exchange; placeholder, verify against Mason 1992
    V_x_asp: float | None = None  # OAA <-> Asp exchange; None -> same as V_x
    V_gln: float = 0.2  # Glu <-> Gln exchange (glutamine synthesis)
    V_lac_out: float = 0.0  # pyruvate/lactate efflux beyond PDH + PC
    V_lac_dil: float = 0.0  # unlabeled lactate/pyruvate influx
    pc_scrambling: float = 0.0  # fraction of PC-derived OAA equilibrated with fumarate

    # Pool sizes (umol/g)
    pools: dict[str, float] = field(
        default_factory=lambda: {
            "Pyr": 1.0,
            "AcCoA": 0.02,
            "OAA": 0.02,
            "aKG": 0.2,
            "Glu": 10.0,
            "Gln": 4.0,
            "Asp": 3.0,
        }
    )

    # Tracer input
    tracer: str = "U-13C"
    natural_abundance: float = 0.011

    @property
    def V_tca(self) -> float:
        return self.V_pdh + self.V_dil

    @property
    def V_gly(self) -> float:
        """Glucose-derived pyruvate production (2 x CMRglc for oxidised glucose)."""
        return self.V_pdh + self.V_pc + self.V_lac_out - self.V_lac_dil

    def validate(self) -> None:
        if self.tracer not in TRACERS:
            raise ValueError(f"tracer must be one of {sorted(TRACERS)}")
        if self.V_pc > self.V_tca:
            raise ValueError("V_pc cannot exceed V_TCA")
        if self.V_gly < 0:
            raise ValueError("V_lac_dil exceeds total pyruvate consumption")
        if not 0 <= self.pc_scrambling <= 1:
            raise ValueError("pc_scrambling must be in [0, 1]")
        for name, flux in vars(self).items():
            if name.startswith("V_") and flux is not None and flux < 0:
                raise ValueError(f"{name} must be non-negative")


def glucose_enrichment(fe_max: float = 0.7, tau: float = 1.0) -> Callable[[float], float]:
    """Mono-exponential rise of brain glucose fractional enrichment."""
    return lambda t: fe_max * (1.0 - np.exp(-t / tau))


def tabulated_enrichment(times: Sequence[float], values: Sequence[float]) -> Callable[[float], float]:
    """Glucose enrichment input from measured values (linear interpolation)."""
    f = interp1d(times, values, bounds_error=False, fill_value=(values[0], values[-1]))
    return lambda t: float(f(t))


class _Rhs:
    def __init__(self, p: Parameters, fe: Callable[[float], float]):
        p.validate()
        self.p, self.fe = p, fe
        na = p.natural_abundance
        self.bg = {name: unlabeled(n, na) for name, n in POOLS.items()}
        self.co2 = unlabeled(1, na)
        tracer = sum(w * point(3, carbons) for carbons, w in TRACERS[p.tracer].items())
        self.tracer_pyr = tracer
        sizes = np.array([2**n for n in POOLS.values()])
        self.slices = dict(zip(POOLS, (slice(a, b) for a, b in zip(np.r_[0, np.cumsum(sizes)[:-1]], np.cumsum(sizes)))))
        self.n_state = int(sizes.sum())

    def initial_state(self) -> np.ndarray:
        return np.concatenate([self.bg[name] for name in POOLS])

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        p, s = self.p, self.slices
        x = {name: y[sl] for name, sl in s.items()}
        V_tca, V_pc, V_x = p.V_tca, p.V_pc, p.V_x
        V_xa = V_x if p.V_x_asp is None else p.V_x_asp
        V_ogdh = V_tca - V_pc
        fe = self.fe(t)

        glc_pyr = fe * self.tracer_pyr + (1 - fe) * self.bg["Pyr"]
        pc = _PC_FWD(x["Pyr"], self.co2)
        if p.pc_scrambling:
            pc = (1 - p.pc_scrambling) * pc + p.pc_scrambling * 0.5 * (pc + _PC_REV(x["Pyr"], self.co2))

        d = {
            "Pyr": p.V_gly * glc_pyr + p.V_lac_dil * self.bg["Pyr"]
            - (p.V_pdh + V_pc + p.V_lac_out) * x["Pyr"],
            "AcCoA": p.V_pdh * _PDH(x["Pyr"]) + p.V_dil * self.bg["AcCoA"] - V_tca * x["AcCoA"],
            "OAA": V_ogdh * 0.5 * (_OGDH_FWD(x["aKG"]) + _OGDH_REV(x["aKG"]))
            + V_pc * pc + V_xa * x["Asp"] - (V_tca + V_xa) * x["OAA"],
            "aKG": V_tca * _CS(x["OAA"], x["AcCoA"]) + V_x * x["Glu"] - (V_tca + V_x) * x["aKG"],
            "Glu": V_x * x["aKG"] + p.V_gln * x["Gln"] - (V_x + p.V_gln) * x["Glu"],
            "Gln": p.V_gln * (x["Glu"] - x["Gln"]),
            "Asp": V_xa * (x["OAA"] - x["Asp"]),
        }
        return np.concatenate([d[name] / p.pools[name] for name in POOLS])


@dataclass
class SimulationResult:
    t: np.ndarray
    isotopomers: dict[str, np.ndarray]  # name -> (len(t), 2**n)
    params: Parameters

    def enrichment(self, pool: str, carbon: int) -> np.ndarray:
        return fractional_enrichment(self.isotopomers[pool], carbon)

    def observables(self, pools: Sequence[str] = ("Glu", "Gln", "Asp")) -> pd.DataFrame:
        """NMR-observable quantities as a wide table indexed by time.

        For every carbon: ``{pool}_C{k}_FE`` (fractional enrichment) and
        ``{pool}_C{k}_conc`` (13C concentration, umol/g).  For interior carbons
        also multiplet fractions ``{pool}_C{k}_S``, ``_D{k-1}{k}``, ``_D{k}{k+1}``
        and ``_Q``.
        """
        cols: dict[str, np.ndarray] = {}
        for pool in pools:
            n = POOLS[pool]
            x = self.isotopomers[pool]
            for k in range(1, n + 1):
                fe = fractional_enrichment(x, k)
                cols[f"{pool}_C{k}_FE"] = fe
                cols[f"{pool}_C{k}_conc"] = fe * self.params.pools[pool]
                m = multiplets(x, k, n)
                cols[f"{pool}_C{k}_S"] = m["S"]
                if k > 1:
                    cols[f"{pool}_C{k}_D{k - 1}{k}"] = m["D_left"]
                if k < n:
                    cols[f"{pool}_C{k}_D{k}{k + 1}"] = m["D_right"]
                if 1 < k < n:
                    cols[f"{pool}_C{k}_Q"] = m["Q"]
        return pd.DataFrame(cols, index=pd.Index(self.t, name="time_min"))


def simulate(
    params: Parameters | None = None,
    t_end: float = 120.0,
    t_eval: Sequence[float] | None = None,
    glucose_fe: Callable[[float], float] | None = None,
) -> SimulationResult:
    """Integrate the model from an unlabeled (natural abundance) state."""
    params = params or Parameters()
    t_eval = np.linspace(0, t_end, 241) if t_eval is None else np.asarray(t_eval, dtype=float)
    rhs = _Rhs(params, glucose_fe or glucose_enrichment())
    sol = solve_ivp(
        rhs,
        (0.0, float(t_eval[-1])),
        rhs.initial_state(),
        method="BDF",
        t_eval=t_eval,
        rtol=1e-7,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    iso = {name: sol.y[sl].T for name, sl in rhs.slices.items()}
    return SimulationResult(sol.t, iso, params)


def with_pdh_activity(base: Parameters, factor: float, mode: str = "compensated") -> Parameters:
    """Scale PDH flux by ``factor``.

    ``compensated``: V_TCA held constant, lost PDH flux replaced by unlabeled
    acetyl-CoA (V_dil).  ``uncompensated``: V_dil unchanged, so V_TCA falls
    with PDH.
    """
    V_pdh = base.V_pdh * factor
    if mode == "compensated":
        return replace(base, V_pdh=V_pdh, V_dil=base.V_tca - V_pdh)
    if mode == "uncompensated":
        return replace(base, V_pdh=V_pdh)
    raise ValueError("mode must be 'compensated' or 'uncompensated'")


def pdh_scan(
    factors: Sequence[float],
    base: Parameters | None = None,
    mode: str = "compensated",
    **simulate_kwargs,
) -> pd.DataFrame:
    """Simulate a series of PDH activities; returns a long table with a
    ``pdh_factor`` column plus all :meth:`SimulationResult.observables`."""
    base = base or Parameters()
    frames = []
    for f in factors:
        res = simulate(with_pdh_activity(base, f, mode), **simulate_kwargs)
        df = res.observables().reset_index()
        df.insert(0, "pdh_factor", f)
        df.insert(1, "V_pdh", res.params.V_pdh)
        df.insert(2, "V_tca", res.params.V_tca)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
