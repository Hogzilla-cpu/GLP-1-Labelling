"""Metabolic model of cerebral 13C labeling from 13C-glucose and/or 13C-acetate.

Pool structure follows Mason et al. (1992), J Cereb Blood Flow Metab
12:434-447 (glucose -> pyruvate/lactate -> acetyl-CoA -> TCA cycle, with
alpha-ketoglutarate/glutamate and oxaloacetate/aspartate exchange and a
glutamine pool).  Unlike the original [1-13C]glucose analysis, every pool
carries a full isotopomer distribution so multiply-labeled tracers such as
[U-13C]glucose or [1,2-13C2]acetate and the resulting NMR multiplets are
simulated exactly.

Two enzyme activities are explicit inputs:

* **PDH.**  Acetyl-CoA balance gives ``V_TCA = V_PDH + V_ac + V_dil`` where
  V_ac is acetate oxidation and V_dil is unlabeled acetyl-CoA from other
  substrates (fatty acids, ketone bodies...).  See :func:`with_pdh_activity`.
* **LDH.**  Pyruvate and lactate are separate pools linked by LDH:
  Pyr -> Lac at ``V_ldh + V_lac_net`` and Lac -> Pyr at ``V_ldh``, so
  ``V_ldh`` is the exchange flux and ``V_lac_net`` the net lactate production
  (negative = net lactate oxidation).  Lactate also exchanges with blood
  (``V_mct_in`` influx, balancing efflux).  See :func:`with_ldh_activity`.

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
    "Pyr": 3,
    "Lac": 3,
    "AcCoA": 2,
    "OAA": 4,
    "aKG": 5,
    "Glu": 5,
    "Gln": 5,
    "Asp": 4,
}

# --- Carbon transitions -----------------------------------------------------
# Pyruvate/lactate: C1 carboxyl, C2 carbonyl/carbinol, C3 methyl (LDH keeps numbering).
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

# Pyruvate (and lactate) isotopomers produced from each glucose tracer, per
# labeled glucose molecule, averaged over the two trioses.
GLUCOSE_TRACERS = {
    "U-13C": {(1, 2, 3): 1.0},
    "1-13C": {(3,): 0.5, (): 0.5},
    "2-13C": {(2,): 0.5, (): 0.5},
    "6-13C": {(3,): 0.5, (): 0.5},
}
# Acetyl-CoA isotopomers from each acetate tracer (acetate C1 carboxyl -> acetyl
# C1 carbonyl, acetate C2 methyl -> acetyl C2 methyl).
ACETATE_TRACERS = {
    "1-13C": {(1,): 1.0},
    "2-13C": {(2,): 1.0},
    "1,2-13C": {(1, 2): 1.0},
}


@dataclass
class Parameters:
    """Model parameters.

    Defaults: V_PDH = 1.58 with V_ac = V_dil = 0 gives V_TCA = 1.58 umol/g/min,
    the rat brain value reported by Mason et al. (1992).  All other defaults
    are illustrative placeholders; set them from your own references.
    """

    # Acetyl-CoA sources (umol/g/min); V_TCA = V_pdh + V_ac + V_dil
    V_pdh: float = 1.58
    V_ac: float = 0.0  # acetate oxidation (acetyl-CoA synthetase)
    V_dil: float = 0.0  # other unlabeled acetyl-CoA entry

    # Pyruvate / lactate (umol/g/min)
    V_ldh: float = 1.0  # LDH exchange: Lac -> Pyr flux (Pyr -> Lac is V_ldh + V_lac_net)
    V_lac_net: float = 0.05  # net lactate production (negative = net lactate use)
    V_mct_in: float = 0.1  # blood lactate influx (efflux = V_mct_in + V_lac_net)

    # TCA cycle and amino acids (umol/g/min)
    V_pc: float = 0.0  # pyruvate carboxylase (anaplerosis, balanced by aKG efflux)
    V_x: float = 57.0  # aKG <-> Glu exchange; placeholder, verify against Mason 1992
    V_x_asp: float | None = None  # OAA <-> Asp exchange; None -> same as V_x
    V_gln: float = 0.2  # Glu <-> Gln exchange (glutamine synthesis)
    pc_scrambling: float = 0.0  # fraction of PC-derived OAA equilibrated with fumarate

    # Pool sizes (umol/g)
    pools: dict[str, float] = field(
        default_factory=lambda: {
            "Pyr": 0.1,
            "Lac": 1.0,
            "AcCoA": 0.02,
            "OAA": 0.02,
            "aKG": 0.2,
            "Glu": 10.0,
            "Gln": 4.0,
            "Asp": 3.0,
        }
    )

    # Tracers (None = that substrate is unlabeled)
    glucose_tracer: str | None = "U-13C"
    acetate_tracer: str | None = None
    natural_abundance: float = 0.011

    @property
    def V_tca(self) -> float:
        return self.V_pdh + self.V_ac + self.V_dil

    @property
    def V_gly(self) -> float:
        """Glucose-derived pyruvate production (2 x CMRglc)."""
        return self.V_pdh + self.V_pc + self.V_lac_net

    @property
    def V_ldh_fwd(self) -> float:
        """Pyr -> Lac flux."""
        return self.V_ldh + self.V_lac_net

    @property
    def V_mct_out(self) -> float:
        return self.V_mct_in + self.V_lac_net

    def validate(self) -> None:
        if self.glucose_tracer is not None and self.glucose_tracer not in GLUCOSE_TRACERS:
            raise ValueError(f"glucose_tracer must be None or one of {sorted(GLUCOSE_TRACERS)}")
        if self.acetate_tracer is not None and self.acetate_tracer not in ACETATE_TRACERS:
            raise ValueError(f"acetate_tracer must be None or one of {sorted(ACETATE_TRACERS)}")
        if self.acetate_tracer is not None and self.V_ac == 0:
            raise ValueError("acetate_tracer is set but V_ac = 0, so acetate would not be metabolised")
        if self.V_pc > self.V_tca:
            raise ValueError("V_pc cannot exceed V_TCA")
        for name, value in [("V_gly", self.V_gly), ("V_ldh + V_lac_net", self.V_ldh_fwd),
                            ("V_mct_in + V_lac_net", self.V_mct_out)]:
            if value < 0:
                raise ValueError(f"{name} must be non-negative (check V_lac_net)")
        if not 0 <= self.pc_scrambling <= 1:
            raise ValueError("pc_scrambling must be in [0, 1]")
        for name, flux in vars(self).items():
            if name.startswith("V_") and name != "V_lac_net" and flux is not None and flux < 0:
                raise ValueError(f"{name} must be non-negative")


Input = Callable[[float], float]


def exponential_enrichment(fe_max: float = 0.7, tau: float = 1.0) -> Input:
    """Mono-exponential rise of precursor fractional enrichment to ``fe_max``."""
    return lambda t: fe_max * (1.0 - np.exp(-t / tau))


glucose_enrichment = exponential_enrichment


def tabulated_enrichment(times: Sequence[float], values: Sequence[float]) -> Input:
    """Precursor enrichment input from measured values (linear interpolation)."""
    f = interp1d(times, values, bounds_error=False, fill_value=(values[0], values[-1]))
    return lambda t: float(f(t))


def _zero(t: float) -> float:
    return 0.0


def _pattern(tracer: dict, n: int) -> np.ndarray:
    return sum(w * point(n, carbons) for carbons, w in tracer.items())


class _Rhs:
    def __init__(self, p: Parameters, glucose_fe: Input, acetate_fe: Input, lactate_fe: Input):
        p.validate()
        self.p = p
        na = p.natural_abundance
        self.bg = {name: unlabeled(n, na) for name, n in POOLS.items()}
        self.co2 = unlabeled(1, na)
        # (enrichment input, labeled pattern) per substrate; unlabeled tracers use zero input
        glc = GLUCOSE_TRACERS.get(p.glucose_tracer)
        ace = ACETATE_TRACERS.get(p.acetate_tracer)
        self.glc = (glucose_fe if glc else _zero, _pattern(glc, 3) if glc else self.bg["Pyr"])
        self.ace = (acetate_fe if ace else _zero, _pattern(ace, 2) if ace else self.bg["AcCoA"])
        # blood lactate is labeled with the same pattern as glucose-derived lactate
        self.lac = (lactate_fe if glc else _zero, self.glc[1])
        sizes = np.array([2**n for n in POOLS.values()])
        ends = np.cumsum(sizes)
        self.slices = {name: slice(int(e - s), int(e)) for name, s, e in zip(POOLS, sizes, ends)}

    def initial_state(self) -> np.ndarray:
        return np.concatenate([self.bg[name] for name in POOLS])

    def _input(self, source: tuple[Input, np.ndarray], bg: np.ndarray, t: float) -> np.ndarray:
        fe_fn, labeled = source
        fe = fe_fn(t)
        return fe * labeled + (1 - fe) * bg

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        p, bg = self.p, self.bg
        x = {name: y[sl] for name, sl in self.slices.items()}
        V_tca, V_pc, V_x = p.V_tca, p.V_pc, p.V_x
        V_xa = V_x if p.V_x_asp is None else p.V_x_asp
        V_ogdh = V_tca - V_pc

        pc = _PC_FWD(x["Pyr"], self.co2)
        if p.pc_scrambling:
            pc = (1 - p.pc_scrambling) * pc + p.pc_scrambling * 0.5 * (pc + _PC_REV(x["Pyr"], self.co2))

        d = {
            "Pyr": p.V_gly * self._input(self.glc, bg["Pyr"], t) + p.V_ldh * x["Lac"]
            - (p.V_pdh + V_pc + p.V_ldh_fwd) * x["Pyr"],
            "Lac": p.V_ldh_fwd * x["Pyr"] + p.V_mct_in * self._input(self.lac, bg["Lac"], t)
            - (p.V_ldh + p.V_mct_out) * x["Lac"],
            "AcCoA": p.V_pdh * _PDH(x["Pyr"]) + p.V_ac * self._input(self.ace, bg["AcCoA"], t)
            + p.V_dil * bg["AcCoA"] - V_tca * x["AcCoA"],
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

    def observables(self, pools: Sequence[str] = ("Glu", "Gln", "Asp", "Lac")) -> pd.DataFrame:
        """NMR-observable quantities as a wide table indexed by time.

        For every carbon: ``{pool}_C{k}_FE`` (fractional enrichment) and
        ``{pool}_C{k}_conc`` (13C concentration, umol/g).  Multiplet fractions
        of each carbon's 13C signal: ``{pool}_C{k}_S``, ``_D{k-1}{k}``,
        ``_D{k}{k+1}`` and, for interior carbons, ``_Q``.
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
    glucose_fe: Input | None = None,
    acetate_fe: Input | None = None,
    lactate_fe: Input | None = None,
) -> SimulationResult:
    """Integrate the model from an unlabeled (natural abundance) state.

    ``glucose_fe`` / ``acetate_fe`` / ``lactate_fe`` give the fractional
    enrichment of brain glucose, acetate and blood lactate over time (min).
    Defaults: glucose and acetate rise as ``exponential_enrichment()`` when
    their tracer is set; blood lactate stays unlabeled.
    """
    params = params or Parameters()
    t_eval = np.linspace(0, t_end, 241) if t_eval is None else np.asarray(t_eval, dtype=float)
    rhs = _Rhs(
        params,
        glucose_fe or exponential_enrichment(),
        acetate_fe or exponential_enrichment(),
        lactate_fe or _zero,
    )
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


# --- Enzyme activity changes -------------------------------------------------

PDH_MODES = ("compensated", "uncompensated")
LDH_MODES = ("both", "exchange", "net")


def with_pdh_activity(base: Parameters, factor: float, mode: str = "compensated",
                      compensate_with: str = "dil") -> Parameters:
    """Scale PDH flux by ``factor``.

    ``compensated``: V_TCA held constant; the change in PDH flux is taken up
    by ``compensate_with`` (``"dil"``: unlabeled substrates, or ``"acetate"``).
    ``uncompensated``: other acetyl-CoA sources unchanged, so V_TCA follows PDH.
    """
    V_pdh = base.V_pdh * factor
    if mode == "uncompensated":
        return replace(base, V_pdh=V_pdh)
    if mode != "compensated":
        raise ValueError(f"mode must be one of {PDH_MODES}")
    delta = base.V_pdh - V_pdh
    if compensate_with == "dil":
        return replace(base, V_pdh=V_pdh, V_dil=base.V_dil + delta)
    if compensate_with == "acetate":
        return replace(base, V_pdh=V_pdh, V_ac=base.V_ac + delta)
    raise ValueError("compensate_with must be 'dil' or 'acetate'")


def with_ldh_activity(base: Parameters, factor: float, mode: str = "both") -> Parameters:
    """Scale LDH activity by ``factor``.

    ``both``: forward (Pyr->Lac) and reverse (Lac->Pyr) fluxes both scale, as
    for a change in enzyme amount at fixed metabolite levels; net lactate
    production and hence glycolysis scale too.  ``exchange``: only the
    Pyr<->Lac exchange scales, net lactate production fixed.  ``net``: only
    net lactate production scales (glycolysis adjusts), exchange fixed.
    Blood lactate exchange (``V_mct_in``) is unchanged in all modes.
    """
    if mode == "both":
        return replace(base, V_ldh=base.V_ldh * factor, V_lac_net=base.V_lac_net * factor)
    if mode == "exchange":
        return replace(base, V_ldh=base.V_ldh * factor)
    if mode == "net":
        return replace(base, V_lac_net=base.V_lac_net * factor)
    raise ValueError(f"mode must be one of {LDH_MODES}")


def with_activity(base: Parameters, enzyme: str, factor: float, mode: str | None = None, **kwargs) -> Parameters:
    """Dispatch to :func:`with_pdh_activity` or :func:`with_ldh_activity`."""
    if enzyme == "pdh":
        return with_pdh_activity(base, factor, mode or "compensated", **kwargs)
    if enzyme == "ldh":
        return with_ldh_activity(base, factor, mode or "both")
    raise ValueError("enzyme must be 'pdh' or 'ldh'")


def activity_scan(
    factors: Sequence[float],
    enzyme: str,
    base: Parameters | None = None,
    mode: str | None = None,
    compensate_with: str = "dil",
    **simulate_kwargs,
) -> pd.DataFrame:
    """Simulate a series of enzyme activities; returns a long table with an
    ``{enzyme}_factor`` column, the resulting key fluxes, and all
    :meth:`SimulationResult.observables`."""
    base = base or Parameters()
    extra = {"compensate_with": compensate_with} if enzyme == "pdh" else {}
    frames = []
    for f in factors:
        p = with_activity(base, enzyme, f, mode, **extra)
        df = simulate(p, **simulate_kwargs).observables().reset_index()
        meta = {f"{enzyme}_factor": f, "V_pdh": p.V_pdh, "V_tca": p.V_tca, "V_ldh": p.V_ldh, "V_lac_net": p.V_lac_net}
        for i, (k, v) in enumerate(meta.items()):
            df.insert(i, k, v)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def pdh_scan(factors: Sequence[float], base: Parameters | None = None, mode: str = "compensated",
             **kwargs) -> pd.DataFrame:
    return activity_scan(factors, "pdh", base, mode, **kwargs)


def ldh_scan(factors: Sequence[float], base: Parameters | None = None, mode: str = "both",
             **kwargs) -> pd.DataFrame:
    return activity_scan(factors, "ldh", base, mode, **kwargs)
