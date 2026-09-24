"""Two-compartment (neuron / astrocyte) 13C isotopomer model of brain metabolism.

Structure: the neuronal-astroglial model with the glutamate-glutamine cycle,
as reviewed in Shen J (2013) "Modeling the glutamate-glutamine
neurotransmitter cycle", Front Neuroenergetics 5:1 (doi:10.3389/fnene.2013.00001),
built on the single-compartment pools of Mason et al. (1992) J Cereb Blood
Flow Metab 12:434-447.  Every pool carries a full isotopomer distribution, so
multiply labeled tracers ([U-13C]glucose, [1,2-13C2]acetate) and NMR
multiplets are simulated exactly.

Neuron (suffix _n)                     Astrocyte (suffix _g)
  Pyr_n --PDH_n--> AcCoA_n               Pyr_g --PDH_g--> AcCoA_g <-- acetate (V_ac), V_dil_g
  AcCoA_n + OAA_n -> aKG_n <-Vx_n-> Glu_n  Pyr_g --PC--> OAA_g
  OAA_n <-Vx_n-> Asp_n                   AcCoA_g + OAA_g -> aKG_g <-Vx_g-> Glu_g
                                         Glu_g --GS (V_cyc + V_pc)--> Gln
Glutamate-glutamine cycle: Glu_n --V_cyc--> Glu_g,  Gln --V_cyc--> Glu_n.
Anaplerosis is balanced by glutamine efflux (V_pc).  Glutamine may also
exchange with an unlabeled source (V_gln_dil; glutamine isotopic dilution).
Lactate is one tissue pool exchanging with each compartment's pyruvate
through that compartment's LDH and with blood lactate (MCT).

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
    "Pyr_n": 3, "Pyr_g": 3, "Lac": 3,
    "AcCoA_n": 2, "OAA_n": 4, "aKG_n": 5, "Glu_n": 5, "Asp_n": 4,
    "AcCoA_g": 2, "OAA_g": 4, "aKG_g": 5, "Glu_g": 5, "Gln": 5,
}
# NMR sees total glutamate: the pool-weighted mix of Glu_n and Glu_g.
COMBINED = {"Glu": ("Glu_n", "Glu_g")}
CARBONS = {**POOLS, "Glu": 5}

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
    "1,6-13C": {(3,): 1.0},
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

    Defaults are illustrative values of the magnitude reported for rat brain
    in the two-compartment literature; they are not taken from Shen (2013).
    Set them from your references.
    """

    # Neuron (umol/g/min); V_TCA_n = V_pdh_n + V_dil_n
    V_pdh_n: float = 0.9
    V_dil_n: float = 0.0  # unlabeled neuronal acetyl-CoA entry
    V_x_n: float = 57.0  # aKG_n <-> Glu_n and OAA_n <-> Asp_n exchange (placeholder)
    V_ldh_n: float = 1.0  # neuronal LDH exchange (Lac -> Pyr_n)
    V_lac_net_n: float = 0.0  # net neuronal lactate production (negative = uptake)

    # Astrocyte (umol/g/min); V_TCA_g = V_pdh_g + V_ac + V_dil_g
    V_pdh_g: float = 0.1
    V_ac: float = 0.0  # acetate oxidation (astrocytic)
    V_dil_g: float = 0.05  # unlabeled astroglial acetyl-CoA entry (astroglial dilution)
    V_pc: float = 0.05  # pyruvate carboxylase, balanced by glutamine efflux
    V_x_g: float = 57.0  # aKG_g <-> Glu_g exchange (placeholder)
    V_ldh_g: float = 0.5  # astrocytic LDH exchange (Lac -> Pyr_g)
    V_lac_net_g: float = 0.05  # net astrocytic lactate production
    pc_scrambling: float = 0.0  # fraction of PC-derived OAA equilibrated with fumarate

    # Intercellular (umol/g/min)
    V_cyc: float = 0.25  # glutamate-glutamine cycle
    V_gln_dil: float = 0.0  # glutamine exchange with an unlabeled source
    V_mct_in: float = 0.1  # blood lactate influx (efflux balances)

    # Pool sizes (umol/g)
    pools: dict[str, float] = field(
        default_factory=lambda: {
            "Pyr_n": 0.08, "Pyr_g": 0.02, "Lac": 1.0,
            "AcCoA_n": 0.02, "OAA_n": 0.02, "aKG_n": 0.2, "Glu_n": 9.0, "Asp_n": 3.0,
            "AcCoA_g": 0.005, "OAA_g": 0.005, "aKG_g": 0.05, "Glu_g": 1.0, "Gln": 4.0,
        }
    )

    # Tracers (None = that substrate is unlabeled)
    glucose_tracer: str | None = "U-13C"
    acetate_tracer: str | None = None
    natural_abundance: float = 0.011

    @property
    def V_tca_n(self) -> float:
        return self.V_pdh_n + self.V_dil_n

    @property
    def V_tca_g(self) -> float:
        return self.V_pdh_g + self.V_ac + self.V_dil_g

    @property
    def V_gs(self) -> float:
        """Glutamine synthesis."""
        return self.V_cyc + self.V_pc

    @property
    def V_gly_n(self) -> float:
        return self.V_pdh_n + self.V_lac_net_n

    @property
    def V_gly_g(self) -> float:
        return self.V_pdh_g + self.V_pc + self.V_lac_net_g

    @property
    def V_mct_out(self) -> float:
        return self.V_mct_in + self.V_lac_net_n + self.V_lac_net_g

    def pool_size(self, name: str) -> float:
        if name in COMBINED:
            return sum(self.pools[p] for p in COMBINED[name])
        return self.pools[name]

    def validate(self) -> None:
        if self.glucose_tracer is not None and self.glucose_tracer not in GLUCOSE_TRACERS:
            raise ValueError(f"glucose_tracer must be None or one of {sorted(GLUCOSE_TRACERS)}")
        if self.acetate_tracer is not None and self.acetate_tracer not in ACETATE_TRACERS:
            raise ValueError(f"acetate_tracer must be None or one of {sorted(ACETATE_TRACERS)}")
        if self.acetate_tracer is not None and self.V_ac == 0:
            raise ValueError("acetate_tracer is set but V_ac = 0, so acetate would not be metabolised")
        if self.V_pc > self.V_tca_g:
            raise ValueError("V_pc cannot exceed V_TCA_g")
        checks = [
            ("V_gly_n", self.V_gly_n), ("V_gly_g", self.V_gly_g), ("V_mct_out", self.V_mct_out),
            ("V_ldh_n + V_lac_net_n", self.V_ldh_n + self.V_lac_net_n),
            ("V_ldh_g + V_lac_net_g", self.V_ldh_g + self.V_lac_net_g),
        ]
        for name, value in checks:
            if value < 0:
                raise ValueError(f"{name} must be non-negative (check V_lac_net_*)")
        if not 0 <= self.pc_scrambling <= 1:
            raise ValueError("pc_scrambling must be in [0, 1]")
        for name, flux in vars(self).items():
            if name.startswith("V_") and not name.startswith("V_lac_net") and flux < 0:
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
        self.bg3, self.bg2, self.bg5 = unlabeled(3, na), unlabeled(2, na), unlabeled(5, na)
        self.co2 = unlabeled(1, na)
        glc = GLUCOSE_TRACERS.get(p.glucose_tracer)
        ace = ACETATE_TRACERS.get(p.acetate_tracer)
        # (enrichment input, labeled pattern) per substrate; unlabeled tracers use zero input
        self.glc = (glucose_fe if glc else _zero, _pattern(glc, 3) if glc else self.bg3)
        self.ace = (acetate_fe if ace else _zero, _pattern(ace, 2) if ace else self.bg2)
        # blood lactate is labeled with the same pattern as glucose-derived lactate
        self.lac = (lactate_fe if glc else _zero, self.glc[1])
        self.bg = {name: unlabeled(n, na) for name, n in POOLS.items()}
        sizes = np.array([2**n for n in POOLS.values()])
        ends = np.cumsum(sizes)
        self.slices = {name: slice(int(e - s), int(e)) for name, s, e in zip(POOLS, sizes, ends)}

    def initial_state(self) -> np.ndarray:
        return np.concatenate([self.bg[name] for name in POOLS])

    @staticmethod
    def _input(source: tuple[Input, np.ndarray], bg: np.ndarray, t: float) -> np.ndarray:
        fe_fn, labeled = source
        fe = fe_fn(t)
        return fe * labeled + (1 - fe) * bg

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        p = self.p
        x = {name: y[sl] for name, sl in self.slices.items()}
        glc = self._input(self.glc, self.bg3, t)
        V_tca_n, V_tca_g, V_pc = p.V_tca_n, p.V_tca_g, p.V_pc
        ldh_fwd_n = p.V_ldh_n + p.V_lac_net_n
        ldh_fwd_g = p.V_ldh_g + p.V_lac_net_g

        pc = _PC_FWD(x["Pyr_g"], self.co2)
        if p.pc_scrambling:
            pc = (1 - p.pc_scrambling) * pc + p.pc_scrambling * 0.5 * (pc + _PC_REV(x["Pyr_g"], self.co2))

        def tca_return(akg: np.ndarray) -> np.ndarray:
            return 0.5 * (_OGDH_FWD(akg) + _OGDH_REV(akg))

        d = {
            # glycolysis / LDH
            "Pyr_n": p.V_gly_n * glc + p.V_ldh_n * x["Lac"] - (p.V_pdh_n + ldh_fwd_n) * x["Pyr_n"],
            "Pyr_g": p.V_gly_g * glc + p.V_ldh_g * x["Lac"] - (p.V_pdh_g + V_pc + ldh_fwd_g) * x["Pyr_g"],
            "Lac": ldh_fwd_n * x["Pyr_n"] + ldh_fwd_g * x["Pyr_g"]
            + p.V_mct_in * self._input(self.lac, self.bg3, t)
            - (p.V_ldh_n + p.V_ldh_g + p.V_mct_out) * x["Lac"],
            # neuron
            "AcCoA_n": p.V_pdh_n * _PDH(x["Pyr_n"]) + p.V_dil_n * self.bg2 - V_tca_n * x["AcCoA_n"],
            "OAA_n": V_tca_n * tca_return(x["aKG_n"]) + p.V_x_n * x["Asp_n"] - (V_tca_n + p.V_x_n) * x["OAA_n"],
            "aKG_n": V_tca_n * _CS(x["OAA_n"], x["AcCoA_n"]) + p.V_x_n * x["Glu_n"]
            - (V_tca_n + p.V_x_n) * x["aKG_n"],
            "Glu_n": p.V_x_n * x["aKG_n"] + p.V_cyc * x["Gln"] - (p.V_x_n + p.V_cyc) * x["Glu_n"],
            "Asp_n": p.V_x_n * (x["OAA_n"] - x["Asp_n"]),
            # astrocyte
            "AcCoA_g": p.V_pdh_g * _PDH(x["Pyr_g"]) + p.V_ac * self._input(self.ace, self.bg2, t)
            + p.V_dil_g * self.bg2 - V_tca_g * x["AcCoA_g"],
            "OAA_g": (V_tca_g - V_pc) * tca_return(x["aKG_g"]) + V_pc * pc - V_tca_g * x["OAA_g"],
            # aKG_g leaves via aKG dehydrogenase (V_tca_g - V_pc), net to Glu_g (V_pc) and exchange
            "aKG_g": V_tca_g * _CS(x["OAA_g"], x["AcCoA_g"]) + p.V_x_g * x["Glu_g"]
            - (V_tca_g + p.V_x_g) * x["aKG_g"],
            "Glu_g": (p.V_x_g + V_pc) * x["aKG_g"] + p.V_cyc * x["Glu_n"] - (p.V_x_g + p.V_gs) * x["Glu_g"],
            "Gln": p.V_gs * x["Glu_g"] + p.V_gln_dil * self.bg5
            - (p.V_cyc + V_pc + p.V_gln_dil) * x["Gln"],
        }
        return np.concatenate([d[name] / p.pools[name] for name in POOLS])


@dataclass
class SimulationResult:
    t: np.ndarray
    isotopomers: dict[str, np.ndarray]  # name -> (len(t), 2**n); includes combined "Glu"
    params: Parameters

    def enrichment(self, pool: str, carbon: int) -> np.ndarray:
        return fractional_enrichment(self.isotopomers[pool], carbon)

    def observables(self, pools: Sequence[str] = ("Glu", "Gln", "Glu_n", "Glu_g", "Asp_n", "Lac")) -> pd.DataFrame:
        """NMR-observable quantities as a wide table indexed by time.

        ``Glu`` is total (neuronal + astrocytic) glutamate, as measured by NMR.
        For every carbon: ``{pool}_C{k}_FE`` (fractional enrichment) and
        ``{pool}_C{k}_conc`` (13C concentration, umol/g).  Multiplet fractions
        of each carbon's 13C signal: ``{pool}_C{k}_S``, ``_D{k-1}{k}``,
        ``_D{k}{k+1}`` and, for interior carbons, ``_Q``.
        """
        cols: dict[str, np.ndarray] = {}
        for pool in pools:
            n = CARBONS[pool]
            x = self.isotopomers[pool]
            for k in range(1, n + 1):
                fe = fractional_enrichment(x, k)
                cols[f"{pool}_C{k}_FE"] = fe
                cols[f"{pool}_C{k}_conc"] = fe * self.params.pool_size(pool)
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
    sol = solve_ivp(rhs, (0.0, float(t_eval[-1])), rhs.initial_state(), method="BDF",
                    t_eval=t_eval, rtol=1e-7, atol=1e-10)
    if not sol.success:
        raise RuntimeError(sol.message)
    iso = {name: sol.y[sl].T for name, sl in rhs.slices.items()}
    for name, parts in COMBINED.items():
        sizes = [params.pools[p] for p in parts]
        iso[name] = sum(s * iso[p] for s, p in zip(sizes, parts)) / sum(sizes)
    return SimulationResult(sol.t, iso, params)


# --- Enzyme activity changes -------------------------------------------------

PDH_MODES = ("compensated", "uncompensated")
LDH_MODES = ("both", "exchange", "net")
COMPARTMENTS = ("both", "neuron", "astrocyte")


def _suffixes(compartment: str) -> list[str]:
    try:
        return {"both": ["n", "g"], "neuron": ["n"], "astrocyte": ["g"]}[compartment]
    except KeyError:
        raise ValueError(f"compartment must be one of {COMPARTMENTS}") from None


def with_pdh_activity(base: Parameters, factor: float, mode: str = "compensated",
                      compartment: str = "both", compensate_with: str = "dil") -> Parameters:
    """Scale PDH flux by ``factor`` in ``compartment`` ("both", "neuron", "astrocyte").

    ``compensated``: each compartment's V_TCA held constant; the change in PDH
    flux is taken up by unlabeled substrates (V_dil_n / V_dil_g) or, in
    astrocytes with ``compensate_with="acetate"``, by acetate (V_ac).
    ``uncompensated``: other acetyl-CoA sources unchanged, so V_TCA follows PDH.
    V_cyc and V_pc are unchanged in both modes.
    """
    if mode not in PDH_MODES:
        raise ValueError(f"mode must be one of {PDH_MODES}")
    if compensate_with not in ("dil", "acetate"):
        raise ValueError("compensate_with must be 'dil' or 'acetate'")
    changes: dict[str, float] = {}
    for c in _suffixes(compartment):
        old = getattr(base, f"V_pdh_{c}")
        changes[f"V_pdh_{c}"] = old * factor
        if mode == "compensated":
            sink = "V_ac" if (c == "g" and compensate_with == "acetate") else f"V_dil_{c}"
            changes[sink] = getattr(base, sink) + old * (1 - factor)
    return replace(base, **changes)


def with_ldh_activity(base: Parameters, factor: float, mode: str = "both",
                      compartment: str = "both") -> Parameters:
    """Scale LDH activity by ``factor`` in ``compartment``.

    ``both``: forward (Pyr->Lac) and reverse (Lac->Pyr) fluxes both scale (a
    change in enzyme amount at fixed metabolite levels), so net lactate
    production and glycolysis scale too.  ``exchange``: only the Pyr<->Lac
    exchange scales.  ``net``: only net lactate production scales.  Blood
    lactate exchange (``V_mct_in``) is unchanged.
    """
    if mode not in LDH_MODES:
        raise ValueError(f"mode must be one of {LDH_MODES}")
    changes: dict[str, float] = {}
    for c in _suffixes(compartment):
        if mode in ("both", "exchange"):
            changes[f"V_ldh_{c}"] = getattr(base, f"V_ldh_{c}") * factor
        if mode in ("both", "net"):
            changes[f"V_lac_net_{c}"] = getattr(base, f"V_lac_net_{c}") * factor
    return replace(base, **changes)


def with_activity(base: Parameters, enzyme: str, factor: float, mode: str | None = None,
                  compartment: str = "both", **kwargs) -> Parameters:
    """Dispatch to :func:`with_pdh_activity` or :func:`with_ldh_activity`."""
    if enzyme == "pdh":
        return with_pdh_activity(base, factor, mode or "compensated", compartment, **kwargs)
    if enzyme == "ldh":
        return with_ldh_activity(base, factor, mode or "both", compartment)
    raise ValueError("enzyme must be 'pdh' or 'ldh'")


def activity_scan(
    factors: Sequence[float],
    enzyme: str,
    base: Parameters | None = None,
    mode: str | None = None,
    compartment: str = "both",
    compensate_with: str = "dil",
    **simulate_kwargs,
) -> pd.DataFrame:
    """Simulate a series of enzyme activities; returns a long table with an
    ``{enzyme}_factor`` column, the key fluxes, and all
    :meth:`SimulationResult.observables`."""
    base = base or Parameters()
    extra = {"compensate_with": compensate_with} if enzyme == "pdh" else {}
    frames = []
    for f in factors:
        p = with_activity(base, enzyme, f, mode, compartment, **extra)
        df = simulate(p, **simulate_kwargs).observables().reset_index()
        meta = {f"{enzyme}_factor": f, "V_tca_n": p.V_tca_n, "V_tca_g": p.V_tca_g,
                "V_pdh_n": p.V_pdh_n, "V_pdh_g": p.V_pdh_g, "V_ldh_n": p.V_ldh_n, "V_ldh_g": p.V_ldh_g}
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
