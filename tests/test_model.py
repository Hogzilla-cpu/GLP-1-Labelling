import numpy as np
import pytest

from c13model import (
    Parameters,
    add_gaussian_noise,
    fit_ldh,
    fit_pdh,
    ldh_scan,
    pdh_scan,
    simulate,
    with_ldh_activity,
    with_pdh_activity,
)
from c13model.isotopomers import index_of, multiplets, point, unlabeled
from c13model.model import _CS, _OGDH_FWD, _OGDH_REV, _PDH

NO_DIL = dict(natural_abundance=0.0, V_dil_g=0.0, V_mct_in=0.0, V_gln_dil=0.0)


# --- carbon transitions -----------------------------------------------------

def test_pdh_maps_pyruvate_c2_c3_to_acetyl():
    assert _PDH(point(3, [3]))[index_of([2])] == 1.0  # [3-13C]pyr -> [2-13C]acetyl (methyl)
    assert _PDH(point(3, [1]))[0] == 1.0  # C1 lost as CO2


def test_acetyl_methyl_labels_glutamate_c4():
    assert _CS(unlabeled(4), point(2, [2]))[index_of([4])] == 1.0
    assert _CS(unlabeled(4), point(2, [1, 2]))[index_of([4, 5])] == 1.0


def test_second_turn_c4_label_goes_to_c2_and_c3():
    oaa = 0.5 * (_OGDH_FWD(point(5, [4])) + _OGDH_REV(point(5, [4])))
    akg = _CS(oaa, unlabeled(2))
    assert akg[index_of([2])] == pytest.approx(0.5)
    assert akg[index_of([3])] == pytest.approx(0.5)


def test_multiplets_sum_to_one():
    x = np.random.default_rng(0).dirichlet(np.ones(32))
    assert sum(multiplets(x, 3, 5).values()) == pytest.approx(1.0)


# --- whole model ------------------------------------------------------------

def test_isotopomer_distributions_stay_normalised():
    res = simulate(t_end=60)
    for dist in res.isotopomers.values():
        np.testing.assert_allclose(dist.sum(axis=1), 1.0, atol=1e-6)


def test_total_glutamate_is_pool_weighted():
    p = Parameters()
    res = simulate(p, t_eval=[0, 30])
    n, g = p.pools["Glu_n"], p.pools["Glu_g"]
    expected = (n * res.enrichment("Glu_n", 4) + g * res.enrichment("Glu_g", 4)) / (n + g)
    np.testing.assert_allclose(res.enrichment("Glu", 4), expected)


def test_complete_labeling_without_dilution():
    # V_pc = 0: PC adds unlabeled CO2 carbon (-> Glu C1), which would prevent full labeling
    res = simulate(Parameters(**NO_DIL, V_x_n=5.0, V_x_g=5.0, V_pc=0.0), t_eval=[0, 3000], glucose_fe=lambda t: 1.0)
    full = index_of([1, 2, 3, 4, 5])
    assert res.isotopomers["Glu"][-1, full] == pytest.approx(1.0, abs=1e-3)
    assert res.isotopomers["Gln"][-1, full] == pytest.approx(1.0, abs=1e-3)


def _c4_steady_state(p: Parameters, a_n: float, a_g: float) -> dict[str, float]:
    """Steady-state C4 FE of aKG_n, Glu_n, aKG_g, Glu_g, Gln given acetyl-CoA C2 FEs."""
    Vx, Vxg, Vc, Vpc, Vd = p.V_x_n, p.V_x_g, p.V_cyc, p.V_pc, p.V_gln_dil
    A = np.array([
        [p.V_tca_n + Vx, -Vx, 0, 0, 0],
        [-Vx, Vx + Vc, 0, 0, -Vc],
        [0, 0, p.V_tca_g + Vxg, -Vxg, 0],
        [0, -Vc, -(Vxg + Vpc), Vxg + p.V_gs, 0],
        [0, 0, 0, -p.V_gs, Vc + Vpc + Vd],
    ])
    b = np.array([p.V_tca_n * a_n, 0, p.V_tca_g * a_g, 0, 0])
    return dict(zip(["aKG_n", "Glu_n", "aKG_g", "Glu_g", "Gln"], np.linalg.solve(A, b)))


@pytest.mark.parametrize("gln_dil", [0.0, 0.1])
def test_steady_state_c4_matches_linear_solution(gln_dil):
    # C4 comes only from acetyl-CoA C2, so the steady state is a small linear system.
    p = Parameters(natural_abundance=0.0, V_mct_in=0.0, V_gln_dil=gln_dil)
    F = 0.6
    res = simulate(p, t_eval=[0, 5000], glucose_fe=lambda t: F)
    ss = _c4_steady_state(p, F * p.V_pdh_n / p.V_tca_n, F * p.V_pdh_g / p.V_tca_g)
    for pool in ["Glu_n", "Glu_g", "Gln"]:
        assert res.enrichment(pool, 4)[-1] == pytest.approx(ss[pool], abs=2e-3)


def test_glutamine_dilution_lowers_glutamine_labeling():
    base = simulate(Parameters(), t_eval=[0, 120]).observables()
    dil = simulate(Parameters(V_gln_dil=0.2), t_eval=[0, 120]).observables()
    assert dil["Gln_C4_FE"].iloc[-1] < base["Gln_C4_FE"].iloc[-1] - 0.02


def test_uniformly_labeled_glucose_gives_c4_c5_doublet():
    obs = simulate(t_end=30).observables()
    assert obs["Glu_C4_D45"].iloc[-1] > obs["Glu_C4_S"].iloc[-1]


def test_pyruvate_carboxylase_labels_glial_glu_c2_c3_first_turn():
    p = Parameters(**{**NO_DIL, "V_dil_g": 0.15}, V_pdh_g=0.0, V_pdh_n=0.0, V_dil_n=0.9, V_pc=0.1)
    obs = simulate(p, t_eval=[0, 5]).observables()
    assert obs["Glu_g_C2_FE"].iloc[-1] > 0 and obs["Glu_g_C4_FE"].iloc[-1] == pytest.approx(0, abs=1e-6)


# --- acetate ----------------------------------------------------------------

def _acetate(tracer="2-13C", **kw):
    return Parameters(glucose_tracer=None, acetate_tracer=tracer, V_ac=0.1, natural_abundance=0.0, **kw)


def test_acetate_labels_glutamine_before_glutamate():
    obs = simulate(_acetate(), t_eval=[0, 10]).observables()
    assert obs["Gln_C4_FE"].iloc[-1] > 2 * obs["Glu_C4_FE"].iloc[-1]
    assert obs["Glu_g_C4_FE"].iloc[-1] > obs["Glu_n_C4_FE"].iloc[-1]


def test_acetate_steady_state_c4():
    p = _acetate()
    res = simulate(p, t_eval=[0, 5000], acetate_fe=lambda t: 0.8)
    ss = _c4_steady_state(p, 0.0, 0.8 * p.V_ac / p.V_tca_g)
    for pool in ["Glu_n", "Gln"]:
        assert res.enrichment(pool, 4)[-1] == pytest.approx(ss[pool], abs=2e-3)


def test_acetate_tracer_requires_acetate_flux():
    with pytest.raises(ValueError):
        simulate(Parameters(acetate_tracer="2-13C", V_ac=0.0))


# --- PDH --------------------------------------------------------------------

def test_pdh_compartments_and_modes():
    base = Parameters()
    p = with_pdh_activity(base, 0.5, "compensated", "neuron")
    assert p.V_pdh_n == pytest.approx(0.45) and p.V_tca_n == pytest.approx(base.V_tca_n)
    assert p.V_pdh_g == base.V_pdh_g
    p = with_pdh_activity(base, 0.5, "uncompensated", "astrocyte")
    assert p.V_tca_g == pytest.approx(base.V_tca_g - 0.05) and p.V_pdh_n == base.V_pdh_n
    p = with_pdh_activity(Parameters(V_ac=0.1), 0.5, "compensated", "astrocyte", compensate_with="acetate")
    assert p.V_ac == pytest.approx(0.15)


def test_lower_pdh_lowers_glutamate_and_glutamine_labeling():
    df = pdh_scan([0.5, 1.0], t_eval=[0, 30])
    late = df[df.time_min == 30].set_index("pdh_factor")
    assert late.loc[0.5, "Glu_C4_FE"] < late.loc[1.0, "Glu_C4_FE"]
    assert late.loc[0.5, "Gln_C4_FE"] < late.loc[1.0, "Gln_C4_FE"]


# --- LDH --------------------------------------------------------------------

@pytest.mark.parametrize("factor", [0.2, 1.0, 5.0])
def test_ldh_steady_state_pyruvate_dilution(factor):
    p = with_ldh_activity(Parameters(natural_abundance=0.0, V_mct_in=0.3), factor)
    F = 0.6
    res = simulate(p, t_eval=[0, 5000], glucose_fe=lambda t: F)
    fn, fg = p.V_ldh_n + p.V_lac_net_n, p.V_ldh_g + p.V_lac_net_g
    A = np.array([
        [p.V_pdh_n + fn, 0, -p.V_ldh_n],
        [0, p.V_pdh_g + p.V_pc + fg, -p.V_ldh_g],
        [-fn, -fg, p.V_ldh_n + p.V_ldh_g + p.V_mct_out],
    ])
    b = np.array([p.V_gly_n * F, p.V_gly_g * F, 0])
    pyr_n, pyr_g, lac = np.linalg.solve(A, b)
    assert res.enrichment("Pyr_n", 3)[-1] == pytest.approx(pyr_n, abs=1e-3)
    assert res.enrichment("Pyr_g", 3)[-1] == pytest.approx(pyr_g, abs=1e-3)
    assert res.enrichment("Lac", 3)[-1] == pytest.approx(lac, abs=1e-3)


def test_higher_ldh_speeds_lactate_labeling():
    df = ldh_scan([0.2, 5.0], t_eval=[0, 5])
    early = df[df.time_min == 5].set_index("ldh_factor")
    assert early.loc[5.0, "Lac_C3_FE"] > early.loc[0.2, "Lac_C3_FE"]


def test_ldh_modes():
    base = Parameters()
    p = with_ldh_activity(base, 2, "both", "astrocyte")
    assert p.V_ldh_g == pytest.approx(2 * base.V_ldh_g) and p.V_ldh_n == base.V_ldh_n
    assert with_ldh_activity(base, 2, "exchange").V_lac_net_g == base.V_lac_net_g
    assert with_ldh_activity(base, 2, "net").V_ldh_n == base.V_ldh_n


# --- noise and fitting ------------------------------------------------------

def test_noise_is_reproducible_and_unbiased():
    obs = simulate(t_end=60).observables()
    a = add_gaussian_noise(obs, 0.01, seed=1)
    b = add_gaussian_noise(obs, 0.01, seed=1)
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
    resid = (a["Glu_C4_FE"] - obs["Glu_C4_FE"]).to_numpy()
    assert abs(resid.mean()) < 0.005 and resid.std() == pytest.approx(0.01, rel=0.3)
    np.testing.assert_array_equal(a["Glu_C4_S"].to_numpy(), obs["Glu_C4_S"].to_numpy())


@pytest.mark.parametrize("mode,true", [("compensated", 0.5), ("uncompensated", 0.3)])
def test_fit_recovers_pdh_factor(mode, true):
    cols = ["Glu_C4_FE", "Glu_C3_FE", "Gln_C4_FE", "Gln_C3_FE"]
    t = np.arange(0, 121, 5.0)
    obs = simulate(with_pdh_activity(Parameters(), true, mode), t_eval=t).observables()
    noisy = add_gaussian_noise(obs, 0.02, columns=cols, seed=3).reset_index()
    fit = fit_pdh(noisy, cols, mode=mode)
    assert fit.success
    assert abs(fit.factor - true) < 3 * fit.factor_se + 0.02
    assert fit.residual_sd == pytest.approx(0.02, rel=0.3)


def test_fit_recovers_ldh_factor():
    cols = ["Lac_C3_FE", "Glu_C4_FE"]
    t = np.arange(0, 61, 2.0)
    obs = simulate(with_ldh_activity(Parameters(), 0.3), t_eval=t).observables()
    noisy = add_gaussian_noise(obs, 0.01, columns=cols, seed=5).reset_index()
    fit = fit_ldh(noisy, cols)
    assert fit.success and abs(fit.factor - 0.3) < 3 * fit.factor_se + 0.03
