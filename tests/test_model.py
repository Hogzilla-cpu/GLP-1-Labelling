import numpy as np
import pytest

from c13model import Parameters, add_gaussian_noise, pdh_scan, simulate, with_pdh_activity
from c13model.isotopomers import index_of, multiplets, point, unlabeled
from c13model.model import _CS, _OGDH_FWD, _OGDH_REV, _PDH


def test_pdh_maps_pyruvate_c2_c3_to_acetyl():
    assert _PDH(point(3, [3]))[index_of([2])] == 1.0  # [3-13C]pyr -> [2-13C]acetyl (methyl)
    assert _PDH(point(3, [1]))[0] == 1.0  # C1 lost as CO2


def test_acetyl_methyl_labels_glutamate_c4():
    akg = _CS(unlabeled(4), point(2, [2]))
    assert akg[index_of([4])] == 1.0
    akg = _CS(unlabeled(4), point(2, [1, 2]))
    assert akg[index_of([4, 5])] == 1.0


def test_second_turn_c4_label_goes_to_c2_and_c3():
    oaa = 0.5 * (_OGDH_FWD(point(5, [4])) + _OGDH_REV(point(5, [4])))
    akg = _CS(oaa, unlabeled(2))
    assert akg[index_of([2])] == pytest.approx(0.5)
    assert akg[index_of([3])] == pytest.approx(0.5)


def test_multiplets_sum_to_one():
    x = np.random.default_rng(0).dirichlet(np.ones(32))
    m = multiplets(x, 3, 5)
    assert sum(v for v in m.values()) == pytest.approx(1.0)


def test_isotopomer_distributions_stay_normalised():
    res = simulate(t_end=60)
    for dist in res.isotopomers.values():
        np.testing.assert_allclose(dist.sum(axis=1), 1.0, atol=1e-6)


def test_complete_labeling_without_dilution():
    p = Parameters(natural_abundance=0.0, V_x=5.0, V_mct_in=0.0)
    res = simulate(p, t_eval=[0, 2000], glucose_fe=lambda t: 1.0)
    assert res.isotopomers["Glu"][-1, index_of([1, 2, 3, 4, 5])] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("factor", [0.25, 0.5, 1.0])
def test_steady_state_c4_enrichment_tracks_pdh_fraction(factor):
    # With V_TCA constant, steady-state Glu C4 FE = FE_pyr * V_PDH / V_TCA.
    p = with_pdh_activity(Parameters(natural_abundance=0.0, V_mct_in=0.0), factor, "compensated")
    res = simulate(p, t_eval=[0, 3000], glucose_fe=lambda t: 0.6)
    assert res.enrichment("Glu", 4)[-1] == pytest.approx(0.6 * factor, abs=2e-3)
    assert p.V_tca == pytest.approx(1.58)


def test_uncompensated_mode_lowers_tca_rate():
    p = with_pdh_activity(Parameters(), 0.5, "uncompensated")
    assert p.V_tca == pytest.approx(0.79)


def test_lower_pdh_slows_labeling():
    df = pdh_scan([0.5, 1.0], t_eval=[0, 10, 30])
    c4 = df[df.time_min == 30].set_index("pdh_factor")["Glu_C4_FE"]
    assert c4[0.5] < c4[1.0]


def test_uniformly_labeled_glucose_gives_c4_c5_doublet():
    obs = simulate(t_end=30).observables()
    # Early on, [U-13C]acetyl-CoA labels C4 together with C5.
    assert obs["Glu_C4_D45"].iloc[-1] > obs["Glu_C4_S"].iloc[-1]


def test_pyruvate_carboxylase_labels_glu_c2_c3_first_turn():
    p = Parameters(natural_abundance=0.0, V_pc=0.3, V_pdh=0.0, V_dil=1.58)
    obs = simulate(p, t_eval=[0, 5]).observables()
    assert obs["Glu_C2_FE"].iloc[-1] > 0 and obs["Glu_C4_FE"].iloc[-1] == pytest.approx(0, abs=1e-6)


def test_noise_is_reproducible_and_unbiased():
    obs = simulate(t_end=60).observables()
    a = add_gaussian_noise(obs, 0.01, seed=1)
    b = add_gaussian_noise(obs, 0.01, seed=1)
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
    resid = (a["Glu_C4_FE"] - obs["Glu_C4_FE"]).to_numpy()
    assert abs(resid.mean()) < 0.005 and resid.std() == pytest.approx(0.01, rel=0.3)
    # multiplet fractions untouched by default
    np.testing.assert_array_equal(a["Glu_C4_S"].to_numpy(), obs["Glu_C4_S"].to_numpy())


@pytest.mark.parametrize("mode,true", [("compensated", 0.5), ("uncompensated", 0.3)])
def test_fit_recovers_pdh_factor(mode, true):
    from c13model import fit_pdh

    cols = ["Glu_C4_FE", "Glu_C3_FE"]
    t = np.arange(0, 121, 5.0)
    obs = simulate(with_pdh_activity(Parameters(), true, mode), t_eval=t).observables()
    noisy = add_gaussian_noise(obs, 0.02, columns=cols, seed=3).reset_index()
    fit = fit_pdh(noisy, cols, mode=mode)
    assert fit.success
    assert abs(fit.factor - true) < 3 * fit.factor_se + 0.02
    assert fit.residual_sd == pytest.approx(0.02, rel=0.3)


# --- acetate ----------------------------------------------------------------

def _acetate_params(tracer, **kw):
    return Parameters(glucose_tracer=None, acetate_tracer=tracer, V_ac=0.3, natural_abundance=0.0, **kw)


@pytest.mark.parametrize("tracer,carbon", [("2-13C", 4), ("1-13C", 5)])
def test_acetate_labels_expected_glutamate_carbon_first(tracer, carbon):
    obs = simulate(_acetate_params(tracer), t_eval=[0, 3]).observables()
    other = 9 - carbon  # 4 <-> 5
    assert obs[f"Glu_C{carbon}_FE"].iloc[-1] > 10 * obs[f"Glu_C{other}_FE"].iloc[-1]


def test_acetate_steady_state_c4_enrichment():
    p = _acetate_params("2-13C")
    res = simulate(p, t_eval=[0, 3000], acetate_fe=lambda t: 0.8)
    assert res.enrichment("Glu", 4)[-1] == pytest.approx(0.8 * p.V_ac / p.V_tca, abs=2e-3)
    assert res.enrichment("Lac", 3)[-1] == pytest.approx(0.0, abs=1e-9)  # no PC/ME route back


def test_doubly_labeled_acetate_gives_c4_c5_doublet():
    obs = simulate(_acetate_params("1,2-13C"), t_eval=[0, 10]).observables()
    assert obs["Glu_C4_D45"].iloc[-1] > 0.9


def test_acetate_tracer_requires_acetate_flux():
    with pytest.raises(ValueError):
        simulate(Parameters(acetate_tracer="2-13C", V_ac=0.0))


def test_pdh_compensated_by_acetate():
    p = with_pdh_activity(Parameters(V_ac=0.2), 0.5, "compensated", compensate_with="acetate")
    assert p.V_ac == pytest.approx(0.2 + 0.79) and p.V_tca == pytest.approx(1.78)


# --- LDH --------------------------------------------------------------------

@pytest.mark.parametrize("factor", [0.2, 1.0, 5.0])
def test_ldh_steady_state_pyruvate_dilution(factor):
    # Unlabeled blood lactate enters via LDH exchange; analytic steady state.
    from c13model import with_ldh_activity

    p = with_ldh_activity(Parameters(natural_abundance=0.0, V_mct_in=0.3), factor, "both")
    F = 0.6
    res = simulate(p, t_eval=[0, 3000], glucose_fe=lambda t: F)
    lac_frac = p.V_ldh_fwd / (p.V_ldh + p.V_mct_out)
    pyr = p.V_gly * F / (p.V_pdh + p.V_pc + p.V_ldh_fwd - p.V_ldh * lac_frac)
    assert res.enrichment("Glu", 4)[-1] == pytest.approx(pyr * p.V_pdh / p.V_tca, abs=2e-3)


def test_higher_ldh_increases_dilution_and_speeds_lactate_labeling():
    from c13model import ldh_scan

    df = ldh_scan([0.2, 5.0], t_eval=[0, 5, 3000])
    early = df[df.time_min == 5].set_index("ldh_factor")
    late = df[df.time_min == 3000].set_index("ldh_factor")
    assert early.loc[5.0, "Lac_C3_FE"] > early.loc[0.2, "Lac_C3_FE"]
    assert late.loc[5.0, "Glu_C4_FE"] < late.loc[0.2, "Glu_C4_FE"]


def test_ldh_modes():
    from c13model import with_ldh_activity

    base = Parameters(V_ldh=1.0, V_lac_net=0.1)
    assert with_ldh_activity(base, 2, "both").V_ldh_fwd == pytest.approx(2.2)
    assert with_ldh_activity(base, 2, "exchange").V_lac_net == pytest.approx(0.1)
    assert with_ldh_activity(base, 2, "net").V_gly == pytest.approx(base.V_gly + 0.1)


def test_fit_recovers_ldh_factor():
    from c13model import fit_ldh, with_ldh_activity

    cols = ["Lac_C3_FE", "Glu_C4_FE"]
    t = np.arange(0, 61, 2.0)
    obs = simulate(with_ldh_activity(Parameters(), 0.3), t_eval=t).observables()
    noisy = add_gaussian_noise(obs, 0.01, columns=cols, seed=5).reset_index()
    fit = fit_ldh(noisy, cols)
    assert fit.success and abs(fit.factor - 0.3) < 3 * fit.factor_se + 0.03
