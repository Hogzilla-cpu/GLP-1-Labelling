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
    p = Parameters(natural_abundance=0.0, V_x=5.0)
    res = simulate(p, t_eval=[0, 2000], glucose_fe=lambda t: 1.0)
    assert res.isotopomers["Glu"][-1, index_of([1, 2, 3, 4, 5])] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("factor", [0.25, 0.5, 1.0])
def test_steady_state_c4_enrichment_tracks_pdh_fraction(factor):
    # With V_TCA constant, steady-state Glu C4 FE = FE_pyr * V_PDH / V_TCA.
    p = with_pdh_activity(Parameters(natural_abundance=0.0), factor, "compensated")
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
