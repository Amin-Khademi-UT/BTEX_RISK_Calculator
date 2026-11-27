import numpy as np
import pandas as pd
import pytest

from landgem_methane import model as M
from landgem_methane import observations as O
from landgem_methane import uncertainty as U


def series(rate=100_000.0, open_=2000, close=2024, end=2090):
    return M.waste_series(open_, close, end, "constant", rate=rate)


@pytest.mark.parametrize("k", [0.01, 0.02, 0.05, 0.08, 0.15])
def test_closed_form_equals_landgem_double_sum(k):
    years, w = M.waste_series(2000, 2019, 2060, "growth", rate=50_000, growth_pct=3)
    fast = M.methane_generation_m3(w, k, 170.0)[0]
    slow = M.reference_double_sum(w, k, 170.0)
    assert np.allclose(fast, slow, rtol=1e-12)


def test_hand_calculation_first_generating_year():
    # 1 year of waste: Q(open+1) = k L0 (M/10) * sum_{j=1..10} exp(-k j/10)
    k, L0, Mi = 0.05, 170.0, 100_000.0
    w = np.array([Mi, 0.0, 0.0])
    q1 = M.methane_generation_m3(w, k, L0)[0][1]
    expected = sum(k * L0 * (Mi / 10) * np.exp(-k * j / 10) for j in range(1, 11))
    assert q1 == pytest.approx(expected, rel=1e-12)


def test_open_year_is_zero_and_peak_follows_closure():
    years, w = series()
    G = M.methane_generation_m3(w, 0.05, 170)[0]
    assert G[0] == 0.0                                  # n = Y - open year = 0
    assert years[np.argmax(G)] == 2025                  # peak the year after the last waste


def test_decay_after_closure_is_exp_minus_k_per_year():
    years, w = series()
    k = 0.06
    G = M.methane_generation_m3(w, k, 170)[0]
    after = G[years >= 2026]
    assert np.allclose(after[1:] / after[:-1], np.exp(-k))


def test_mass_balance_matches_analytic_total():
    # Over an infinite horizon the total equals (k L0/10) f(k) M / (1 - e^-k), about L0*M*(1 - k/20)
    k, L0 = 0.05, 170.0
    years, w = M.waste_series(2000, 2024, 2000 + 3000, "constant", rate=100_000)
    total = M.methane_generation_m3(w, k, L0)[0].sum()
    exact = k * L0 / 10 * M.section_factor(k) * w.sum() / (1 - np.exp(-k))
    assert total == pytest.approx(exact, rel=1e-9)
    assert total == pytest.approx(L0 * w.sum(), rel=0.01)       # within 1 % of the potential


def test_linear_in_L0_and_waste():
    years, w = series()
    a = M.methane_generation_m3(w, 0.05, 100)[0]
    assert np.allclose(M.methane_generation_m3(w, 0.05, 200)[0], 2 * a)
    assert np.allclose(M.methane_generation_m3(2 * w, 0.05, 100)[0], 2 * a)
    assert np.allclose(M.methane_generation_m3(w, 0.05, 100, mass_scale=2.0)[0], 2 * a)


def test_vectorised_samples_match_scalar_runs():
    years, w = series()
    ks, Ls = np.array([0.03, 0.05, 0.09]), np.array([100.0, 150.0, 200.0])
    G = M.methane_generation_m3(w, ks, Ls)
    for n in range(3):
        assert np.allclose(G[n], M.methane_generation_m3(w, ks[n], Ls[n])[0])


def test_invalid_k_rejected():
    with pytest.raises(ValueError):
        M.methane_generation_m3(np.ones(5), 0.0, 100)


def test_waste_modes_and_capacity():
    y, w = M.waste_series(2000, 2009, 2020, "total", total=1_000_000)
    assert w.sum() == pytest.approx(1_000_000) and w[10:].sum() == 0
    y, w = M.waste_series(2000, 2009, 2020, "table", table={2000: 10, 2003: 30})
    assert w[0] == 10 and w[3] == 30 and w.sum() == 40
    y, w = M.waste_series(2000, 2009, 2020, "constant", rate=100, capacity=250)
    assert w.sum() == pytest.approx(250) and w[2] == pytest.approx(50)
    with pytest.raises(ValueError):
        M.waste_series(2010, 2005, 2020)


def test_units_and_density():
    rho = M.methane_density_kg_m3(20)
    assert rho == pytest.approx(0.667, abs=0.002)
    m3 = 1_000_000.0
    assert m3 * M.unit_factor("Mg CH4/yr", rho, 28) == pytest.approx(m3 * rho / 1000)
    assert m3 * M.unit_factor("kg CH4/h", rho, 28) == pytest.approx(m3 * rho / 8760)
    assert m3 * M.unit_factor("Mg CO2e/yr", rho, 28) == pytest.approx(m3 * rho / 1000 * 28)
    assert m3 * M.unit_factor("m3 LFG/yr", rho, 28, 0.5) == pytest.approx(2 * m3)


def test_collection_and_oxidation():
    years, w = series()
    G = M.methane_generation_m3(w, 0.05, 170)
    E = M.surface_emission(G, years, 0.8, 2010, 0.1)[0]
    assert np.allclose(E[years < 2010], G[0][years < 2010] * 0.9)
    assert np.allclose(E[years >= 2010], G[0][years >= 2010] * 0.2 * 0.9)


# ---- Monte Carlo ----------------------------------------------------------
def test_fixed_distributions_collapse_to_deterministic():
    years, w = series()
    s = U.ModelSettings()
    mc = U.MCSettings(n=50, k=U.DistSpec("fixed"), L0=U.DistSpec("fixed"), mass_scale=U.DistSpec("fixed"),
                      collection_eff=U.DistSpec("fixed"), oxidation=U.DistSpec("fixed"))
    r = U.run_monte_carlo(w, years, s, mc)
    det = U.deterministic(w, years, s)
    assert np.allclose(r["generation"], det["generation"])


def test_monte_carlo_reproducible_and_ordered():
    years, w = series()
    s = U.ModelSettings()
    a = U.run_monte_carlo(w, years, s, U.MCSettings(n=800, seed=3))
    b = U.run_monte_carlo(w, years, s, U.MCSettings(n=800, seed=3))
    c = U.run_monte_carlo(w, years, s, U.MCSettings(n=800, seed=4))
    assert np.array_equal(a["emission"], b["emission"]) and not np.array_equal(a["emission"], c["emission"])
    bands = U.percentile_bands(a["emission"])
    assert np.all(bands["p5"] <= bands["p50"]) and np.all(bands["p50"] <= bands["p95"])


def test_k_L0_correlation_is_applied():
    years, w = series()
    mc = U.MCSettings(n=6000, seed=1, corr_k_L0=0.8)
    r = U.run_monte_carlo(w, years, U.ModelSettings(), mc)["samples"]
    assert np.corrcoef(np.log(r["k"]), np.log(r["L0"]))[0, 1] == pytest.approx(0.8, abs=0.05)


def test_lognormal_geometric_mean_and_gsd():
    d = U.DistSpec("lognormal", gsd=1.6)
    x = d.ppf(np.random.default_rng(0).uniform(size=40_000), 0.05)
    assert np.exp(np.log(x).mean()) == pytest.approx(0.05, rel=0.02)
    assert np.exp(np.log(x).std()) == pytest.approx(1.6, rel=0.02)


def test_rank_drivers_identifies_main_inputs():
    years, w = series()
    r = U.run_monte_carlo(w, years, U.ModelSettings(), U.MCSettings(n=3000))
    i = list(years).index(2025)
    d = U.rank_drivers(r["samples"], r["emission"][:, i])
    assert set(d.index[:2]) == {"k", "L0"}


# ---- observations ---------------------------------------------------------
def test_parse_units_dates_and_uncertainty():
    df = pd.DataFrame({"date": ["2020-07-01", "2021-01-01"], "emission": [1.0, 8760.0],
                       "unit": ["t/h", "Mg/yr"], "uncertainty": [0.1, 876.0]})
    o = O.parse_observations(df)
    assert o["kg_h"].tolist() == pytest.approx([1000.0, 1000.0])
    assert o["sigma_kg_h"].tolist() == pytest.approx([100.0, 100.0])
    assert 2020.4 < o["year"][0] < 2020.6


def test_ambiguous_or_missing_input_rejected():
    with pytest.raises(ValueError):
        O.parse_observations(pd.DataFrame({"year": [2020], "emission": [1], "unit": ["mg/yr"]}))
    with pytest.raises(ValueError):
        O.parse_observations(pd.DataFrame({"emission": [1]}))


def test_fit_recovers_true_parameters_from_noise_free_observations():
    years, w = M.waste_series(2000, 2025, 2080, "constant", rate=100_000)
    rho = M.methane_density_kg_m3(20)
    truth = M.methane_generation_m3(w, 0.07, 130.0)[0] * rho / M.HOURS_PER_YEAR
    yrs = [2008, 2012, 2016, 2020, 2024, 2030, 2040]
    obs = pd.DataFrame({"year": np.array(yrs) + 0.5, "kg_h": [truth[y - 2000] for y in yrs], "sigma_kg_h": np.nan})
    res = O.misfit_surface(w, years, obs, np.linspace(0.01, 0.20, 191), np.linspace(40, 300, 261), rho)
    assert res["best_k"] == pytest.approx(0.07, abs=0.002)
    assert res["best_L0"] == pytest.approx(130.0, abs=2.0)
    assert res["best_chi"] < 0.05


def test_compare_and_metrics():
    years = np.arange(2000, 2010)
    p50 = np.arange(10, 20, dtype=float)
    obs = pd.DataFrame({"year": [2003.2, 2008.9], "kg_h": [14.0, 20.0], "sigma_kg_h": [np.nan, np.nan], "label": ["", ""]})
    c = O.compare(obs, years, p50, p50 * 0.8, p50 * 1.2)
    assert c["model_p50"].tolist() == [13.0, 18.0]
    m = O.metrics(c)
    assert m["n"] == 2 and 0 <= m["share_in_P5_P95"] <= 1
