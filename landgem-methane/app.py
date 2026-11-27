"""LandGEM Methane Explorer: interactive Streamlit app.

Run:  python -m streamlit run app.py
"""
from __future__ import annotations

import io
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from landgem_methane import charts as C
from landgem_methane import model as M
from landgem_methane import observations as O
from landgem_methane import uncertainty as U

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="LandGEM Methane Explorer", layout="wide", initial_sidebar_state="expanded")

PRESETS = {
    "EPA CAA default (k 0.05, L0 170)": (0.05, 170.0),
    "EPA AP-42 inventory (k 0.04, L0 100)": (0.04, 100.0),
    "CAA arid / dry climate (k 0.02, L0 170)": (0.02, 170.0),
    "AP-42 arid / dry climate (k 0.02, L0 100)": (0.02, 100.0),
    "Custom": None,
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def show(obj):
    """Render a Plotly figure full-width (works across Streamlit versions)."""
    try:
        st.plotly_chart(obj, width="stretch")
    except TypeError:
        st.plotly_chart(obj, use_container_width=True)


def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def fmt(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.3g}"


def on_preset():
    p = PRESETS.get(st.session_state.preset)
    if p:
        st.session_state.k, st.session_state.L0 = p


def apply_fit(k: float, L0: float):
    st.session_state.k = float(np.clip(round(k, 3), 0.005, 0.20))
    st.session_state.L0 = float(np.clip(round(L0), 20, 300))
    st.session_state.preset = "Custom"


def dist_ui(label: str, key: str, default: U.DistSpec) -> U.DistSpec:
    with st.expander(label):
        kind = st.selectbox("Distribution", U.KINDS, index=U.KINDS.index(default.kind), key=f"{key}_kind")
        gsd, low, high, cv = default.gsd, default.low_mult, default.high_mult, default.cv
        if kind == "lognormal":
            gsd = st.slider("Geometric standard deviation", 1.01, 3.0, float(default.gsd), 0.01, key=f"{key}_gsd")
        elif kind in ("triangular", "uniform"):
            low = st.slider("Minimum (x base value)", 0.0, 1.0, float(default.low_mult), 0.05, key=f"{key}_lo")
            high = st.slider("Maximum (x base value)", 1.0, 3.0, float(default.high_mult), 0.05, key=f"{key}_hi")
        elif kind == "normal":
            cv = st.slider("Coefficient of variation (%)", 1, 100, int(default.cv * 100), key=f"{key}_cv") / 100.0
    return U.DistSpec(kind, gsd, low, high, cv)


@st.cache_data(show_spinner=False, max_entries=24)
def mc_cached(waste: np.ndarray, years: np.ndarray, s: U.ModelSettings, mc: U.MCSettings):
    return U.run_monte_carlo(waste, years, s, mc)


@st.cache_data(show_spinner=False)
def load_text(name: str) -> str:
    return (DATA_DIR / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# session defaults
# --------------------------------------------------------------------------
st.session_state.setdefault("preset", "EPA CAA default (k 0.05, L0 170)")
st.session_state.setdefault("k", 0.05)
st.session_state.setdefault("L0", 170.0)

# --------------------------------------------------------------------------
# sidebar: every input
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    with st.expander("1. Landfill and waste", expanded=True):
        lf_name = st.text_input("Landfill name", "Example landfill")
        c1, c2 = st.columns(2)
        open_year = int(c1.number_input("Open year", 1900, 2100, 2000, step=1))
        close_year = int(c2.number_input("Last year of waste", 1900, 2150, 2025, step=1))
        end_year = int(st.number_input("Model until year", 1900, 2300, 2080, step=1))
        mode_label = st.radio("Waste acceptance", ["Constant rate", "Growing rate", "Total mass", "Custom table / CSV"])
        table = None
        rate = growth = total = 0.0
        if mode_label == "Constant rate":
            rate = st.number_input("Waste accepted (Mg/yr)", 0.0, 5e7, 100_000.0, step=5_000.0)
        elif mode_label == "Growing rate":
            rate = st.number_input("Waste in the first year (Mg/yr)", 0.0, 5e7, 60_000.0, step=5_000.0)
            growth = st.slider("Annual growth (%)", -5.0, 15.0, 2.0, 0.5)
        elif mode_label == "Total mass":
            total = st.number_input("Total waste to accept (Mg)", 0.0, 5e9, 2_500_000.0, step=100_000.0)
        else:
            up_w = st.file_uploader("CSV with columns year, mg", type="csv", key="waste_csv")
            if up_w is not None:
                try:
                    wdf = pd.read_csv(up_w)
                    wdf.columns = [c.strip().lower() for c in wdf.columns]
                    table = {int(a): float(b) for a, b in zip(wdf["year"], wdf.iloc[:, 1])}
                except Exception as e:  # noqa: BLE001
                    st.error(f"Could not read the waste CSV: {e}")
            if table is None and close_year >= open_year:
                base = pd.DataFrame({"year": range(open_year, close_year + 1), "Mg": 100_000.0})
                edited = st.data_editor(base, key=f"wt_{open_year}_{close_year}", hide_index=True, num_rows="fixed")
                table = {int(a): float(b) for a, b in zip(edited["year"], edited["Mg"])}
        use_cap = st.checkbox("Cap total at design capacity")
        capacity = st.number_input("Design capacity (Mg)", 1.0, 5e9, 3_000_000.0, step=100_000.0) if use_cap else None

    with st.expander("2. LandGEM parameters", expanded=True):
        st.selectbox("Start from a preset", list(PRESETS), key="preset", on_change=on_preset)
        k = st.slider("k, methane generation rate (1/yr)", 0.005, 0.20, step=0.005, format="%.3f", key="k")
        L0 = st.slider("L0, potential methane capacity (m3/Mg)", 20.0, 300.0, step=1.0, key="L0")
        ch4_frac = st.slider("Methane share of landfill gas (%)", 30, 70, 50) / 100.0

    with st.expander("3. Collection and oxidation (extension)"):
        st.caption("LandGEM estimates generation only. Turn this on to estimate the surface emission "
                   "that a satellite or drone would see.")
        convert = st.checkbox("Convert generation to surface emission")
        eta = start = ox = None
        if convert:
            eta = st.slider("Gas collection efficiency (%)", 0, 100, 75) / 100.0
            start = int(st.number_input("Collection starts in", 1900, 2300, min(open_year + 5, close_year), step=1))
            ox = st.slider("Oxidation in the cover soil (%)", 0, 50, 10) / 100.0

    with st.expander("4. Units"):
        unit = st.selectbox("Show results in", M.UNIT_LABELS, index=1)
        gwp_name = st.selectbox("GWP100 of methane", list(M.GWP100), index=1)
        gas_t = st.slider("Gas reference temperature (C)", 0, 30, 20)

    with st.expander("5. Monte Carlo uncertainty", expanded=True):
        mc_on = st.checkbox("Run Monte Carlo", value=True)
        mc = U.MCSettings()
        if mc_on:
            c1, c2 = st.columns(2)
            n_iter = int(c1.number_input("Iterations", 500, 50_000, 5_000, step=500))
            seed = int(c2.number_input("Random seed", 0, 99_999, 42))
            mc = U.MCSettings(n=n_iter, seed=seed)
            mc.k = dist_ui("k distribution", "mc_k", mc.k)
            mc.L0 = dist_ui("L0 distribution", "mc_L0", mc.L0)
            mc.mass_scale = dist_ui("Waste mass uncertainty", "mc_m", mc.mass_scale)
            if convert:
                mc.collection_eff = dist_ui("Collection efficiency", "mc_eta", mc.collection_eff)
                mc.oxidation = dist_ui("Oxidation", "mc_ox", mc.oxidation)
            mc.corr_k_L0 = st.slider("Correlation between k and L0", -0.9, 0.9, 0.0, 0.05)

    with st.expander("6. Observations"):
        st.caption("Emission estimates from satellites or aircraft (for example Carbon Mapper or Sentinel-5P "
                   "derived rates).")
        up_o = st.file_uploader("CSV: year or date, emission, unit, uncertainty", type="csv", key="obs_csv")
        demo = st.checkbox("Use synthetic demo observations", value=False)

# --------------------------------------------------------------------------
# core calculation
# --------------------------------------------------------------------------
if not (open_year <= close_year <= end_year):
    st.error("Check the years: open year <= last year of waste <= model-until year.")
    st.stop()
if end_year - open_year > 300:
    st.error("Model period is limited to 300 years.")
    st.stop()

years, waste = M.waste_series(open_year, close_year, end_year,
                              {"Constant rate": "constant", "Growing rate": "growth", "Total mass": "total",
                               "Custom table / CSV": "table"}[mode_label],
                              rate=rate, growth_pct=growth, total=total, table=table, capacity=capacity)
if waste.sum() <= 0:
    st.warning("No waste has been entered yet.")
    st.stop()

rho = M.methane_density_kg_m3(gas_t)
gwp = M.GWP100[gwp_name]
fac = M.unit_factor(unit, rho, gwp, ch4_frac)
s = U.ModelSettings(k=k, L0=L0, mass_scale=1.0, collection_eff=eta or 0.0,
                    collection_start=start if convert else None, oxidation=ox or 0.0)
det = U.deterministic(waste, years, s)
mc_res = mc_cached(waste, years, s, mc) if mc_on else None
bands = U.percentile_bands(mc_res["emission"], fac) if mc_res else None
kgh = rho / M.HOURS_PER_YEAR                                   # m3/yr -> kg/h

obs = None
if up_o is not None:
    try:
        obs = O.parse_observations(pd.read_csv(up_o))
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"Observation file: {e}")
elif demo:
    obs = O.parse_observations(pd.read_csv(DATA_DIR / "example_observations_synthetic.csv"))

peak_i = int(np.argmax(det["emission"]))
if "sel_year" not in st.session_state or not (years[0] <= st.session_state.sel_year <= years[-1]):
    st.session_state.sel_year = int(years[peak_i])

# --------------------------------------------------------------------------
# header
# --------------------------------------------------------------------------
st.title("LandGEM Methane Explorer")
st.caption(f"{lf_name}  |  first-order-decay model (EPA LandGEM equation)  |  k = {k:.3f} 1/yr, "
           f"L0 = {L0:.0f} m3/Mg  |  adjust everything in the sidebar")

sel_year = st.slider("Year to inspect", int(years[0]), int(years[-1]), key="sel_year")
i = int(sel_year - years[0])
wip = float(np.cumsum(waste)[i])
cum_gen_mg = float(det["generation"][: i + 1].sum()) * rho / 1000.0

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Peak emission ({int(years[peak_i])})", f"{fmt(det['emission'][peak_i] * fac)}", unit)
if bands:
    c2.metric(f"Emission in {sel_year} (median)", fmt(bands["p50"][i]), unit)
    c2.caption(f"P5 to P95: {fmt(bands['p5'][i])} to {fmt(bands['p95'][i])}")
else:
    c2.metric(f"Emission in {sel_year}", fmt(det["emission"][i] * fac), unit)
c3.metric(f"Waste in place in {sel_year}", f"{wip / 1e6:.2f} Mt")
c4.metric("Methane generated so far", f"{cum_gen_mg / 1e3:,.0f} kt CH4",
          f"{cum_gen_mg * gwp / 1e6:.2f} Mt CO2e", delta_color="off")

tab_t, tab_u, tab_s, tab_o, tab_d, tab_m = st.tabs(
    ["Timeline", "Uncertainty", "Sensitivity", "Observations", "Data and export", "Method"])

# ---- timeline --------------------------------------------------------------
obs_plot = None
if obs is not None:
    obs_plot = obs.copy()
    obs_plot["y_plot"] = obs["kg_h"] / kgh * fac
    obs_plot["sigma_kg_h_u"] = obs["sigma_kg_h"] / kgh * fac if obs["sigma_kg_h"].notna().any() else None

with tab_t:
    markers = {"closure": close_year}
    if convert and start:
        markers["collection starts"] = start
    show(C.timeline(years, unit, det["emission"] * fac, bands,
                    gen=det["generation"] * fac if convert else None, waste=waste, obs=obs_plot,
                    markers=markers, selected_year=sel_year))
    st.caption("Generation rises while waste is accepted, peaks right after closure and then decays by exp(-k) per year. "
               "Shaded bands show Monte Carlo percentiles; drag on the chart to zoom, double-click to reset.")

# ---- uncertainty -----------------------------------------------------------
with tab_u:
    if mc_res is None:
        st.info("Enable Monte Carlo in the sidebar (section 5).")
    else:
        em_i = mc_res["emission"][:, i] * fac
        a, b = st.columns(2)
        with a:
            show(C.distribution(em_i, float(det["emission"][i] * fac), unit, sel_year))
        with b:
            dr = U.rank_drivers(mc_res["samples"], mc_res["emission"][:, i])
            if dr.empty:
                st.info("All parameters are fixed, so there is nothing to rank.")
            else:
                show(C.drivers(dr, sel_year))
        pct = pd.DataFrame({"Percentile": ["P5", "P25", "P50", "P75", "P95", "Mean"],
                            unit: list(np.percentile(em_i, U.PCTS)) + [float(em_i.mean())]})
        show_df(pct.style.format({unit: "{:,.4g}"}), hide_index=True)
        st.caption(f"{mc.n:,} iterations. Rank correlation near +1 means that parameter explains most of the spread.")

# ---- sensitivity -----------------------------------------------------------
with tab_s:
    kg = np.linspace(0.01, 0.15, 57)
    lg = np.linspace(40, 300, 53)
    unitG = M.methane_generation_m3(waste, kg, 1.0)
    unitE = U.surface_emission(unitG, years, s.collection_eff, s.collection_start, s.oxidation)
    Z = unitE[:, i][:, None] * lg[None, :] * fac
    show(C.heatmap(kg, lg, Z, unit, sel_year, k, L0))
    st.caption("Emission is linear in L0 but strongly non-linear in k: the colour changes faster across than up.")
    ks = st.multiselect("Compare values of k", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15],
                        default=[0.02, 0.04, 0.05, 0.08])
    curves = {}
    for kv in sorted(set(ks + [round(k, 3)])):
        g = M.methane_generation_m3(waste, kv, L0)
        e = U.surface_emission(g, years, s.collection_eff, s.collection_start, s.oxidation)[0]
        curves[f"k = {kv:g}"] = e * fac
    show(C.k_family(years, curves, unit, f"k = {round(k, 3):g}"))

# ---- observations ----------------------------------------------------------
with tab_o:
    if obs is None:
        st.info("Upload observation-based emission estimates in the sidebar (section 6), or tick the synthetic demo.")
        st.markdown("**File format** (one row per observation)")
        show_df(pd.DataFrame({"year": [2019.4, 2021.7], "emission": [820, 1130], "unit": ["kg/h", "kg/h"],
                              "uncertainty": [240, 310]}), hide_index=True)
        st.caption("Use `year` or `date`. Units: kg/h, t/h, kg/s, kg/yr, t/yr, Mg/yr. `uncertainty` is optional (1 sigma).")
        st.download_button("Download an example file", load_text("example_observations_synthetic.csv"),
                           "example_observations.csv", "text/csv")
    else:
        p50 = (bands["p50"] if bands else det["emission"] * fac) / fac * kgh
        p5 = bands["p5"] / fac * kgh if bands else None
        p95 = bands["p95"] / fac * kgh if bands else None
        cmp = O.compare(obs, years, p50, p5, p95)
        met = O.metrics(cmp)
        if not met:
            st.warning("None of the observations falls inside the modelled period.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Observations used", met["n"])
            m2.metric("Median obs / model", f"{met['median_obs_over_model']:.2f}")
            m3.metric("RMSE", f"{fmt(met['rmse_kg_h'])} kg/h")
            if "share_in_P5_P95" in met:
                m4.metric("Inside model P5 to P95", f"{100 * met['share_in_P5_P95']:.0f} %")
            show(C.obs_vs_model(cmp))
            res = O.misfit_surface(waste, years, obs, np.linspace(0.01, 0.20, 40), np.linspace(40, 300, 53), rho,
                                   s.collection_eff, s.collection_start, s.oxidation)
            show(C.misfit(res, k, L0))
            b1, b2 = st.columns([2, 1])
            b1.markdown(f"**Best fit on this grid:** k = {res['best_k']:.3f} 1/yr and L0 = {res['best_L0']:.0f} m3/Mg "
                        f"(misfit {res['best_chi']:.2f}, {res['n_obs']} observations). "
                        "Different (k, L0) pairs can fit almost equally well: look for the long valley in the plot, "
                        "not just the star.")
            b2.button("Apply best fit to the sliders", on_click=apply_fit, args=(res["best_k"], res["best_L0"]))
            show_df(cmp.drop(columns=["label"]).round(3), hide_index=True)
        st.caption("Satellite plume rates are snapshots, usually larger-scatter than annual means. Misfit uses the given "
                   "uncertainty, or 30 % of each observation when none is given.")

# ---- data and export -------------------------------------------------------
with tab_d:
    df = pd.DataFrame({"year": years, "waste_accepted_Mg": waste, "waste_in_place_Mg": np.cumsum(waste),
                       f"generation_{unit}": det["generation"] * fac, f"emission_{unit}": det["emission"] * fac})
    if bands:
        for key in ("p5", "p25", "p50", "p75", "p95"):
            df[f"emission_{key}_{unit}"] = bands[key]
    show_df(df.style.format({c: "{:,.4g}" for c in df.columns if c != "year"}), hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button("Download results (CSV)", df.to_csv(index=False), "landgem_results.csv", "text/csv")
    settings = {"landfill": lf_name, "open_year": open_year, "close_year": close_year, "end_year": end_year,
                "waste_mode": mode_label, "k": k, "L0": L0, "methane_fraction": ch4_frac,
                "collection": {"on": convert, "efficiency": eta, "start": start, "oxidation": ox},
                "unit": unit, "gwp": gwp_name, "gas_temperature_c": gas_t, "methane_density_kg_m3": rho,
                "monte_carlo": {"on": mc_on, **(asdict(mc) if mc_on else {})}}
    d2.download_button("Download settings (JSON)", json.dumps(settings, indent=2, default=float),
                       "landgem_settings.json", "application/json")
    show(C.waste_chart(years, waste))

# ---- method ----------------------------------------------------------------
with tab_m:
    st.subheader("Model")
    st.latex(r"Q_{CH_4}=\sum_{i=1}^{n}\sum_{j=0.1}^{1}k\,L_0\,\frac{M_i}{10}\,e^{-k\,t_{ij}}")
    st.markdown("""
`Q` methane generation (m3/yr), `k` generation rate constant (1/yr), `L0` potential methane generation capacity (m3/Mg),
`M_i` waste accepted in year *i* (Mg), `t_ij` age of the *j*-th tenth-of-a-year section of that waste, and
`n` = (year of calculation) - (initial year of waste acceptance), so the open year itself generates nothing.
The app evaluates the double sum in closed form (it is a geometric series), which gives the same numbers as the literal sum.

**EPA default values.** CAA: k = 0.05, L0 = 170. AP-42 inventory: k = 0.04, L0 = 100. Arid areas (below about 25 inches of rain):
k = 0.02. Landfill gas is taken as 50 % methane.

**Extension (not in LandGEM).** Surface emission = generation x (1 - collection) x (1 - oxidation).

**Monte Carlo.** Each iteration draws k, L0, a waste-mass multiplier and, if enabled, collection efficiency and oxidation
from the chosen distributions, and runs the full equation. k and L0 can be correlated through a Gaussian copula.

**Limitations.** One uniform k and L0 for all waste; no waste-composition or climate dependence; no lag phase; the quality of the
result depends on the waste history. Methane density is an ideal-gas value at the chosen temperature. Check your reporting convention.
""")
    st.caption("Reference: U.S. EPA (2005), Landfill Gas Emissions Model (LandGEM) Version 3.02 User's Guide, EPA-600/R-05/047.")
