"""Observation-based emission estimates (e.g. satellite plume quantification):
parsing, comparison with the model, and calibration (k, L0) by misfit surface."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .model import HOURS_PER_YEAR, methane_generation_m3, surface_emission

# unit -> factor to kg CH4 / h  (case-sensitive on purpose: "Mg" is megagram, "mg" milligram)
TO_KG_H = {
    "kg/h": 1.0, "t/h": 1000.0, "kg/s": 3600.0,
    "kg/yr": 1.0 / HOURS_PER_YEAR, "t/yr": 1000.0 / HOURS_PER_YEAR, "Mg/yr": 1000.0 / HOURS_PER_YEAR,
}


def parse_observations(df: pd.DataFrame, default_unit: str = "kg/h") -> pd.DataFrame:
    """Return columns: year (decimal), kg_h, sigma_kg_h (NaN if unknown), label.

    Needs a time column (`year` or `date`) and an emission column (`emission` or
    `value`). Optional: `unit`, `uncertainty` (1 sigma, same unit) or `low`/`high`
    (a ~90 % interval), `label`.
    """
    d = df.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    if "year" in d.columns:
        year = pd.to_numeric(d["year"], errors="coerce")
    elif "date" in d.columns:
        dt = pd.to_datetime(d["date"], errors="coerce")
        year = dt.dt.year + (dt.dt.dayofyear - 1) / 365.25
    else:
        raise ValueError("Observation file needs a `year` or `date` column.")
    vcol = "emission" if "emission" in d.columns else "value" if "value" in d.columns else None
    if vcol is None:
        raise ValueError("Observation file needs an `emission` (or `value`) column.")
    unit = d["unit"].astype(str).str.strip() if "unit" in d.columns else pd.Series(default_unit, index=d.index)
    bad = sorted(set(unit) - set(TO_KG_H))
    if bad:
        raise ValueError(f"Unsupported unit(s) {bad}. Use one of {list(TO_KG_H)}.")
    f = unit.map(TO_KG_H).astype(float)
    kg_h = pd.to_numeric(d[vcol], errors="coerce") * f
    if "uncertainty" in d.columns:
        sig = pd.to_numeric(d["uncertainty"], errors="coerce") * f
    elif {"low", "high"} <= set(d.columns):
        sig = (pd.to_numeric(d["high"], errors="coerce") - pd.to_numeric(d["low"], errors="coerce")) * f / 3.29
    else:
        sig = pd.Series(np.nan, index=d.index)
    out = pd.DataFrame({"year": year, "kg_h": kg_h, "sigma_kg_h": sig,
                        "label": d["label"] if "label" in d.columns else ""})
    out = out.dropna(subset=["year", "kg_h"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid observation rows found.")
    return out


def compare(obs: pd.DataFrame, years: np.ndarray, p50_kg_h: np.ndarray, p5_kg_h: Optional[np.ndarray] = None,
            p95_kg_h: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Model value for the calendar year of each observation."""
    idx = np.searchsorted(years, np.floor(obs["year"].to_numpy()))
    ok = (idx >= 0) & (idx < years.size) & (years[np.clip(idx, 0, years.size - 1)] == np.floor(obs["year"].to_numpy()))
    res = obs.copy()
    res["model_p50"] = np.where(ok, p50_kg_h[np.clip(idx, 0, years.size - 1)], np.nan)
    if p5_kg_h is not None:
        res["model_p5"] = np.where(ok, p5_kg_h[np.clip(idx, 0, years.size - 1)], np.nan)
        res["model_p95"] = np.where(ok, p95_kg_h[np.clip(idx, 0, years.size - 1)], np.nan)
        res["in_band"] = (res["kg_h"] >= res["model_p5"]) & (res["kg_h"] <= res["model_p95"])
    res["obs_over_model"] = res["kg_h"] / res["model_p50"]
    return res


def metrics(cmp: pd.DataFrame) -> Dict[str, float]:
    c = cmp.dropna(subset=["model_p50"])
    if c.empty:
        return {}
    r = c["kg_h"] - c["model_p50"]
    out = {"n": int(len(c)), "bias_kg_h": float(r.mean()), "rmse_kg_h": float(np.sqrt((r ** 2).mean())),
           "median_obs_over_model": float(c["obs_over_model"].median())}
    if "in_band" in c:
        out["share_in_P5_P95"] = float(c["in_band"].mean())
    return out


def misfit_surface(waste, years, obs: pd.DataFrame, k_grid, L0_grid, rho: float, collection_eff: float = 0.0,
                   collection_start: Optional[int] = None, oxidation: float = 0.0, mass_scale: float = 1.0,
                   default_rel_sigma: float = 0.30) -> Dict[str, object]:
    """Misfit between observations and model emission over a (k, L0) grid.

    misfit = sqrt(mean(((model - obs) / sigma)^2)), sigma = observation uncertainty,
    or ``default_rel_sigma * obs`` when none is given. The model is linear in L0, so
    the best L0 for every k is computed in closed form.
    """
    o = obs.copy()
    o["yr"] = np.floor(o["year"]).astype(int)
    o = o[o["yr"].isin(years)]
    if o.empty:
        raise ValueError("No observation falls inside the modelled period.")
    idx = np.searchsorted(years, o["yr"].to_numpy())
    obs_v = o["kg_h"].to_numpy()
    sig = o["sigma_kg_h"].to_numpy()
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, default_rel_sigma * obs_v)
    w = 1.0 / sig ** 2

    k_grid, L0_grid = np.asarray(k_grid, float), np.asarray(L0_grid, float)
    unit_G = methane_generation_m3(waste, k_grid, 1.0, mass_scale)                    # (nk, ny) for L0 = 1
    unit_E = surface_emission(unit_G, years, collection_eff, collection_start, oxidation)
    m1 = unit_E[:, idx] * rho / HOURS_PER_YEAR                                          # (nk, nobs) kg/h per (m3/Mg)
    pred = m1[:, None, :] * L0_grid[None, :, None]                                      # (nk, nL, nobs)
    chi = np.sqrt(np.mean(((pred - obs_v) / sig) ** 2, axis=2))                         # (nk, nL)

    L0_best_k = (m1 * obs_v * w).sum(1) / np.maximum((m1 ** 2 * w).sum(1), 1e-30)       # closed form
    L0_best_k = np.clip(L0_best_k, L0_grid.min(), L0_grid.max())                        # stay inside the physical range
    chi_best_k = np.sqrt(np.mean(((m1 * L0_best_k[:, None] - obs_v) / sig) ** 2, axis=1))
    j = int(np.argmin(chi_best_k))
    return {"k_grid": k_grid, "L0_grid": L0_grid, "chi": chi, "L0_best_for_k": L0_best_k,
            "chi_best_for_k": chi_best_k, "best_k": float(k_grid[j]), "best_L0": float(L0_best_k[j]),
            "best_chi": float(chi_best_k[j]), "n_obs": int(len(obs_v))}
