"""Monte Carlo propagation of parameter uncertainty through the LandGEM equation."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .model import methane_generation_m3, surface_emission

KINDS = ["lognormal", "triangular", "uniform", "normal", "fixed"]


@dataclass
class DistSpec:
    """Distribution of one parameter, defined relative to its base value.

    lognormal  : geometric mean = base, geometric standard deviation = ``gsd``
    triangular : min = base*low_mult, mode = base, max = base*high_mult
    uniform    : between base*low_mult and base*high_mult
    normal     : mean = base, standard deviation = base*cv   (truncated at > 0)
    fixed      : always the base value
    """
    kind: str = "fixed"
    gsd: float = 1.5
    low_mult: float = 0.5
    high_mult: float = 1.5
    cv: float = 0.1

    def ppf(self, u: np.ndarray, base: float, lower: float = 1e-9, upper: Optional[float] = None) -> np.ndarray:
        u = np.clip(u, 1e-9, 1 - 1e-9)
        if self.kind == "fixed" or base == 0:
            x = np.full(u.shape, float(base))
        elif self.kind == "lognormal":
            x = np.exp(np.log(base) + np.log(max(self.gsd, 1.0 + 1e-9)) * stats.norm.ppf(u))
        elif self.kind == "uniform":
            lo, hi = base * self.low_mult, base * self.high_mult
            x = lo + (hi - lo) * u
        elif self.kind == "triangular":
            lo, hi = base * self.low_mult, base * self.high_mult
            lo, hi = min(lo, base), max(hi, base)
            if hi - lo <= 0:
                x = np.full(u.shape, float(base))
            else:
                x = stats.triang.ppf(u, (base - lo) / (hi - lo), loc=lo, scale=hi - lo)
        elif self.kind == "normal":
            x = base * (1.0 + self.cv * stats.norm.ppf(u))
        else:
            raise ValueError(f"unknown distribution '{self.kind}'")
        return np.clip(x, lower, upper) if upper is not None else np.maximum(x, lower)


@dataclass
class ModelSettings:
    k: float = 0.05
    L0: float = 170.0
    mass_scale: float = 1.0
    collection_eff: float = 0.0           # 0..1
    collection_start: Optional[int] = None
    oxidation: float = 0.0                # 0..1


@dataclass
class MCSettings:
    n: int = 5000
    seed: int = 42
    k: DistSpec = field(default_factory=lambda: DistSpec("lognormal", gsd=1.6))
    L0: DistSpec = field(default_factory=lambda: DistSpec("lognormal", gsd=1.25))
    mass_scale: DistSpec = field(default_factory=lambda: DistSpec("normal", cv=0.10))
    collection_eff: DistSpec = field(default_factory=lambda: DistSpec("triangular", low_mult=0.8, high_mult=1.1))
    oxidation: DistSpec = field(default_factory=lambda: DistSpec("triangular", low_mult=0.0, high_mult=2.0))
    corr_k_L0: float = 0.0                # Gaussian-copula correlation between k and L0


def deterministic(waste, years, s: ModelSettings) -> Dict[str, np.ndarray]:
    G = methane_generation_m3(waste, s.k, s.L0, s.mass_scale)
    E = surface_emission(G, years, s.collection_eff, s.collection_start, s.oxidation)
    return {"generation": G[0], "emission": E[0]}


def run_monte_carlo(waste, years, s: ModelSettings, mc: MCSettings) -> Dict[str, object]:
    rng = np.random.default_rng(mc.seed)
    n = int(mc.n)
    rho = float(np.clip(mc.corr_k_L0, -0.99, 0.99))
    z1 = rng.standard_normal(n)
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
    u_k, u_l = stats.norm.cdf(z1), stats.norm.cdf(z2)
    k = mc.k.ppf(u_k, s.k, lower=1e-4, upper=1.0)
    L0 = mc.L0.ppf(u_l, s.L0, lower=1.0)
    ms = mc.mass_scale.ppf(rng.uniform(size=n), s.mass_scale, lower=0.05)
    eta = mc.collection_eff.ppf(rng.uniform(size=n), s.collection_eff, lower=0.0, upper=1.0)
    ox = mc.oxidation.ppf(rng.uniform(size=n), s.oxidation, lower=0.0, upper=1.0)
    G = methane_generation_m3(waste, k, L0, ms)
    E = surface_emission(G, years, eta, s.collection_start, ox)
    samples = pd.DataFrame({"k": k, "L0": L0, "waste scale": ms, "collection efficiency": eta, "oxidation": ox})
    return {"generation": G, "emission": E, "samples": samples}


PCTS = (5, 25, 50, 75, 95)


def percentile_bands(arr: np.ndarray, factor: float = 1.0) -> Dict[str, np.ndarray]:
    q = np.percentile(arr, PCTS, axis=0) * factor
    return {f"p{p}": q[i] for i, p in enumerate(PCTS)}


def rank_drivers(samples: pd.DataFrame, response: np.ndarray) -> pd.Series:
    """Spearman rank correlation of each varying input with the response."""
    y = pd.Series(response).rank()
    out = {}
    for c in samples.columns:
        col = samples[c]
        if col.nunique() > 1:
            out[c] = float(col.rank().corr(y))
    return pd.Series(out).sort_values(key=np.abs, ascending=False)
