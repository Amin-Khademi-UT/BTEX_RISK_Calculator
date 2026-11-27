"""First-order-decay methane generation (EPA LandGEM equation).

    Q_CH4(Y) = sum_{i=1..n} sum_{j=0.1..1.0} k * L0 * (M_i / 10) * exp(-k * t_ij)

    Q_CH4  methane generation in calculation year Y            [m3/yr]
    k      methane generation rate constant                   [1/yr]
    L0     potential methane generation capacity               [m3/Mg]
    M_i    mass of waste accepted in year i                    [Mg]
    t_ij   age of the j-th (1/10-year) section of M_i          [yr]
    n      = (year of calculation) - (initial year of waste acceptance)

Convention used here: waste accepted in year W is split into ten sections and in
calculation year Y > W the section ages are t = (Y - W - 1) + j, j = 0.1 ... 1.0.
The open year therefore generates 0, as in LandGEM.

Because the ten sections form a geometric series, the double sum collapses to

    Q(Y) = k * L0 / 10 * f(k) * S(Y),    f(k) = sum_{s=1..10} exp(-k s/10)
    S(Y+1) = S(Y) * exp(-k) + M(Y),      S(open year) = 0

which is exact (not an approximation) and vectorises over Monte Carlo samples.
"""
from __future__ import annotations

from typing import Mapping, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]

R_GAS = 8.314462618          # J/(mol K)
MW_CH4 = 16.043              # g/mol
HOURS_PER_YEAR = 365.0 * 24.0
MINUTES_PER_YEAR = HOURS_PER_YEAR * 60.0
FT3_PER_M3 = 35.3146667

GWP100 = {                    # CH4, 100-year horizon
    "IPCC AR4 (25)": 25.0,
    "IPCC AR5 (28)": 28.0,
    "IPCC AR6 non-fossil (27.0)": 27.0,
}


def section_factor(k: ArrayLike) -> np.ndarray:
    """f(k) = sum_{s=1..10} exp(-k s / 10)  (closed form)."""
    k = np.asarray(k, dtype=float)
    x = k / 10.0
    return np.exp(-x) * (-np.expm1(-k)) / (-np.expm1(-x))


def methane_generation_m3(waste: ArrayLike, k: ArrayLike, L0: ArrayLike,
                          mass_scale: ArrayLike = 1.0) -> np.ndarray:
    """Methane generation (m3 CH4 / yr) for every calculation year.

    Parameters
    ----------
    waste : 1-D array, mass accepted in each calculation year (Mg); the first
        entry is the landfill open year. Years after closure are 0.
    k, L0, mass_scale : scalars or 1-D arrays of length N (Monte Carlo samples).

    Returns an array of shape (N, n_years); N = 1 for scalar inputs.
    """
    waste = np.asarray(waste, dtype=float)
    k = np.atleast_1d(np.asarray(k, dtype=float))
    L0 = np.atleast_1d(np.asarray(L0, dtype=float))
    ms = np.atleast_1d(np.asarray(mass_scale, dtype=float))
    n = max(k.size, L0.size, ms.size)
    k, L0, ms = (np.broadcast_to(a, (n,)) for a in (k, L0, ms))
    if np.any(k <= 0):
        raise ValueError("k must be > 0")
    pref = k * L0 / 10.0 * section_factor(k)
    decay = np.exp(-k)
    out = np.zeros((n, waste.size))
    S = np.zeros(n)
    for y in range(waste.size):
        out[:, y] = pref * S
        S = S * decay + waste[y] * ms
    return out


def reference_double_sum(waste, k: float, L0: float) -> np.ndarray:
    """Literal (slow) evaluation of the LandGEM double sum. Used in tests."""
    waste = np.asarray(waste, dtype=float)
    out = np.zeros(waste.size)
    for y in range(waste.size):                       # calculation year index
        total = 0.0
        for i in range(y):                            # waste years 0 .. y-1  (n = y)
            for j in range(1, 11):
                t = (y - i - 1) + j / 10.0
                total += k * L0 * (waste[i] / 10.0) * np.exp(-k * t)
        out[y] = total
    return out


# --------------------------------------------------------------------------
# Waste acceptance scenarios
# --------------------------------------------------------------------------
def waste_series(open_year: int, close_year: int, end_year: int, mode: str = "constant",
                 rate: float = 100_000.0, growth_pct: float = 0.0, total: float = 0.0,
                 table: Mapping[int, float] | None = None, capacity: float | None = None):
    """Return (years, waste) for open_year ... end_year.

    mode: 'constant' (Mg/yr), 'growth' (start rate + %/yr), 'total' (Mg spread evenly
    over open..close) or 'table' ({year: Mg}). Acceptance runs from open_year to
    close_year inclusive; ``capacity`` (Mg) optionally caps the cumulative total.
    """
    if end_year < close_year or close_year < open_year:
        raise ValueError("Need open_year <= close_year <= end_year")
    years = np.arange(open_year, end_year + 1)
    w = np.zeros(years.size)
    n_acc = close_year - open_year + 1
    if mode == "constant":
        w[:n_acc] = rate
    elif mode == "growth":
        w[:n_acc] = rate * (1.0 + growth_pct / 100.0) ** np.arange(n_acc)
    elif mode == "total":
        w[:n_acc] = total / n_acc
    elif mode == "table":
        for y, m in (table or {}).items():
            if open_year <= int(y) <= end_year:
                w[int(y) - open_year] = float(m)
    else:
        raise ValueError(f"unknown waste mode '{mode}'")
    if capacity is not None and capacity > 0:
        cum = np.cumsum(w)
        over = cum > capacity
        if over.any():
            first = int(np.argmax(over))
            w[first] = max(capacity - (cum[first - 1] if first else 0.0), 0.0)
            w[first + 1:] = 0.0
    return years, w


# --------------------------------------------------------------------------
# Emission adjustments and unit conversion
# --------------------------------------------------------------------------
def methane_density_kg_m3(temp_c: float = 20.0, pressure_kpa: float = 101.325) -> float:
    """Ideal-gas methane density (kg/m3). 0.667 kg/m3 at 20 C and 1 atm."""
    return MW_CH4 / 1000.0 * pressure_kpa * 1000.0 / (R_GAS * (temp_c + 273.15))


def surface_emission(generation: np.ndarray, years: np.ndarray, collection_eff: ArrayLike = 0.0,
                     collection_start: int | None = None, oxidation: ArrayLike = 0.0) -> np.ndarray:
    """Fugitive emission = generation * (1 - collection) * (1 - oxidation).

    Collection acts from ``collection_start`` onward. This step is NOT part of
    LandGEM, which estimates generation only.
    """
    eta = np.atleast_1d(np.asarray(collection_eff, dtype=float))[:, None]
    ox = np.atleast_1d(np.asarray(oxidation, dtype=float))[:, None]
    active = np.ones(years.size) if collection_start is None else (years >= collection_start).astype(float)
    return generation * (1.0 - eta * active[None, :]) * (1.0 - ox)


UNIT_LABELS = ["Mg CH4/yr", "kg CH4/h", "Mg CO2e/yr", "m3 CH4/yr", "m3 LFG/yr", "av ft3/min CH4"]


def unit_factor(unit: str, rho: float, gwp: float, ch4_fraction: float = 0.5) -> float:
    """Multiply m3 CH4/yr by this factor to get the requested unit."""
    if unit == "Mg CH4/yr":
        return rho / 1000.0
    if unit == "kg CH4/h":
        return rho / HOURS_PER_YEAR
    if unit == "Mg CO2e/yr":
        return rho / 1000.0 * gwp
    if unit == "m3 CH4/yr":
        return 1.0
    if unit == "m3 LFG/yr":
        return 1.0 / ch4_fraction
    if unit == "av ft3/min CH4":
        return FT3_PER_M3 / MINUTES_PER_YEAR
    raise ValueError(f"unknown unit '{unit}'")


def to_kg_per_h(m3_per_yr: ArrayLike, rho: float) -> np.ndarray:
    return np.asarray(m3_per_yr) * rho / HOURS_PER_YEAR
