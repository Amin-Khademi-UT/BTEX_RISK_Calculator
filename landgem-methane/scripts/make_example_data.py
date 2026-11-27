"""Generate SYNTHETIC example files (for demonstration only; not real measurements).

* data/example_observations_synthetic.csv : emission "observations" drawn around a
  hidden truth (k = 0.07, L0 = 130, 100,000 Mg/yr from 2000 to 2025) with 25 % noise.
* data/example_waste.csv                 : an example waste-acceptance history.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from landgem_methane import model as M  # noqa: E402

rng = np.random.default_rng(7)
years, w = M.waste_series(2000, 2025, 2080, "constant", 100_000)
G = M.methane_generation_m3(w, 0.07, 130.0)[0]
rho = M.methane_density_kg_m3(20)
kgh = G * rho / M.HOURS_PER_YEAR

rows = []
for yr in (2009, 2011, 2013, 2015, 2016, 2018, 2019, 2021, 2022, 2024):
    truth = kgh[yr - 2000]
    val = truth * np.exp(rng.normal(0, 0.25))
    date = pd.Timestamp(yr, 1, 1) + pd.Timedelta(days=int(rng.integers(0, 365)))
    rows.append(dict(date=date.strftime("%Y-%m-%d"), emission=round(val, 1), unit="kg/h",
                     uncertainty=round(0.25 * val, 1), label=f"synthetic {yr}"))
pd.DataFrame(rows).to_csv("data/example_observations_synthetic.csv", index=False)

grow = 60_000 * 1.03 ** np.arange(26)
pd.DataFrame({"year": range(2000, 2026), "mg": grow.round(0)}).to_csv("data/example_waste.csv", index=False)
print("written")
