"""Static renders of the app's main charts for the README (same engine as the app)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from landgem_methane import model as M, observations as O, uncertainty as U  # noqa: E402

OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, RED, GRAY, GREEN = "#2563eb", "#f59e0b", "#dc2626", "#94a3b8", "#059669"
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 13, "axes.grid": True, "grid.alpha": 0.25})

years, waste = M.waste_series(2000, 2025, 2080, "constant", rate=100_000)
rho = M.methane_density_kg_m3(20)
f = rho / M.HOURS_PER_YEAR                                           # m3/yr -> kg/h
s = U.ModelSettings(k=0.05, L0=170.0)
det = U.deterministic(waste, years, s)
mc = U.run_monte_carlo(waste, years, s, U.MCSettings(n=8000, seed=42))
b = U.percentile_bands(mc["emission"], f)
obs = O.parse_observations(pd.read_csv(ROOT / "data" / "example_observations_synthetic.csv"))

# 1. timeline with Monte Carlo bands and observations -------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5.2))
ax2 = ax.twinx(); ax2.grid(False); ax2.spines["right"].set_visible(True)
ax2.bar(years, waste / 1e3, color=GRAY, alpha=0.22, width=0.8)
ax2.set_ylabel("Waste accepted (kt/yr)"); ax2.set_ylim(0, 400)
ax.fill_between(years, b["p5"], b["p95"], color=BLUE, alpha=0.15, label="P5 to P95")
ax.fill_between(years, b["p25"], b["p75"], color=BLUE, alpha=0.30, label="P25 to P75")
ax.plot(years, b["p50"], color=BLUE, lw=3, label="Monte Carlo median")
ax.plot(years, det["emission"] * f, color=ORANGE, lw=2.4, ls="--", label="Deterministic (k 0.05, L0 170)")
ax.errorbar(obs["year"], obs["kg_h"], yerr=obs["sigma_kg_h"], fmt="o", color=RED, mec="white", ms=8, capsize=3,
            label="Observations (synthetic)")
ax.axvline(2025, color=GRAY, ls="--", lw=1); ax.text(2025.5, ax.get_ylim()[1] * 0.93, "closure", color="#64748b")
ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
ax.set_xlabel("Year"); ax.set_ylabel("Methane (kg CH4/h)"); ax.set_ylim(0)
ax.set_title("Methane emission over time: LandGEM equation, Monte Carlo bands and observations", loc="left")
ax.legend(loc="upper right", frameon=False, ncol=2)
fig.tight_layout(); fig.savefig(OUT / "timeline.png", dpi=150); plt.close(fig)

# 2. k x L0 heat-map (emission in 2026) -----------------------------------------------------------
kg, lg = np.linspace(0.01, 0.15, 57), np.linspace(40, 300, 53)
unitG = M.methane_generation_m3(waste, kg, 1.0)
Z = unitG[:, 26][:, None] * lg[None, :] * f
fig, ax = plt.subplots(figsize=(7.5, 5.2))
im = ax.pcolormesh(kg, lg, Z.T, cmap="YlOrRd", shading="auto"); ax.grid(False)
ax.plot(0.05, 170, "kx", ms=12, mew=3); ax.plot(0.05, 170, "wx", ms=8, mew=1.5)
fig.colorbar(im, label="kg CH4/h")
ax.set_xlabel("k (1/yr)"); ax.set_ylabel("L0 (m3/Mg)")
ax.set_title("Emission in 2026 for every (k, L0)", loc="left")
fig.tight_layout(); fig.savefig(OUT / "heatmap.png", dpi=150); plt.close(fig)

# 3. misfit surface ---------------------------------------------------------------------------------
res = O.misfit_surface(waste, years, obs, np.linspace(0.01, 0.20, 80), np.linspace(40, 300, 105), rho)
fig, ax = plt.subplots(figsize=(7.5, 5.2))
im = ax.pcolormesh(res["k_grid"], res["L0_grid"], np.log10(res["chi"]).T, cmap="viridis_r", shading="auto")
ax.plot(res["k_grid"], res["L0_best_for_k"], "w:", lw=1.5, label="best L0 for each k")
ax.plot(res["best_k"], res["best_L0"], "w*", ms=18, mec="black", label=f"best fit: k {res['best_k']:.3f}, L0 {res['best_L0']:.0f}")
ax.plot(0.05, 170, "kx", ms=11, mew=3); ax.grid(False)
fig.colorbar(im, label="log10 misfit"); ax.legend(frameon=False, loc="upper right", labelcolor="white")
ax.set_xlabel("k (1/yr)"); ax.set_ylabel("L0 (m3/Mg)")
ax.set_title("Which (k, L0) agree with the observations?", loc="left")
fig.tight_layout(); fig.savefig(OUT / "misfit.png", dpi=150); plt.close(fig)

# 4. effect of k ------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5.2))
for kv, col in ((0.02, "#64748b"), (0.04, GREEN), (0.05, ORANGE), (0.08, RED), (0.12, "#7c3aed")):
    g = M.methane_generation_m3(waste, kv, 170.0)[0] * f
    ax.plot(years, g, color=col, lw=3.2 if kv == 0.05 else 2, label=f"k = {kv:g}")
ax.set_xlabel("Year"); ax.set_ylabel("kg CH4/h"); ax.legend(frameon=False)
ax.set_title("Effect of k (L0 = 170)", loc="left")
fig.tight_layout(); fig.savefig(OUT / "k_family.png", dpi=150); plt.close(fig)

# 5. distribution + drivers -------------------------------------------------------------------------
i = 26
em = mc["emission"][:, i] * f
dr = U.rank_drivers(mc["samples"], mc["emission"][:, i])
fig, (a, c) = plt.subplots(1, 2, figsize=(11, 4.4))
a.hist(em, bins=60, color=BLUE, alpha=0.65)
for q, ls in ((5, ":"), (50, "-"), (95, ":")):
    v = np.percentile(em, q); a.axvline(v, color="k", ls=ls); a.text(v, a.get_ylim()[1] * 0.97, f"P{q} {v:,.0f}", ha="center", va="top", fontsize=9)
a.set_xlabel("kg CH4/h"); a.set_ylabel("Iterations"); a.set_title("Distribution in 2026", loc="left")
c.barh(dr.index[::-1], dr.values[::-1], color=BLUE); c.set_xlim(0, 1); c.set_xlabel("Spearman rank correlation")
c.set_title("What drives the uncertainty?", loc="left")
fig.tight_layout(); fig.savefig(OUT / "uncertainty.png", dpi=150); plt.close(fig)
print("figures written:", sorted(p.name for p in OUT.glob("*.png")))
