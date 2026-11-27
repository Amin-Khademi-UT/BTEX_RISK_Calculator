"""Plotly figures for the Streamlit app (interactive: hover, zoom, pan)."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE, ORANGE, RED, GREEN, PURPLE, GRAY = "#3b82f6", "#f59e0b", "#ef4444", "#10b981", "#8b5cf6", "#94a3b8"


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def _layout(fig: go.Figure, title: str = "", height: int = 460, ytitle: str = "", xtitle: str = ""):
    fig.update_layout(title=dict(text=title, x=0.0, font=dict(size=16)), height=height,
                      margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0))
    if ytitle:
        fig.update_yaxes(title_text=ytitle)
    if xtitle:
        fig.update_xaxes(title_text=xtitle)
    return fig


def timeline(years, unit: str, det_em, bands: Optional[Dict[str, np.ndarray]], gen=None, waste=None,
             obs: Optional[pd.DataFrame] = None, markers: Optional[Dict[str, int]] = None,
             selected_year: Optional[int] = None) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if waste is not None:
        fig.add_trace(go.Bar(x=years, y=waste, name="Waste accepted (Mg/yr)", marker_color=_rgba(GRAY, 0.25),
                             hovertemplate="%{y:,.0f} Mg<extra>waste</extra>"), secondary_y=True)
    if bands is not None:
        fig.add_trace(go.Scatter(x=years, y=bands["p95"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=years, y=bands["p5"], fill="tonexty", fillcolor=_rgba(BLUE, 0.15),
                                 line=dict(width=0), name="P5 to P95", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=years, y=bands["p75"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=years, y=bands["p25"], fill="tonexty", fillcolor=_rgba(BLUE, 0.30),
                                 line=dict(width=0), name="P25 to P75", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=years, y=bands["p50"], name="Monte Carlo median", line=dict(color=BLUE, width=3)))
    if gen is not None:
        fig.add_trace(go.Scatter(x=years, y=gen, name="Generation (deterministic)",
                                 line=dict(color=GRAY, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=years, y=det_em, name="Emission (deterministic)",
                             line=dict(color=ORANGE, width=3, dash="dash")))
    if obs is not None and len(obs):
        err = obs["sigma_kg_h_u"] if "sigma_kg_h_u" in obs else None
        fig.add_trace(go.Scatter(
            x=obs["year"], y=obs["y_plot"], mode="markers", name="Observations",
            marker=dict(color=RED, size=11, line=dict(color="white", width=1.5)),
            error_y=dict(type="data", array=err, visible=err is not None, color=RED, thickness=1.5),
            hovertemplate="%{x:.2f}: %{y:,.1f}<extra>observation</extra>"))
    for label, yr in (markers or {}).items():
        fig.add_vline(x=yr, line=dict(color=GRAY, dash="dash", width=1), annotation_text=label,
                      annotation_position="top left", annotation_font_size=11)
    if selected_year is not None:
        fig.add_vline(x=selected_year, line=dict(color=PURPLE, width=2))
    _layout(fig, "Methane over time", 500, xtitle="Year")
    fig.update_yaxes(title_text=unit, secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="Waste accepted (Mg/yr)", secondary_y=True, rangemode="tozero", showgrid=False)
    return fig


def distribution(samples_em: np.ndarray, det: float, unit: str, year: int) -> go.Figure:
    p5, p50, p95 = np.percentile(samples_em, [5, 50, 95])
    fig = go.Figure(go.Histogram(x=samples_em, nbinsx=60, marker_color=_rgba(BLUE, 0.65), name="Monte Carlo"))
    for v, name, col, dash in ((p5, "P5", GRAY, "dot"), (p50, "P50", BLUE, "solid"), (p95, "P95", GRAY, "dot"),
                               (det, "Deterministic", ORANGE, "dash")):
        fig.add_vline(x=v, line=dict(color=col, dash=dash, width=2), annotation_text=f"{name} {v:,.4g}",
                      annotation_position="top")
    fig.update_layout(showlegend=False, bargap=0.02)
    return _layout(fig, f"Distribution in {year}", 420, ytitle="Iterations", xtitle=unit)


def drivers(series: pd.Series, year: int) -> go.Figure:
    s = series.iloc[::-1]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                           marker_color=[BLUE if v >= 0 else RED for v in s.values],
                           text=[f"{v:.2f}" for v in s.values], textposition="outside"))
    fig.update_xaxes(range=[-1.05, 1.05], title_text="Spearman rank correlation")
    return _layout(fig, f"What drives the uncertainty in {year}?", 420)


def heatmap(k_grid, L0_grid, Z, unit: str, year: int, cur_k: float, cur_L0: float) -> go.Figure:
    fig = go.Figure(go.Heatmap(x=k_grid, y=L0_grid, z=Z.T, colorscale="YlOrRd", colorbar=dict(title=unit),
                               hovertemplate="k=%{x:.3f}<br>L0=%{y:.0f}<br>%{z:,.4g}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[cur_k], y=[cur_L0], mode="markers", name="Current",
                             marker=dict(symbol="x", size=14, color="black", line=dict(width=2, color="white"))))
    fig.update_layout(showlegend=False, hovermode="closest")
    return _layout(fig, f"Emission in {year} for every k and L0", 460, ytitle="L0 (m3/Mg)", xtitle="k (1/yr)")


def k_family(years, curves: Dict[str, np.ndarray], unit: str, cur_label: str) -> go.Figure:
    fig = go.Figure()
    for name, y in curves.items():
        cur = name == cur_label
        fig.add_trace(go.Scatter(x=years, y=y, name=name, line=dict(width=4 if cur else 2, color=ORANGE if cur else None)))
    return _layout(fig, "Effect of k (L0 fixed)", 420, ytitle=unit, xtitle="Year")


def obs_vs_model(cmp: pd.DataFrame) -> go.Figure:
    c = cmp.dropna(subset=["model_p50"])
    hi = float(max(c["kg_h"].max(), c["model_p50"].max()) * 1.15)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", name="1:1", line=dict(color=GRAY, dash="dash")))
    err_x = None
    if "model_p95" in c:
        err_x = dict(type="data", symmetric=False, array=(c["model_p95"] - c["model_p50"]).values,
                     arrayminus=(c["model_p50"] - c["model_p5"]).values, color=_rgba(BLUE, 0.6))
    sig = c["sigma_kg_h"].where(c["sigma_kg_h"] > 0, 0.3 * c["kg_h"]).values
    fig.add_trace(go.Scatter(x=c["model_p50"], y=c["kg_h"], mode="markers", name="Observations",
                             marker=dict(size=11, color=RED, line=dict(color="white", width=1.5)),
                             error_x=err_x, error_y=dict(type="data", array=sig, color=_rgba(RED, 0.6)),
                             text=c["year"].round(1).astype(str), hovertemplate="%{text}<br>model %{x:,.1f}<br>obs %{y:,.1f}<extra></extra>"))
    fig.update_layout(hovermode="closest")
    return _layout(fig, "Observed vs modelled (kg CH4/h)", 440, ytitle="Observed (kg CH4/h)",
                   xtitle="Modelled median (kg CH4/h); bars = P5 to P95")


def misfit(res: Dict[str, object], cur_k: float, cur_L0: float) -> go.Figure:
    z = np.log10(np.maximum(res["chi"], 1e-6))
    fig = go.Figure(go.Heatmap(x=res["k_grid"], y=res["L0_grid"], z=z.T, colorscale="Viridis_r",
                               colorbar=dict(title="log10 misfit"),
                               hovertemplate="k=%{x:.3f}<br>L0=%{y:.0f}<br>log10 misfit %{z:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[res["best_k"]], y=[res["best_L0"]], mode="markers", name="Best fit",
                             marker=dict(symbol="star", size=16, color="white", line=dict(width=2, color="black"))))
    fig.add_trace(go.Scatter(x=[cur_k], y=[cur_L0], mode="markers", name="Current setting",
                             marker=dict(symbol="x", size=13, color="black", line=dict(width=2, color="white"))))
    fig.add_trace(go.Scatter(x=res["k_grid"], y=res["L0_best_for_k"], mode="lines", name="Best L0 for each k",
                             line=dict(color="white", width=1.5, dash="dot")))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(range=[res["L0_grid"].min(), res["L0_grid"].max()])
    return _layout(fig, "Misfit surface: which (k, L0) agree with the observations?", 480,
                   ytitle="L0 (m3/Mg)", xtitle="k (1/yr)")


def waste_chart(years, waste) -> go.Figure:
    wip = np.cumsum(waste)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=years, y=waste, name="Accepted (Mg/yr)", marker_color=_rgba(GRAY, 0.6)))
    fig.add_trace(go.Scatter(x=years, y=wip, name="Waste in place (Mg)", line=dict(color=BLUE, width=3)),
                  secondary_y=True)
    _layout(fig, "Waste acceptance", 360, xtitle="Year")
    fig.update_yaxes(title_text="Mg/yr", secondary_y=False)
    fig.update_yaxes(title_text="Mg in place", secondary_y=True, showgrid=False)
    return fig
