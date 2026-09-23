"""Plotly-Abbildungen der MOEA/D-Demo: Streudiagramm (Population + Archiv), Vergleichsabbildungen, Sweep.
Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ARCHIVE_COLOR = "#54a24b"
POP_COLOR = "#9ecae9"
GA_COLOR = "#e45756"
NSGA2_COLOR = "#4c78a8"
MOEAD_COLOR = "#f58518"
REF_COLOR = "#7f7f7f"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.15), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_population_scatter(pop_obj, archive_obj):
    """Aktuelle Arbeitspopulation (blass-blau) und das externe Archiv (grün, verbunden) im Zielraum."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pop_obj[:, 0], y=pop_obj[:, 1], mode="markers", marker=dict(size=6, color=POP_COLOR), name="Arbeitspopulation"))
    order = np.argsort(archive_obj[:, 0])
    fig.add_trace(go.Scatter(x=archive_obj[order, 0], y=archive_obj[order, 1], mode="markers+lines", line=dict(color=ARCHIVE_COLOR, width=1.5, dash="dot"),
                              marker=dict(size=9, color=ARCHIVE_COLOR, line=dict(width=1, color="white")), name="Externes Archiv"))
    fig.update_xaxes(title_text="Distanz (km)")
    fig.update_yaxes(title_text="CO2-Kosten")
    return _base(fig, 380)


def build_archive_size_curve(archive_history):
    xs = list(range(len(archive_history)))
    ys = [len(a) for a in archive_history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=MOEAD_COLOR, width=2.5), showlegend=False))
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="Größe des Archivs")
    return _base(fig, 260)


def build_comparison(report):
    fig = go.Figure()
    fig.add_trace(go.Box(y=report["reached_all"], name="MOEA/D", marker_color=MOEAD_COLOR, boxpoints="all"))
    fig.add_trace(go.Bar(x=["Gewichtete Summe<br>(GA-Demo)"], y=[report["ga_weighted_sum_reached"]], marker_color=GA_COLOR, showlegend=False, width=0.4))
    fig.add_trace(go.Bar(x=["NSGA-II<br>(nsga2-demo)"], y=[report["nsga2_reached"]], marker_color=NSGA2_COLOR, showlegend=False, width=0.4))
    fig.add_hline(y=report["front_size"], line=dict(color=REF_COLOR, dash="dot"), annotation_text="volle Front", annotation_position="top right")
    fig.update_yaxes(title_text="Getroffene Frontpunkte")
    return _base(fig, 360)


def build_neighborhood_experiment(rows):
    fig = go.Figure()
    for r in rows:
        fig.add_trace(go.Box(y=r["archive_size_all"], name=str(r["t"]), marker_color=MOEAD_COLOR, boxpoints="all", showlegend=False))
    fig.update_xaxes(title_text="Nachbarschaftsgröße T")
    fig.update_yaxes(title_text="Archivgröße")
    return _base(fig, 340)


def build_sweep(rows, param_label):
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["archive_size"] for r in rows], mode="lines+markers", line=dict(color=MOEAD_COLOR, width=2.5), showlegend=False))
    fig.update_xaxes(title_text=param_label)
    fig.update_yaxes(title_text="Archivgröße (Mittel)")
    return _base(fig, 300)
