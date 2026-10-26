"""
Interactive scientific Plotly visualizations for satellite clock and ephemeris analysis.
Includes 3D ECEF orbit geometry, clock bias time series, predicted-vs-actual, and residual diagnostics.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_clock_bias_series(
    df: pd.DataFrame,
    sat_id: str,
    target_col: str = "clock_bias_us",
    units: str = "μs",
) -> go.Figure:
    """
    Generate an interactive Plotly line chart of satellite clock bias over time.
    """
    sat_df = df[df["sat_id"] == sat_id].sort_values(by="epoch")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=sat_df["epoch"],
            y=sat_df[target_col],
            mode="lines+markers",
            marker=dict(size=3),
            line=dict(color="#1f77b4", width=1.5),
            name=f"{sat_id} Clock Bias ({units})",
            hovertemplate="<b>Epoch:</b> %{x|%Y-%m-%d %H:%M:%S}<br><b>Bias:</b> %{y:.6f} " + units + "<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Satellite {sat_id} Clock Bias Time Series",
        xaxis_title="Epoch (UTC / GPS)",
        yaxis_title=f"Clock Bias ({units})",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_orbit_3d(
    df: pd.DataFrame,
    sat_id: str,
) -> go.Figure:
    """
    Generate an interactive 3D Earth-Centered Earth-Fixed (ECEF) orbit trajectory.
    Includes a reference sphere representing Earth (radius ~6378 km).
    """
    sat_df = df[df["sat_id"] == sat_id].sort_values(by="epoch")

    fig = go.Figure()

    # 1. Earth sphere reference
    r_earth = 6378.137  # Earth radius in km
    u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 15j]
    x_earth = r_earth * np.cos(u) * np.sin(v)
    y_earth = r_earth * np.sin(u) * np.sin(v)
    z_earth = r_earth * np.cos(v)

    fig.add_trace(
        go.Surface(
            x=x_earth,
            y=y_earth,
            z=z_earth,
            colorscale=[[0, "#2c7bb6"], [1, "#00a6ca"]],
            showscale=False,
            opacity=0.35,
            name="Earth (WGS84)",
            hoverinfo="skip",
        )
    )

    # 2. Satellite trajectory in ECEF
    fig.add_trace(
        go.Scatter3d(
            x=sat_df["x_km"],
            y=sat_df["y_km"],
            z=sat_df["z_km"],
            mode="lines+markers",
            marker=dict(size=2.5, color=np.arange(len(sat_df)), colorscale="Viridis", opacity=0.8),
            line=dict(color="#d7191c", width=3),
            name=f"{sat_id} Orbit Path",
            hovertemplate=(
                "<b>Epoch:</b> %{text}<br>"
                "<b>X:</b> %{x:.1f} km<br>"
                "<b>Y:</b> %{y:.1f} km<br>"
                "<b>Z:</b> %{z:.1f} km<extra></extra>"
            ),
            text=sat_df["epoch"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        )
    )

    # Start and End points
    if len(sat_df) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=[sat_df["x_km"].iloc[0]],
                y=[sat_df["y_km"].iloc[0]],
                z=[sat_df["z_km"].iloc[0]],
                mode="markers",
                marker=dict(size=7, color="green", symbol="circle"),
                name="Start Epoch",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[sat_df["x_km"].iloc[-1]],
                y=[sat_df["y_km"].iloc[-1]],
                z=[sat_df["z_km"].iloc[-1]],
                mode="markers",
                marker=dict(size=7, color="red", symbol="diamond"),
                name="Latest Epoch",
            )
        )

    fig.update_layout(
        title=f"3D ECEF Satellite Orbit: {sat_id}",
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(orientation="h", yanchor="bottom", y=0.95, xanchor="left", x=0),
    )
    return fig


def plot_orbit_components(
    df: pd.DataFrame,
    sat_id: str,
) -> go.Figure:
    """
    Generate 2D subplots for X, Y, and Z coordinate time series.
    """
    sat_df = df[df["sat_id"] == sat_id].sort_values(by="epoch")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=(f"{sat_id} X (km)", f"{sat_id} Y (km)", f"{sat_id} Z (km)")
    )

    fig.add_trace(go.Scatter(x=sat_df["epoch"], y=sat_df["x_km"], line=dict(color="#1b9e77")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sat_df["epoch"], y=sat_df["y_km"], line=dict(color="#d95f02")), row=2, col=1)
    fig.add_trace(go.Scatter(x=sat_df["epoch"], y=sat_df["z_km"], line=dict(color="#7570b3")), row=3, col=1)

    fig.update_layout(
        title=f"ECEF Coordinate Trajectories: {sat_id}",
        template="plotly_white",
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_forecast_vs_actual(
    target_epochs: Union[List, np.ndarray, pd.Series],
    y_actual: Union[List, np.ndarray, pd.Series],
    y_predicted: Union[List, np.ndarray, pd.Series],
    y_baseline: Optional[Union[List, np.ndarray, pd.Series]] = None,
    target_name: str = "clock_bias_us",
    units: str = "μs",
    sat_id: str = "G01",
    model_name: str = "Random Forest",
    horizon_hours: int = 1,
) -> go.Figure:
    """
    Interactive comparison of predicted values vs actual reference observations and persistence.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=target_epochs,
            y=y_actual,
            mode="lines+markers",
            line=dict(color="#2ca02c", width=2),
            marker=dict(size=4),
            name="Actual Reference",
            hovertemplate="<b>Actual:</b> %{y:.6f} " + units + "<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=target_epochs,
            y=y_predicted,
            mode="lines+markers",
            line=dict(color="#d62728", width=2, dash="dash"),
            marker=dict(size=4),
            name=f"Predicted ({model_name})",
            hovertemplate="<b>Predicted:</b> %{y:.6f} " + units + "<extra></extra>",
        )
    )

    if y_baseline is not None:
        fig.add_trace(
            go.Scatter(
                x=target_epochs,
                y=y_baseline,
                mode="lines",
                line=dict(color="#7f7f7f", width=1.5, dash="dot"),
                name="Persistence Baseline",
                hovertemplate="<b>Baseline:</b> %{y:.6f} " + units + "<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"{sat_id} Forecast vs Actual ({horizon_hours}h Horizon) - {target_name}",
        xaxis_title="Target Epoch",
        yaxis_title=f"{target_name} ({units})",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_residuals(
    target_epochs: Union[List, np.ndarray, pd.Series],
    residuals: Union[List, np.ndarray, pd.Series],
    units: str = "μs",
) -> go.Figure:
    """
    Plot residuals over time and error distribution histogram side-by-side.
    """
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.65, 0.35],
        subplot_titles=("Forecast Residuals Over Time (Predicted - Actual)", "Residual Distribution")
    )

    # Time series of residuals
    fig.add_trace(
        go.Scatter(
            x=target_epochs,
            y=residuals,
            mode="markers+lines",
            marker=dict(size=4, color="#e41a1c"),
            line=dict(width=1),
            name="Residual",
        ),
        row=1, col=1,
    )
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)

    # Distribution histogram
    fig.add_trace(
        go.Histogram(
            y=residuals,
            nbinsy=25,
            marker_color="#377eb8",
            name="Residual Count",
        ),
        row=1, col=2,
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text=f"Error ({units})", row=1, col=1)
    fig.update_xaxes(title_text="Frequency", row=1, col=2)
    return fig


def plot_horizon_comparison(
    comparison_df: pd.DataFrame,
    metric_col: str = "mae",
    metric_label: str = "Mean Absolute Error",
    units: str = "μs",
) -> go.Figure:
    """
    Plot bar comparison of model vs baseline across horizons (1h, 6h, 12h, 24h).
    """
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[f"{h}h" for h in comparison_df["horizon_hours"]],
            y=comparison_df["model_mae"],
            name="Forecasting Model",
            marker_color="#2b83ba",
        )
    )

    if "baseline_mae" in comparison_df.columns:
        fig.add_trace(
            go.Bar(
                x=[f"{h}h" for h in comparison_df["horizon_hours"]],
                y=comparison_df["baseline_mae"],
                name="Persistence Baseline",
                marker_color="#fdae61",
            )
        )

    fig.update_layout(
        title=f"Forecast Error vs Horizon ({metric_label})",
        xaxis_title="Forecast Horizon",
        yaxis_title=f"{metric_label} ({units})",
        barmode="group",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
