"""Interactive, animated 3D particle scatter plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _column(frame: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Particle CSV needs a {label} column; tried {', '.join(candidates)}")


def write_particle_animation(
    particles_csv: str | Path,
    output_html: str | Path,
    *,
    title: str = "HoloD3 particle measurements",
    embed_plotly: bool = True,
) -> Path:
    """Create an offline-capable Plotly animation from a pipeline CSV."""

    import plotly.graph_objects as go

    source = Path(particles_csv).expanduser().resolve()
    destination = Path(output_html).expanduser().resolve()
    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError(f"Particle CSV is empty: {source}")
    x_col = _column(frame, ("x_um", "seg_xc", "xc"), "x coordinate")
    y_col = _column(frame, ("y_um", "seg_yc", "yc"), "y coordinate")
    z_col = _column(frame, ("depth_um", "z_um"), "z coordinate")
    diameter_col = _column(frame, ("final_diameter_um", "slice_diam_pred_um", "diameter_um"), "diameter")
    frame_col = "frame" if "frame" in frame.columns else None
    if frame_col is None:
        frame = frame.assign(frame=0)
        frame_col = "frame"

    for column in (x_col, y_col, z_col, diameter_col, frame_col):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[x_col, y_col, z_col, diameter_col, frame_col]).copy()
    if frame.empty:
        raise ValueError("Particle CSV has no rows with finite coordinates, frame, and diameter.")

    frame_values = sorted(frame[frame_col].unique().tolist())
    diameter = frame[diameter_col].to_numpy(float)
    low, high = float(np.percentile(diameter, 5)), float(np.percentile(diameter, 95))
    span = max(high - low, 1e-6)

    def marker_sizes(values: pd.Series) -> np.ndarray:
        return 4.0 + 10.0 * np.clip((values.to_numpy(float) - low) / span, 0.0, 1.0)

    def trace(subset: pd.DataFrame) -> go.Scatter3d:
        hover = [
            f"frame={int(row[frame_col])}<br>diameter={float(row[diameter_col]):.2f} µm"
            for _, row in subset.iterrows()
        ]
        return go.Scatter3d(
            x=subset[x_col],
            y=subset[y_col],
            z=subset[z_col],
            mode="markers",
            marker={
                "size": marker_sizes(subset[diameter_col]),
                "color": subset[diameter_col],
                "colorscale": "Viridis",
                "cmin": float(diameter.min()),
                "cmax": float(diameter.max()),
                "colorbar": {"title": "Diameter (µm)"},
                "opacity": 0.75,
            },
            text=hover,
            hoverinfo="text+x+y+z",
        )

    first = frame[frame[frame_col] == frame_values[0]]
    animation_frames = [
        go.Frame(data=[trace(frame[frame[frame_col] == value])], name=str(value)) for value in frame_values
    ]
    steps = [
        {
            "args": [[str(value)], {"frame": {"duration": 250, "redraw": True}, "mode": "immediate"}],
            "label": str(value),
            "method": "animate",
        }
        for value in frame_values
    ]
    figure = go.Figure(data=[trace(first)], frames=animation_frames)
    figure.update_layout(
        title=title,
        template="plotly_white",
        scene={
            "xaxis_title": f"{x_col} (µm)" if x_col.endswith("_um") else x_col,
            "yaxis_title": f"{y_col} (µm)" if y_col.endswith("_um") else y_col,
            "zaxis_title": f"{z_col} (µm)" if z_col.endswith("_um") else z_col,
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "y": 0.0,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 250, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[{"active": 0, "currentvalue": {"prefix": "Frame "}, "steps": steps}],
        annotations=[
            {
                "text": "Independent per-frame measurements; points are not linked as trajectories.",
                "xref": "paper",
                "yref": "paper",
                "x": 1.0,
                "y": 0.0,
                "showarrow": False,
                "xanchor": "right",
            }
        ],
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        destination,
        include_plotlyjs=True if embed_plotly else "cdn",
        full_html=True,
        auto_play=False,
    )
    return destination
