import os
from typing import Dict, List, Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _estimate_track_duration(track_data: Dict[str, Any]) -> float:
    """Return the total duration of the track based on step durations."""
    total = 0.0
    for step in track_data.get("steps", []):
        try:
            total += float(step.get("duration", 0))
        except (TypeError, ValueError):
            pass
    return total


def visualize_track_timeline(
    track_data: Dict[str, Any],
    save_path: Optional[str] = None,
) -> None:
    """Visualize where different elements overlap across the track.

    Parameters
    ----------
    track_data:
        Dictionary describing the track (same structure as used by
        :func:`sound_creator.generate_audio`).
    save_path:
        If provided, the timeline is saved to this file path instead of
        displayed interactively.
    """
    categories = {
        "binaurals": 0,
        "vocals": 1,
        "effects": 2,
        "noise": 3,
    }

    # Base color scales for per-category shades
    colormaps = {
        "binaurals": px.colors.sequential.Blues,
        "vocals": px.colors.sequential.Oranges,
        "effects": px.colors.sequential.Greens,
        "noise": px.colors.sequential.Greys,
    }

    events: List[Dict[str, Any]] = []
    step_boundaries: List[float] = [0.0]
    current_time = 0.0
    for step in track_data.get("steps", []):
        step_duration = float(step.get("duration", 0))
        voices = step.get("voices", [])
        for voice in voices:
            category = voice.get("category", "binaurals")
            start = current_time
            end = start + step_duration
            amp = float(voice.get("params", {}).get("amp", 1.0))
            label = voice.get("description") or voice.get("synth_function_name", "voice")
            events.append(
                {
                    "category": category,
                    "start": start,
                    "end": end,
                    "amp": amp,
                    "label": label,
                }
            )
        current_time += step_duration
        step_boundaries.append(current_time)

    for clip in track_data.get("clips", []):
        cat = clip.get("category", "effects")
        start = float(clip.get("start", clip.get("start_time", 0)))
        duration = float(clip.get("duration", 0))
        amp = float(clip.get("amp", 1.0))
        label = clip.get("description") or os.path.basename(clip.get("file_path", "clip"))
        if duration > 0:
            events.append(
                {
                    "category": cat,
                    "start": start,
                    "end": start + duration,
                    "amp": amp,
                    "label": label,
                }
            )

    noise_cfg = track_data.get("background_noise", {})
    if isinstance(noise_cfg, dict) and noise_cfg.get("file_path"):
        start = float(noise_cfg.get("start_time", 0))
        duration = _estimate_track_duration(track_data) - start
        amp = float(noise_cfg.get("amp", 1.0))
        label = os.path.basename(noise_cfg.get("file_path"))
        if duration > 0:
            events.append(
                {
                    "category": "noise",
                    "start": start,
                    "end": start + duration,
                    "amp": amp,
                    "label": label,
                }
            )

    if not events:
        print("No timeline events to display.")
        return

    # Assign distinct colors per element within each category
    events_by_cat = {cat: [] for cat in categories.keys()}
    for ev in events:
        events_by_cat.setdefault(ev["category"], []).append(ev)

    for cat, evs in events_by_cat.items():
        cmap = colormaps.get(cat, px.colors.sequential.Blues)
        n = len(evs)
        for i, ev in enumerate(evs):
            frac = i / max(n - 1, 1)
            ev["color"] = px.colors.sample_colorscale(cmap, 0.3 + 0.7 * frac)[0]

    # Build interactive timeline using Plotly
    fig = go.Figure()
    for boundary in step_boundaries:
        fig.add_vline(
            x=boundary,
            line_color="gray",
            line_dash="dash",
            opacity=0.5,
            line_width=1,
        )

    for ev in events:
        y = categories.get(ev["category"], 0)
        width = ev["end"] - ev["start"]
        fig.add_trace(
            go.Bar(
                x=[width],
                y=[y],
                base=[ev["start"]],
                orientation="h",
                marker=dict(
                    color=ev.get("color", "blue"),
                    opacity=min(0.8, 0.4 + ev["amp"] * 0.6),
                ),
                hovertemplate=
                f"{ev.get('label', ev['category'])}<br>Start: {ev['start']} s" +
                f"<br>End: {ev['end']} s<extra></extra>",
                text=[ev.get("label", ev["category"])],
                textposition="inside",
                showlegend=False,
            )
        )

    duration = max(ev["end"] for ev in events)
    fig.update_yaxes(
        tickvals=list(categories.values()),
        ticktext=list(categories.keys()),
        range=[len(categories) - 0.5, -1],
    )
    fig.update_xaxes(title="Time (s)", range=[0, duration], dtick=1)
    fig.update_layout(
        bargap=0.1,
        barmode="stack",
        template="plotly_white",
        height=300,
        margin=dict(l=40, r=40, t=20, b=40),
    )

    if save_path:
        if save_path.lower().endswith(".html"):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path)
    else:
        fig.show()


__all__ = ["visualize_track_timeline"]
