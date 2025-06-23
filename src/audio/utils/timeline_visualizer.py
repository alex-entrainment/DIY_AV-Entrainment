import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List, Any, Optional


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

    # Base colormaps for per-category shades
    colormaps = {
        "binaurals": plt.cm.Blues,
        "vocals": plt.cm.Oranges,
        "effects": plt.cm.Greens,
        "noise": plt.cm.Greys,
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
        cmap = colormaps.get(cat, plt.cm.Blues)
        n = len(evs)
        for i, ev in enumerate(evs):
            frac = i / max(n - 1, 1)
            ev["color"] = cmap(0.3 + 0.7 * frac)

    fig, ax = plt.subplots(figsize=(12, 4))
    for boundary in step_boundaries:
        ax.axvline(boundary, color="k", linestyle="--", linewidth=0.5, alpha=0.3)

    for ev in events:
        y = categories.get(ev["category"], 0)
        width = ev["end"] - ev["start"]
        rect = Rectangle(
            (ev["start"], y - 0.4),
            width,
            0.8,
            facecolor=ev.get("color", "tab:blue"),
            alpha=min(0.8, 0.4 + ev["amp"] * 0.6),
        )
        ax.add_patch(rect)
        ax.text(
            ev["start"] + 0.05,
            y,
            ev.get("label", ev["category"]),
            va="center",
            ha="left",
            fontsize=8,
            color="white",
        )

    duration = max(ev["end"] for ev in events)
    ax.set_yticks(list(categories.values()))
    ax.set_yticklabels(list(categories.keys()))
    ax.set_xlabel("Time (s)")
    ax.set_xlim(0, duration)
    ax.set_ylim(-1, len(categories))
    ax.set_xticks([t for t in range(int(duration) + 1)])
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


__all__ = ["visualize_track_timeline"]
