from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

# Categories used when building the visual timeline
CATEGORY_BINARUALS = "binaurals"
CATEGORY_VOCALS = "vocals"
CATEGORY_EFFECTS = "effects"
CATEGORY_NOISE = "noise"

@dataclass
class TimelineEntry:
    start: float
    end: float
    category: str
    amplitude_points: List[Tuple[float, float]]
    info: Dict[str, Any]

def _categorize_voice(func_name: str) -> str:
    name = (func_name or "").lower()
    if "noise" in name:
        return CATEGORY_NOISE
    if any(k in name for k in ("binaural", "monaural", "isochronic", "qam", "spatial", "waveshaping", "stereo_am")):
        return CATEGORY_BINARUALS
    if any(k in name for k in ("vocal", "voice", "subliminal")):
        return CATEGORY_VOCALS
    if "effect" in name:
        return CATEGORY_EFFECTS
    return CATEGORY_EFFECTS

def _extract_envelope_points(env: Dict[str, Any], duration: float) -> List[Tuple[float, float]]:
    if not env or not isinstance(env, dict):
        return [(0.0, 1.0), (duration, 1.0)]
    env_type = env.get("type")
    params = env.get("params", {})
    if env_type == "linear_fade":
        fade_dur = float(params.get("fade_duration", 0.0))
        start_amp = float(params.get("start_amp", 0.0))
        end_amp = float(params.get("end_amp", 1.0))
        if fade_dur <= 0:
            return [(0.0, end_amp), (duration, end_amp)]
        return [(0.0, start_amp), (min(fade_dur, duration), end_amp), (duration, end_amp)]
    if env_type == "adsr":
        a = float(params.get("attack", 0))
        d = float(params.get("decay", 0))
        s = float(params.get("sustain_level", 1))
        r = float(params.get("release", 0))
        pts = [(0.0, 0.0)]
        t = 0.0
        t += a
        pts.append((t, 1.0))
        t += d
        pts.append((t, s))
        end_sustain = max(duration - r, t)
        pts.append((end_sustain, s))
        pts.append((duration, 0.0))
        return pts
    if env_type == "points":
        pts = []
        for p in params.get("points", []):
            try:
                pts.append((float(p[0]), float(p[1])))
            except Exception:
                continue
        pts.sort()
        if not pts or pts[0][0] > 0:
            pts.insert(0, (0.0, 1.0))
        if pts[-1][0] < duration:
            pts.append((duration, pts[-1][1]))
        return pts
    return [(0.0, 1.0), (duration, 1.0)]

def build_timeline(track_data: Dict[str, Any]) -> List[TimelineEntry]:
    timeline: List[TimelineEntry] = []
    current_time = 0.0
    for step in track_data.get("steps", []):
        duration = float(step.get("duration", 0.0))
        for voice in step.get("voices", []):
            category = _categorize_voice(voice.get("synth_function_name", ""))
            env_points = _extract_envelope_points(voice.get("volume_envelope"), duration)
            timeline.append(TimelineEntry(
                start=current_time,
                end=current_time + duration,
                category=category,
                amplitude_points=env_points,
                info={"voice": voice.get("description", "")}
            ))
        current_time += duration
    noise = track_data.get("background_noise")
    if noise:
        n_dur = float(noise.get("duration", 0.0))
        start = float(noise.get("start_time", 0.0))
        env_points = _extract_envelope_points(noise.get("volume_envelope"), n_dur)
        timeline.append(TimelineEntry(
            start=start,
            end=start + n_dur,
            category=CATEGORY_NOISE,
            amplitude_points=env_points,
            info={"noise_type": noise.get("noise_type", "pink")}
        ))
    return timeline

__all__ = [
    "CATEGORY_BINARUALS",
    "CATEGORY_VOCALS",
    "CATEGORY_EFFECTS",
    "CATEGORY_NOISE",
    "TimelineEntry",
    "build_timeline",
]
