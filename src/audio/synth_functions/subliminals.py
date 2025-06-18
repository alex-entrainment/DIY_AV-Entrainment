import numpy as np
import soundfile as sf

try:
    import librosa
except Exception:
    librosa = None


def subliminal_encode(duration, sample_rate=44100, **params):
    """Encode one or more audio files as high frequency subliminal messages.

    Parameters
    ----------
    duration : float
        Length of the output audio in seconds.
    sample_rate : int, optional
        Target sample rate of the output.
    params : dict
        ``audio_path`` for a single file or ``audio_paths`` for multiple files
        to encode. ``mode`` selects ``'stack'`` to mix all files together or
        ``'sequence'`` to play them one after another. ``carrierFreq`` specifies
        the ultrasonic modulation frequency (15000-20000 Hz) and ``amp`` sets the
        final amplitude (0-1.0).
    """

    carrier = float(params.get("carrierFreq", 17500.0))
    amp = float(params.get("amp", 0.5))
    mode = str(params.get("mode", "sequence")).lower()

    # Gather list of file paths
    audio_paths = params.get("audio_paths")
    if audio_paths is None:
        audio_path = params.get("audio_path")
        audio_paths = [audio_path] if audio_path else []
    elif isinstance(audio_paths, str):
        audio_paths = [p.strip() for p in audio_paths.split(";") if p.strip()]

    carrier = np.clip(carrier, 15000.0, 20000.0)

    N = int(duration * sample_rate)
    if not audio_paths:
        return np.zeros((N, 2), dtype=np.float32)

    def load_and_modulate(path):
        try:
            data, sr = sf.read(path)
        except Exception:
            return np.array([], dtype=np.float32)

        if data.ndim > 1:
            data = np.mean(data, axis=1)

        if sr != sample_rate:
            if librosa is not None:
                data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
            else:
                t_old = np.linspace(0, len(data) / sr, num=len(data), endpoint=False)
                t_new = np.linspace(0, len(data) / sr, num=int(len(data) * sample_rate / sr), endpoint=False)
                data = np.interp(t_new, t_old, data)

        if len(data) == 0:
            return np.array([], dtype=np.float32)

        t = np.arange(len(data)) / float(sample_rate)
        mod = np.sin(2 * np.pi * carrier * t)
        out = data * mod
        max_val = np.max(np.abs(out))
        if max_val > 0:
            out = out / max_val
        return out.astype(np.float32)

    segments = [load_and_modulate(p) for p in audio_paths]
    segments = [s for s in segments if len(s) > 0]
    if not segments:
        return np.zeros((N, 2), dtype=np.float32)

    if mode == "stack":
        out = np.zeros(N, dtype=np.float32)
        for seg in segments:
            repeated = np.resize(seg, N)
            out += repeated
        out /= len(segments)
    else:  # sequence mode
        out = np.zeros(N, dtype=np.float32)
        pos = 0
        idx = 0
        pause_samples = sample_rate  # one second pause between segments
        while pos < N:
            seg = segments[idx % len(segments)]
            seg_len = len(seg)
            copy_len = min(seg_len, N - pos)
            out[pos:pos + copy_len] = seg[:copy_len]
            pos += copy_len
            if pos >= N:
                break
            # insert silence after each subliminal
            pause_len = min(pause_samples, N - pos)
            pos += pause_len  # zeros are already present in 'out'
            idx += 1

    max_val = np.max(np.abs(out))
    if max_val > 0:
        out = out / max_val
    out *= amp

    stereo = np.column_stack((out, out))
    return stereo.astype(np.float32)
