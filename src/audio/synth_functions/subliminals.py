import numpy as np
import soundfile as sf

try:
    import librosa
except Exception:
    librosa = None


def subliminal_encode(duration, sample_rate=44100, **params):
    """Encode an audio file as a high frequency subliminal message.

    Parameters
    ----------
    duration : float
        Length of the output audio in seconds.
    sample_rate : int, optional
        Target sample rate of the output.
    params : dict
        Should contain ``audio_path`` with the file to encode, ``carrierFreq``
        for the ultrasonic modulation frequency (15000-20000 Hz) and ``amp``
        for the output amplitude (0-1.0).
    """
    audio_path = params.get("audio_path")
    carrier = float(params.get("carrierFreq", 17500.0))
    amp = float(params.get("amp", 0.5))

    carrier = np.clip(carrier, 15000.0, 20000.0)

    N = int(duration * sample_rate)
    if audio_path is None:
        return np.zeros((N, 2), dtype=np.float32)

    try:
        data, sr = sf.read(audio_path)
    except Exception:
        return np.zeros((N, 2), dtype=np.float32)

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if sr != sample_rate:
        if librosa is not None:
            data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
        else:
            # fallback simple resample
            t_old = np.linspace(0, len(data) / sr, num=len(data), endpoint=False)
            t_new = np.linspace(0, len(data) / sr, num=int(len(data) * sample_rate / sr), endpoint=False)
            data = np.interp(t_new, t_old, data)

    if len(data) < N:
        data = np.pad(data, (0, N - len(data)))
    else:
        data = data[:N]

    t = np.arange(N) / float(sample_rate)
    mod = np.sin(2 * np.pi * carrier * t)
    out = data * mod

    max_val = np.max(np.abs(out))
    if max_val > 0:
        out = out / max_val
    out *= amp

    stereo = np.column_stack((out, out))
    return stereo.astype(np.float32)
