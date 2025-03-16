
import numpy as np
import librosa
import soundfile as sf

# -----------------------------------------------------------
# Data Classes
# -----------------------------------------------------------

class Node:
    def __init__(self, duration, base_freq, beat_freq, volume_left, volume_right):
        self.duration = float(duration)
        self.base_freq = float(base_freq)
        self.beat_freq = float(beat_freq)
        self.volume_left = float(volume_left)
        self.volume_right = float(volume_right)


class Voice:
    def __init__(self, nodes, sample_rate=44100):
        self.nodes = nodes
        self.sample_rate = sample_rate
        self._build_timeline()

    def _build_timeline(self):
        self.start_times = [0.0]
        for i in range(len(self.nodes) - 1):
            self.start_times.append(self.start_times[i] + self.nodes[i].duration)
        self.total_duration = sum(n.duration for n in self.nodes)
        self.num_samples = int(self.total_duration * self.sample_rate)

    def _get_interpolated_params(self, t):
        """
        Returns (base_freq, beat_freq, volume_left, volume_right)
        with linear interpolation from node to node.
        """
        if t >= self.total_duration:
            last = self.nodes[-1]
            return (last.base_freq, last.beat_freq,
                    last.volume_left, last.volume_right)

        for i in range(len(self.nodes) - 1):
            t0 = self.start_times[i]
            t1 = self.start_times[i+1]
            if t0 <= t < t1:
                frac = (t - t0) / (t1 - t0)
                n0, n1 = self.nodes[i], self.nodes[i+1]
                base = n0.base_freq + frac*(n1.base_freq - n0.base_freq)
                beat = n0.beat_freq + frac*(n1.beat_freq - n0.beat_freq)
                vl = n0.volume_left + frac*(n1.volume_left - n0.volume_left)
                vr = n0.volume_right + frac*(n1.volume_right - n0.volume_right)
                return (base, beat, vl, vr)

        # If we somehow didn't return, it means we're after last node but < total_duration
        last = self.nodes[-1]
        return (last.base_freq, last.beat_freq, last.volume_left, last.volume_right)

    def generate_samples(self):
        """
        Subclasses override this method to produce the actual audio buffer
        as a NumPy array of shape (num_samples, 2).
        """
        raise NotImplementedError()

# -----------------------------------------------------------
# Spatial Angle Modulation (SAM) Voice
# -----------------------------------------------------------

class SpatialAngleModVoice(Voice):
    """
    A general-purpose Spatial Angle Modulation (SAM) voice that lets you:
      - Use a carrier frequency (or each node's base_freq, if you wish)
      - Define an arc "movement" frequency (arc_freq) for each channel
      - Define max arc angles, optional phase offsets, or different arcs left vs right
    This replicates the Monroe Institute concept that you can move a perceived
    source in a stereo field at certain rates to encourage brainwave entrainment,
    especially in gamma or other bands, but generalized for user-defined arcs.
    """

    def __init__(self, nodes, sample_rate=44100,
                 # Arc shape parameters
                 carrier_from_nodes=False,      # If True, each node's base_freq is used as carrier
                 carrier_freq=200.0,           # Default if not from nodes
                 arc_freq_left=10.0,           # Movement freq for left channel
                 arc_freq_right=10.0,          # Movement freq for right channel
                 arc_peak_left=np.pi/2,        # Max angle offset (radians) for left channel
                 arc_peak_right=np.pi/2,       # Max angle offset (radians) for right channel
                 phase_offset_left=0.0,        # Additional offset for left arc
                 phase_offset_right=0.0,       # Additional offset for right arc,
                 arc_function='sin'            # 'sin' or 'triangle' or any shape you define
                 ):
        """
        :param nodes: list of Node objects (for durations & volume).
        :param sample_rate: standard sample rate
        :param carrier_from_nodes: whether to read node.base_freq at each sample
                                   or just use a fixed 'carrier_freq' param
        :param carrier_freq: base frequency if not using node base_freq
        :param arc_freq_left: how many arcs/sec on left
        :param arc_freq_right: how many arcs/sec on right
        :param arc_peak_left: amplitude (peak) of angle for left channel
        :param arc_peak_right: amplitude (peak) of angle for right channel
        :param phase_offset_left: shift arcs on left by some radians
        :param phase_offset_right: shift arcs on right by some radians
        :param arc_function: 'sin' is typical. You could do custom wave shapes.
        """
        super().__init__(nodes, sample_rate)
        self.carrier_from_nodes = carrier_from_nodes
        self.carrier_freq = carrier_freq

        self.arc_freq_left = arc_freq_left
        self.arc_freq_right = arc_freq_right
        self.arc_peak_left = arc_peak_left
        self.arc_peak_right = arc_peak_right
        self.phase_offset_left = phase_offset_left
        self.phase_offset_right = phase_offset_right

        self.arc_function = arc_function.lower()

    def _arc_value(self, t, freq, peak, phase):
        """
        Return the instantaneous angle offset for a given time t,
        using the chosen wave shape. Currently supports 'sin' or 'triangle'.
        """
        if self.arc_function == 'sin':
            return peak * np.sin(2.0 * np.pi * freq * t + phase)
        elif self.arc_function == 'triangle':
            # A quick triangular wave approach:
            # We'll use sawtooth & then convert to triangle, or direct:
            # triangle( x ) = 2*|2*((x+0.25) mod 1) -1| -1
            # But simpler is to do a linear ramp modded. We'll do a simplified version.
            # We'll do 'abs(sawtooth)', etc. For brevity:
            import scipy.signal
            saw = scipy.signal.sawtooth(2.0*np.pi*freq*t + phase, 0.5)  # tri wave
            return peak * saw
        else:
            # Fallback: sin
            return peak * np.sin(2.0 * np.pi * freq * t + phase)

    def generate_samples(self):
        audio = np.zeros((self.num_samples, 2), dtype=np.float32)

        for i in range(self.num_samples):
            t = i / self.sample_rate

            # Interpolate node volumes, etc.
            base, beat, vol_left, vol_right = self._get_interpolated_params(t)

            # Choose a carrier freq
            carrier = base if self.carrier_from_nodes else self.carrier_freq

            # Determine arc offsets for left/right
            # i.e. the user "angle" in radians
            angle_l = self._arc_value(t, self.arc_freq_left, self.arc_peak_left, self.phase_offset_left)
            angle_r = self._arc_value(t, self.arc_freq_right, self.arc_peak_right, self.phase_offset_right)

            # Basic stereo signals:
            # left ear => sin(2 pi carrier * t + angle_l)
            # right ear => sin(2 pi carrier * t + angle_r)
            phase_left = 2.0*np.pi*carrier*t + angle_l
            phase_right = 2.0*np.pi*carrier*t + angle_r

            s_left = np.sin(phase_left) * vol_left
            s_right = np.sin(phase_right) * vol_right

            audio[i, 0] = s_left
            audio[i, 1] = s_right

        return audio

# -----------------------------------------------------------
# Binaural Beats (unchanged)
# -----------------------------------------------------------

class BinauralBeatVoice(Voice):
    def generate_samples(self):
        audio = np.zeros((self.num_samples, 2), dtype=np.float32)
        for i in range(self.num_samples):
            t = i / self.sample_rate
            base, beat, vl, vr = self._get_interpolated_params(t)
            left_freq = base - beat/2.0
            right_freq = base + beat/2.0
            audio[i, 0] = np.sin(2*np.pi*left_freq*t) * vl
            audio[i, 1] = np.sin(2*np.pi*right_freq*t) * vr
        return audio


# -----------------------------------------------------------
# Trapezoidal Isochronic (NEW approach)
# -----------------------------------------------------------
def trapezoid_envelope(t_in_cycle, cycle_len, ramp_percent, gap_percent):
    """
    Returns amplitude in [0..1] for the time-in-cycle.
    We break the cycle into four parts:
      - ramp_up
      - stable_top
      - ramp_down
      - gap (zero amplitude)

    cycle_len = 1.0 / beat
    ramp_percent: fraction (0..0.5) that controls ramp up/down
    gap_percent : fraction (0..0.5) that controls trailing gap
    The remainder is the stable_top portion.
    """
    if cycle_len <= 0:
        return 0.0

    # The portion of the cycle used for the audible pulse
    audible_len = cycle_len * (1 - gap_percent)
    # The total portion for ramp up and ramp down combined
    ramp_total = audible_len * ramp_percent * 2.0
    # The stable portion is what's left in audible_len
    stable_len = audible_len - ramp_total
    if stable_len < 0:
        # If ramp_percent + gap_percent is too large, clamp stable to 0
        ramp_total = audible_len
        stable_len = 0

    # Boundaries:
    #   ramp_up ends at ramp_up_len
    #   stable ends at ramp_up_len + stable_len
    #   ramp_down ends at ramp_up_len + stable_len + ramp_down_len
    ramp_up_len = ramp_total / 2.0
    ramp_down_len = ramp_total / 2.0

    if t_in_cycle >= audible_len:
        # inside the gap portion
        return 0.0

    # Ramp up region
    if t_in_cycle < ramp_up_len:
        return t_in_cycle / ramp_up_len

    # Stable top region
    if t_in_cycle < (ramp_up_len + stable_len):
        return 1.0

    # Ramp down region
    time_into_down = t_in_cycle - (ramp_up_len + stable_len)
    return 1.0 - (time_into_down / ramp_down_len)


class IsochronicVoice(Voice):
    """
    A trapezoidal isochronic generator: each beat cycle has
    ramp up, stable top, ramp down, and gap. We then multiply
    that amplitude by a carrier sine wave at base_freq.
    """
    def __init__(self, nodes, sample_rate=44100,
                 ramp_percent=0.2, gap_percent=0.15, amplitude=1.0):
        super().__init__(nodes, sample_rate)
        self.ramp_percent = ramp_percent
        self.gap_percent = gap_percent
        self.amplitude = amplitude

    def generate_samples(self):
        audio = np.zeros((self.num_samples, 2), dtype=np.float32)
        for i in range(self.num_samples):
            t = i / self.sample_rate
            base, beat, vl, vr = self._get_interpolated_params(t)
            if beat <= 0:
                continue
            cycle_len = 1.0 / beat
            t_in_cycle = (t % cycle_len)
            env = trapezoid_envelope(t_in_cycle, cycle_len,
                                     self.ramp_percent, self.gap_percent)
            # Multiply by a carrier wave at base_freq
            carrier = np.sin(2 * np.pi * base * t) * env * self.amplitude
            audio[i, 0] = carrier * vl
            audio[i, 1] = carrier * vr
        return audio


class AltIsochronicVoice(Voice):
    """
    Same trapezoidal approach, but we alternate entire cycles
    between left and right channels.
    """
    def __init__(self, nodes, sample_rate=44100,
                 ramp_percent=0.2, gap_percent=0.15, amplitude=1.0):
        super().__init__(nodes, sample_rate)
        self.ramp_percent = ramp_percent
        self.gap_percent = gap_percent
        self.amplitude = amplitude

    def generate_samples(self):
        audio = np.zeros((self.num_samples, 2), dtype=np.float32)
        current_left = True
        cycle_start = 0.0
        for i in range(self.num_samples):
            t = i / self.sample_rate
            base, beat, vl, vr = self._get_interpolated_params(t)
            if beat <= 0:
                continue
            cycle_duration = 1.0 / beat
            if t - cycle_start >= cycle_duration:
                cycle_start += cycle_duration
                current_left = not current_left

            # time within current beat cycle
            t_in_cycle = t - cycle_start
            env = trapezoid_envelope(t_in_cycle, cycle_duration,
                                     self.ramp_percent, self.gap_percent)
            carrier = np.sin(2 * np.pi * base * t) * env * self.amplitude
            if current_left:
                audio[i, 0] = carrier * vl
                audio[i, 1] = 0.0
            else:
                audio[i, 0] = 0.0
                audio[i, 1] = carrier * vr
        return audio


class AltIsochronic2Voice(Voice):
    """
    'Intra-beat' approach: half cycle on left, half cycle on right.
    We'll still shape each half-cycle with the trapezoidal envelope.
    """
    def __init__(self, nodes, sample_rate=44100,
                 ramp_percent=0.2, gap_percent=0.15, amplitude=1.0):
        super().__init__(nodes, sample_rate)
        self.ramp_percent = ramp_percent
        self.gap_percent = gap_percent
        self.amplitude = amplitude

    def generate_samples(self):
        audio = np.zeros((self.num_samples, 2), dtype=np.float32)
        for i in range(self.num_samples):
            t = i / self.sample_rate
            base, beat, vl, vr = self._get_interpolated_params(t)
            if beat <= 0:
                continue
            cycle_len = 1.0 / beat
            t_in_cycle = (t % cycle_len)

            # We'll split the cycle in half: 0..0.5 for left, 0.5..1.0 for right
            half_cycle = cycle_len / 2.0
            if t_in_cycle < half_cycle:
                # left channel
                env = trapezoid_envelope(t_in_cycle, half_cycle,
                                         self.ramp_percent, self.gap_percent)
                carrier = np.sin(2*np.pi*base*t) * env * self.amplitude
                audio[i, 0] = carrier * vl
                audio[i, 1] = 0.0
            else:
                # right channel
                # time in half-cycle
                t_in_half = t_in_cycle - half_cycle
                env = trapezoid_envelope(t_in_half, half_cycle,
                                         self.ramp_percent, self.gap_percent)
                carrier = np.sin(2*np.pi*base*t) * env * self.amplitude
                audio[i, 0] = 0.0
                audio[i, 1] = carrier * vr
        return audio


# -----------------------------------------------------------
# Pink Noise Voice (unchanged)
# -----------------------------------------------------------

class PinkNoiseVoice(Voice):
    def generate_samples(self):
        audio = np.zeros((self.num_samples, 2), dtype=np.float32)
        white = np.random.normal(0, 1, self.num_samples).astype(np.float32)
        pink = np.zeros(self.num_samples, dtype=np.float32)
        alpha = 0.98
        pink[0] = white[0]
        for i in range(1, self.num_samples):
            pink[i] = alpha*pink[i-1] + (1-alpha)*white[i]
        for i in range(self.num_samples):
            t = i / self.sample_rate
            _, _, vl, vr = self._get_interpolated_params(t)
            audio[i, 0] = pink[i] * vl * 0.1
            audio[i, 1] = pink[i] * vr * 0.1
        return audio


# -----------------------------------------------------------
# External Audio Voice (unchanged)
# -----------------------------------------------------------

class ExternalAudioVoice(Voice):
    def __init__(self, nodes, file_path, sample_rate=44100):
        super().__init__(nodes, sample_rate)
        self.file_path = file_path
        data, sr = librosa.load(self.file_path, sr=sample_rate, mono=False)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        self.ext_audio = data
        self.ext_length = self.ext_audio.shape[1]

    def generate_samples(self):
        audio = np.zeros((self.num_samples, 2), dtype=np.float32)
        for i in range(self.num_samples):
            t = i / self.sample_rate
            _, _, vl, vr = self._get_interpolated_params(t)
            idx = i
            if idx < self.ext_length:
                if self.ext_audio.shape[0] == 1:
                    val_l = self.ext_audio[0, idx]
                    val_r = self.ext_audio[0, idx]
                else:
                    val_l = self.ext_audio[0, idx]
                    val_r = self.ext_audio[1, idx]
            else:
                val_l = 0.0
                val_r = 0.0
            audio[i, 0] = val_l * vl
            audio[i, 1] = val_r * vr
        return audio


# -----------------------------------------------------------
# Track-Wide Generation
# -----------------------------------------------------------

def generate_track_audio(voices, sample_rate=44100):
    track_length_seconds = max(v.total_duration for v in voices)
    total_samples = int(track_length_seconds * sample_rate)
    mixed = np.zeros((total_samples, 2), dtype=np.float32)
    for v in voices:
        buf = v.generate_samples()
        length = buf.shape[0]
        mixed[:length] += buf
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed /= max_val
    return mixed


# -----------------------------------------------------------
# Export Helpers
# -----------------------------------------------------------

def export_wav(audio_data, sample_rate, file_path):
    sf.write(file_path, audio_data, sample_rate, format='WAV')

def export_flac(audio_data, sample_rate, file_path):
    sf.write(file_path, audio_data, sample_rate, format='FLAC')

def export_mp3(audio_data, sample_rate, file_path):
    from pydub import AudioSegment
    temp_wav = file_path + ".temp.wav"
    sf.write(temp_wav, audio_data, sample_rate, format='WAV')
    seg = AudioSegment.from_wav(temp_wav)
    seg.export(file_path, format="mp3")
    import os
    os.remove(temp_wav)

