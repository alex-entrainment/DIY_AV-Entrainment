import numpy as np
import librosa
import soundfile as sf
import scipy.signal


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
    """
    Base voice class. Subclasses should override `generate_samples` with
    their specific audio synthesis logic. This version provides:

      1) Timeline building: total duration, sample count, and node start times.
      2) Automatic creation of interpolation arrays for base_freq, beat_freq,
         volume_left, volume_right. This avoids Python-level per-sample loops.
    """
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

        # Pre-build arrays for interpolation
        self.node_times = np.array(self.start_times, dtype=np.float64)
        self.base_freq_values = np.array([n.base_freq for n in self.nodes], dtype=np.float64)
        self.beat_freq_values = np.array([n.beat_freq for n in self.nodes], dtype=np.float64)
        self.vol_left_values = np.array([n.volume_left for n in self.nodes], dtype=np.float64)
        self.vol_right_values = np.array([n.volume_right for n in self.nodes], dtype=np.float64)

    def _get_param_arrays(self):
        """
        Returns:
          t_array          : time array [0..total_duration)
          base_freq_array  : base freq interpolated across nodes
          beat_freq_array  : beat freq interpolated across nodes
          vol_left_array   : volume left interpolated across nodes
          vol_right_array  : volume right interpolated across nodes
        """
        if self.num_samples == 0:
            # Edge case: no samples
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )

        t_array = np.arange(self.num_samples, dtype=np.float64) / self.sample_rate

        # Use np.interp to do linear interpolation across node boundaries
        base_freq_array = np.interp(
            t_array, self.node_times, self.base_freq_values
        )
        beat_freq_array = np.interp(
            t_array, self.node_times, self.beat_freq_values
        )
        vol_left_array = np.interp(
            t_array, self.node_times, self.vol_left_values
        )
        vol_right_array = np.interp(
            t_array, self.node_times, self.vol_right_values
        )

        return (
            t_array,
            base_freq_array,
            beat_freq_array,
            vol_left_array,
            vol_right_array,
        )

    def generate_samples(self):
        """
        Subclasses override this method to produce an audio buffer
        as a NumPy array of shape (num_samples, 2), float32.
        """
        raise NotImplementedError()


# -----------------------------------------------------------
# Spatial Angle Modulation (SAM) Voice
# -----------------------------------------------------------

class SpatialAngleModVoice(Voice):
    """
    A general-purpose Spatial Angle Modulation (SAM) voice that:
      - Uses interpolated base_freq and volumes from node data.
      - Each ear has its own angle motion: arc_freq_left, arc_freq_right, etc.
      - arc_function can be 'sin' or 'triangle'.
    """

    def __init__(self, nodes, sample_rate=44100,
                 arc_freq_left=10.0,
                 arc_freq_right=10.0,
                 arc_center_left=0.0,
                 arc_center_right=0.0,
                 arc_peak_left=np.pi/2,
                 arc_peak_right=np.pi/2,
                 phase_offset_left=0.0,
                 phase_offset_right=0.0,
                 arc_function='sin'):
        super().__init__(nodes, sample_rate)
        self.arc_freq_left = float(arc_freq_left)
        self.arc_freq_right = float(arc_freq_right)
        self.arc_center_left = float(arc_center_left)
        self.arc_center_right = float(arc_center_right)
        self.arc_peak_left = float(arc_peak_left)
        self.arc_peak_right = float(arc_peak_right)
        self.phase_offset_left = float(phase_offset_left)
        self.phase_offset_right = float(phase_offset_right)
        self.arc_function = arc_function.lower().strip()

    def _arc_value_vector(self, t, freq, center, peak, phase, shape):
        """
        Return the instantaneous angle offset for array t in radians:
          angle = center + peak * wave(2π * freq * t + phase).
        """
        omega_t = 2.0 * np.pi * freq * t + phase
        if shape == 'triangle':
            # Triangular wave in range [-1, +1], width=0.5 => symmetrical triangle
            val = scipy.signal.sawtooth(omega_t, width=0.5)
        else:
            # default 'sin'
            val = np.sin(omega_t)
        return center + peak * val

    def generate_samples(self):
        # 1) Build param arrays
        t_array, base_freq_array, _, vol_left_array, vol_right_array = self._get_param_arrays()
        num_samples = t_array.size
        if num_samples == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # 2) Compute angles for left/right
        angle_left_array = self._arc_value_vector(
            t_array,
            self.arc_freq_left,
            self.arc_center_left,
            self.arc_peak_left,
            self.phase_offset_left,
            self.arc_function
        )
        angle_right_array = self._arc_value_vector(
            t_array,
            self.arc_freq_right,
            self.arc_center_right,
            self.arc_peak_right,
            self.phase_offset_right,
            self.arc_function
        )

        # 3) Compute the final waveforms
        # left ear => sin(2π * base_freq_array * t + angle_left_array)
        # right ear => sin(2π * base_freq_array * t + angle_right_array)
        phase_left = 2.0 * np.pi * base_freq_array * t_array + angle_left_array
        phase_right = 2.0 * np.pi * base_freq_array * t_array + angle_right_array

        s_left = np.sin(phase_left) * vol_left_array
        s_right = np.sin(phase_right) * vol_right_array

        audio = np.column_stack((s_left, s_right)).astype(np.float32)
        return audio


# -----------------------------------------------------------
# Binaural Beats
# -----------------------------------------------------------

class BinauralBeatVoice(Voice):
    def generate_samples(self):
        # Vectorized approach for Binaural Beats
        t_array, base_freq_array, beat_freq_array, vol_left_array, vol_right_array = self._get_param_arrays()
        num_samples = t_array.size
        if num_samples == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # left_freq = base - (beat/2), right_freq = base + (beat/2)
        left_freq_array = base_freq_array - beat_freq_array / 2.0
        right_freq_array = base_freq_array + beat_freq_array / 2.0

        phase_left = 2.0 * np.pi * left_freq_array * t_array
        phase_right = 2.0 * np.pi * right_freq_array * t_array

        s_left = np.sin(phase_left) * vol_left_array
        s_right = np.sin(phase_right) * vol_right_array

        audio = np.column_stack((s_left, s_right)).astype(np.float32)
        return audio


# -----------------------------------------------------------
# Isochronic Trapezoid Envelope (Vectorized)
# -----------------------------------------------------------

def trapezoid_envelope_vectorized(t_in_cycle, cycle_len, ramp_percent, gap_percent):
    """
    Vectorized trapezoidal envelope. Inputs are arrays of the same shape.
    Each element i satisfies:
      - 0 <= t_in_cycle[i] < cycle_len[i]
      - cycle_len[i] > 0
    We'll break the cycle into: ramp_up + stable + ramp_down + gap.
    Return envelope in [0..1].
    If cycle_len <= 0, envelope is 0. (Handled via mask.)
    """
    # Allocate the envelope
    env = np.zeros_like(t_in_cycle, dtype=np.float64)

    # Protect against invalid or zero cycle lengths
    valid_mask = cycle_len > 0
    if not np.any(valid_mask):
        return env  # all zeros

    # The portion used for the audible pulse
    audible_len = (1.0 - gap_percent) * cycle_len
    # The total ramp portion
    ramp_total = audible_len * ramp_percent * 2.0
    # The stable portion is what's left
    stable_len = audible_len - ramp_total

    # We also need to handle stable_len < 0
    # We'll do it in piecewise logic (some cycles might differ from others)
    # So let's define piecewise approach:

    # For each element:
    #   if t_in_cycle[i] >= audible_len[i], env=0
    #   else if t_in_cycle[i] < ramp_up_len[i], env = scale up
    #   else if t_in_cycle[i] < stable_end[i], env = 1
    #   else if t_in_cycle[i] < audible_len[i], env = ramp down

    # Build arrays
    ramp_up_len = ramp_total / 2.0
    # If stable_len < 0 => stable portion is 0, so ramp_up_len & ramp_down_len = audible_len/2
    stable_len_clamped = np.maximum(stable_len, 0.0)
    ramp_total_clamped = audible_len - stable_len_clamped
    ramp_up_len_clamped = ramp_total_clamped / 2.0

    stable_end = ramp_up_len_clamped + stable_len_clamped

    # For any t_in_cycle >= audible_len, env = 0
    in_gap_mask = (t_in_cycle >= audible_len)

    # Ramp up: t_in_cycle < ramp_up_len_clamped
    ramp_up_mask = (t_in_cycle < ramp_up_len_clamped) & (~in_gap_mask) & valid_mask
    # ramp down: t_in_cycle >= stable_end & t_in_cycle < audible_len
    ramp_down_mask = (t_in_cycle >= stable_end) & (t_in_cycle < audible_len) & valid_mask
    # stable: between ramp up end and stable_end
    stable_mask = (t_in_cycle >= ramp_up_len_clamped) & (t_in_cycle < stable_end) & (~in_gap_mask) & valid_mask

    # Ramp up
    env[ramp_up_mask] = (
        t_in_cycle[ramp_up_mask] / ramp_up_len_clamped[ramp_up_mask]
    )
    # Stable
    env[stable_mask] = 1.0
    # Ramp down
    # We compute how far we are into the ramp down
    time_into_down = (t_in_cycle[ramp_down_mask] - stable_end[ramp_down_mask])
    down_len = (audible_len[ramp_down_mask] - stable_end[ramp_down_mask])
    # envelope is 1 - (time_into_down / down_len)
    env[ramp_down_mask] = 1.0 - (time_into_down / down_len)

    # Any invalid cycle_len or in the gap => env=0 by default
    return env


class IsochronicVoice(Voice):
    """
    A trapezoidal isochronic generator: each beat cycle has
    ramp up, stable top, ramp down, and gap. We multiply that amplitude
    by a carrier sine wave at base_freq. This is vectorized, but note that
    if beat_freq changes continuously, the notion of 'cycle' also continuously
    changes. We handle that by computing t_in_cycle = t % (1/beat_array),
    which is approximate if beat_array is not constant. 
    """
    def __init__(self, nodes, sample_rate=44100,
                 ramp_percent=0.2, gap_percent=0.15, amplitude=1.0):
        super().__init__(nodes, sample_rate)
        self.ramp_percent = float(ramp_percent)
        self.gap_percent = float(gap_percent)
        self.amplitude = float(amplitude)

    def generate_samples(self):
        t_array, base_freq_array, beat_freq_array, vol_left_array, vol_right_array = self._get_param_arrays()
        num_samples = t_array.size
        if num_samples == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # cycle_len_array = 1 / beat_freq_array
        # but watch out for zero or negative beat_freq
        cycle_len_array = np.zeros_like(beat_freq_array, dtype=np.float64)
        valid_beat_mask = (beat_freq_array > 0)
        cycle_len_array[valid_beat_mask] = 1.0 / beat_freq_array[valid_beat_mask]

        # t_in_cycle = t % cycle_len_array, but that doesn't work well if cycle_len is changing each sample
        # We'll do it anyway as an approximation:
        t_in_cycle = np.mod(t_array, cycle_len_array, where=valid_beat_mask, out=np.zeros_like(t_array))

        env = trapezoid_envelope_vectorized(t_in_cycle, cycle_len_array,
                                            self.ramp_percent, self.gap_percent)

        # Multiply by a carrier wave at base_freq
        carrier = np.sin(2.0 * np.pi * base_freq_array * t_array) * env * self.amplitude

        left = carrier * vol_left_array
        right = carrier * vol_right_array
        audio = np.column_stack((left, right)).astype(np.float32)
        return audio


class AltIsochronicVoice(Voice):
    """
    Same trapezoidal approach, but we alternate entire cycles
    between left and right channels. We do so with a loop because
    truly vectorizing "stateful toggling" requires more advanced
    logic (tracking the sum of cycle lengths).
    
    If you want a fully vectorized solution, we need to do a cumulative
    sum of cycle_len_array, see when t_array crosses each cycle boundary,
    etc. This is more complicated if beat_freq changes over time.

    Below we at least vectorize the envelope calculations inside each cycle,
    but we still do a for-loop over cycles. If beat_freq changes at node
    boundaries only (not constantly changing each sample), you could break
    it up by node to reduce overhead.
    """
    def __init__(self, nodes, sample_rate=44100,
                 ramp_percent=0.2, gap_percent=0.15, amplitude=1.0):
        super().__init__(nodes, sample_rate)
        self.ramp_percent = float(ramp_percent)
        self.gap_percent = float(gap_percent)
        self.amplitude = float(amplitude)

    def generate_samples(self):
        # We do a single pass from t=0 to t=total_duration, toggling channels each cycle.
        # If beat is changing, it's tricky to interpret "one cycle is 1.0/beat" at each moment.
        # We'll treat the FIRST sample's beat as the cycle length, then recalc at each cycle boundary.
        # If you want a different approach, we'd need more design input.
        num_samples = self.num_samples
        if num_samples == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # Interpolation arrays
        t_array, base_freq_array, beat_freq_array, vol_left_array, vol_right_array = self._get_param_arrays()

        audio = np.zeros((num_samples, 2), dtype=np.float32)

        current_left = True
        i_start = 0
        while i_start < num_samples:
            # read beat freq at i_start
            beat = beat_freq_array[i_start]
            if beat <= 0:
                # skip or treat as silence
                i_start += 1
                continue
            cycle_duration = 1.0 / beat
            if cycle_duration <= 0:
                i_start += 1
                continue

            # figure out how many samples remain in this cycle
            # we want all samples from t_array[i_start] up to t_array[i_start] + cycle_duration
            t0 = t_array[i_start]
            t_end = t0 + cycle_duration

            # We can find the index range:
            # we want all i in [i_start..i_end) s.t. t_array[i] < t_end
            # do a search
            # if t_end > total_duration, i_end = num_samples
            # else find i_end = largest i where t_array[i] < t_end
            # We'll use np.searchsorted:
            i_end = np.searchsorted(t_array, t_end, side='left')
            if i_end > num_samples:
                i_end = num_samples

            # Now we have a subarray from i_start to i_end
            sub_indices = slice(i_start, i_end)
            sub_t = t_array[sub_indices]
            sub_base = base_freq_array[sub_indices]
            sub_vol_left = vol_left_array[sub_indices]
            sub_vol_right = vol_right_array[sub_indices]

            # Within this subarray, we define t_in_cycle = sub_t - t0
            sub_t_in_cycle = sub_t - t0
            # Vector envelope
            env = trapezoid_envelope_vectorized(
                sub_t_in_cycle, 
                np.full_like(sub_t_in_cycle, cycle_duration),
                self.ramp_percent,
                self.gap_percent
            )
            carrier = np.sin(2.0 * np.pi * sub_base * sub_t) * env * self.amplitude

            if current_left:
                audio[sub_indices, 0] = carrier * sub_vol_left
                audio[sub_indices, 1] = 0.0
            else:
                audio[sub_indices, 0] = 0.0
                audio[sub_indices, 1] = carrier * sub_vol_right

            current_left = not current_left
            i_start = i_end

        return audio


class AltIsochronic2Voice(Voice):
    """
    Half-cycle on left, half-cycle on right for each beat cycle,
    with a trapezoid envelope for each half. This also has a
    stateful "which half are we in?" approach, so we do a loop
    over each cycle (similar to AltIsochronicVoice).
    """
    def __init__(self, nodes, sample_rate=44100,
                 ramp_percent=0.2, gap_percent=0.15, amplitude=1.0):
        super().__init__(nodes, sample_rate)
        self.ramp_percent = float(ramp_percent)
        self.gap_percent = float(gap_percent)
        self.amplitude = float(amplitude)

    def generate_samples(self):
        num_samples = self.num_samples
        if num_samples == 0:
            return np.zeros((0, 2), dtype=np.float32)

        t_array, base_freq_array, beat_freq_array, vol_left_array, vol_right_array = self._get_param_arrays()
        audio = np.zeros((num_samples, 2), dtype=np.float32)

        i_start = 0
        while i_start < num_samples:
            beat = beat_freq_array[i_start]
            if beat <= 0:
                i_start += 1
                continue
            cycle_duration = 1.0 / beat
            if cycle_duration <= 0:
                i_start += 1
                continue

            t0 = t_array[i_start]
            t_end = t0 + cycle_duration
            i_end = np.searchsorted(t_array, t_end, side='left')
            if i_end > num_samples:
                i_end = num_samples

            sub_indices = slice(i_start, i_end)
            sub_t = t_array[sub_indices]
            sub_base = base_freq_array[sub_indices]
            sub_vol_left = vol_left_array[sub_indices]
            sub_vol_right = vol_right_array[sub_indices]

            # half cycle
            half_duration = cycle_duration / 2.0

            # For each sample in sub_t, check if it's in the left half or right half:
            # local time
            sub_t_in_cycle = sub_t - t0
            left_mask = (sub_t_in_cycle < half_duration)
            right_mask = ~left_mask

            # Envelope for left half
            t_in_left = sub_t_in_cycle[left_mask]
            env_left = trapezoid_envelope_vectorized(
                t_in_left,
                np.full_like(t_in_left, half_duration),
                self.ramp_percent,
                self.gap_percent
            )
            carrier_left = np.sin(2.0 * np.pi * sub_base[left_mask] * sub_t[left_mask]) * env_left * self.amplitude
            audio[sub_indices][left_mask, 0] = carrier_left * sub_vol_left[left_mask]
            audio[sub_indices][left_mask, 1] = 0.0

            # Envelope for right half
            t_in_right = sub_t_in_cycle[right_mask] - half_duration
            env_right = trapezoid_envelope_vectorized(
                t_in_right,
                np.full_like(t_in_right, half_duration),
                self.ramp_percent,
                self.gap_percent
            )
            carrier_right = np.sin(2.0 * np.pi * sub_base[right_mask] * sub_t[right_mask]) * env_right * self.amplitude
            audio[sub_indices][right_mask, 0] = 0.0
            audio[sub_indices][right_mask, 1] = carrier_right * sub_vol_right[right_mask]

            i_start = i_end

        return audio


# -----------------------------------------------------------
# Pink Noise Voice
# -----------------------------------------------------------

class PinkNoiseVoice(Voice):
    """
    We can vectorize pink noise generation too, at least partly.
    The standard "Voss-McCartney" or "filtered white" approach can be used.
    Below is a simple IIR filter approach: pink[i] = alpha*pink[i-1] + (1-alpha)*white[i].
    We still do a loop inside NumPy arrays. Alternatively, we can do it with np.cumsum or
    an IIR filter from scipy.signal.lfilter. We'll do a quick partial approach.
    """
    def generate_samples(self):
        t_array, _, _, vol_left_array, vol_right_array = self._get_param_arrays()
        num_samples = t_array.size
        if num_samples == 0:
            return np.zeros((0, 2), dtype=np.float32)

        audio = np.zeros((num_samples, 2), dtype=np.float32)
        # Generate white noise
        white = np.random.normal(0, 1, num_samples).astype(np.float32)
        # We do a simple 1-pole filter to approximate pink noise
        alpha = 0.98
        pink = np.zeros(num_samples, dtype=np.float32)
        pink[0] = white[0]
        for i in range(1, num_samples):
            pink[i] = alpha * pink[i-1] + (1.0 - alpha) * white[i]

        # Now multiply by volumes
        audio[:, 0] = pink * vol_left_array * 0.1
        audio[:, 1] = pink * vol_right_array * 0.1
        return audio


# -----------------------------------------------------------
# External Audio Voice
# -----------------------------------------------------------

class ExternalAudioVoice(Voice):
    """
    Loads external audio from file (mono or stereo),
    then multiplies it by the volume envelopes from the nodes.
    """
    def __init__(self, nodes, file_path, sample_rate=44100):
        super().__init__(nodes, sample_rate)
        self.file_path = file_path
        data, sr = librosa.load(self.file_path, sr=sample_rate, mono=False)
        if data.ndim == 1:
            # shape = (samples,)
            data = data[np.newaxis, :]  # shape = (1, samples)
        self.ext_audio = data
        self.ext_length = self.ext_audio.shape[1]

    def generate_samples(self):
        t_array, _, _, vol_left_array, vol_right_array = self._get_param_arrays()
        num_samples = t_array.size
        if num_samples == 0:
            return np.zeros((0, 2), dtype=np.float32)

        audio = np.zeros((num_samples, 2), dtype=np.float32)

        # Build an index array: 0..num_samples-1
        idx_array = np.arange(num_samples)
        # clamp to the ext_length
        idx_array = np.minimum(idx_array, self.ext_length - 1)

        if self.ext_audio.shape[0] == 1:
            # Mono: use same signal for L and R
            mono_samples = self.ext_audio[0][idx_array]
            audio[:, 0] = mono_samples * vol_left_array
            audio[:, 1] = mono_samples * vol_right_array
        else:
            # Stereo
            left_samples = self.ext_audio[0][idx_array]
            right_samples = self.ext_audio[1][idx_array]
            audio[:, 0] = left_samples * vol_left_array
            audio[:, 1] = right_samples * vol_right_array

        return audio


# -----------------------------------------------------------
# Track-Wide Generation
# -----------------------------------------------------------

def generate_track_audio(voices, sample_rate=44100):
    """
    Given a list of Voice objects, generate each voice's samples,
    mix them together, and normalize if needed.
    """
    if not voices:
        return np.zeros((0, 2), dtype=np.float32)

    track_length_seconds = max(v.total_duration for v in voices)
    total_samples = int(track_length_seconds * sample_rate)
    mixed = np.zeros((total_samples, 2), dtype=np.float32)

    for v in voices:
        buf = v.generate_samples()
        length = buf.shape[0]
        if length > 0:
            mixed[:length, :] += buf

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
    import os
    temp_wav = file_path + ".temp.wav"
    sf.write(temp_wav, audio_data, sample_rate, format='WAV')
    seg = AudioSegment.from_wav(temp_wav)
    seg.export(file_path, format="mp3")
    os.remove(temp_wav)

