use rand::Rng;

pub mod noise_flanger;
pub mod trig;

pub fn generate_pink_noise_samples(n_samples: usize) -> Vec<f32> {
    // Simple approximation of pink noise using Voss-McCartney algorithm
    let mut rng = rand::thread_rng();
    let mut b0 = 0.0f32;
    let mut b1 = 0.0f32;
    let mut b2 = 0.0f32;
    let mut b3 = 0.0f32;
    let mut b4 = 0.0f32;
    let mut b5 = 0.0f32;
    let mut out = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let w = rng.gen::<f32>();
        b0 = 0.99886 * b0 + w * 0.0555179;
        b1 = 0.99332 * b1 + w * 0.0750759;
        b2 = 0.96900 * b2 + w * 0.1538520;
        b3 = 0.86650 * b3 + w * 0.3104856;
        b4 = 0.55000 * b4 + w * 0.5329522;
        b5 = -0.7616 * b5 - w * 0.0168980;
        out.push((b0 + b1 + b2 + b3 + b4 + b5) * 0.11);
    }
    out
}

pub fn generate_brown_noise_samples(n_samples: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut cumulative = 0.0f32;
    let mut out = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        cumulative += rng.gen::<f32>() - 0.5;
        out.push(cumulative);
    }
    let max_abs = out.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
    if max_abs > 0.0 {
        for v in &mut out {
            *v /= max_abs;
        }
    }
    out
}

pub fn sine_wave(freq: f32, t: f32, phase: f32) -> f32 {
    crate::dsp::trig::sin_lut(2.0 * std::f32::consts::PI * freq * t + phase)
}

pub fn adsr_envelope(t: &[f32], attack: f32, decay: f32, sustain_level: f32, release: f32) -> Vec<f32> {
    let total_samples = t.len();
    if total_samples == 0 {
        return Vec::new();
    }
    let duration = t[total_samples - 1] - t[0] + if total_samples > 1 { t[1] - t[0] } else { 0.0 };
    let sr = total_samples as f32 / duration;
    let attack_samples = (attack * sr) as usize;
    let decay_samples = (decay * sr) as usize;
    let release_samples = (release * sr) as usize;
    let sustain_samples = total_samples.saturating_sub(attack_samples + decay_samples + release_samples);
    let mut env = Vec::with_capacity(total_samples);
    for i in 0..attack_samples {
        env.push(i as f32 / attack_samples as f32);
    }
    for i in 0..decay_samples {
        let level = 1.0 - (1.0 - sustain_level) * (i as f32 / decay_samples as f32);
        env.push(level);
    }
    for _ in 0..sustain_samples {
        env.push(sustain_level);
    }
    for i in 0..release_samples {
        let level = sustain_level * (1.0 - (i as f32 / release_samples as f32));
        env.push(level);
    }
    env.truncate(total_samples);
    env
}

pub fn pan2(signal: f32, pan: f32) -> (f32, f32) {
    let pan = pan.clamp(-1.0, 1.0);
    let angle = (pan + 1.0) * std::f32::consts::FRAC_PI_4;
    let left = crate::dsp::trig::cos_lut(angle) * signal;
    let right = crate::dsp::trig::sin_lut(angle) * signal;
    (left, right)
}

pub fn trapezoid_envelope(
    t_in_cycle: f32,
    cycle_len: f32,
    ramp_percent: f32,
    gap_percent: f32,
) -> f32 {
    if cycle_len <= 0.0 {
        return 0.0;
    }

    let audible_len = (1.0 - gap_percent).clamp(0.0, 1.0) * cycle_len;
    let ramp_total = (audible_len * ramp_percent * 2.0).clamp(0.0, audible_len);
    let stable_len = audible_len - ramp_total;
    let ramp_up_len = ramp_total / 2.0;
    let stable_end = ramp_up_len + stable_len;

    if t_in_cycle >= audible_len {
        0.0
    } else if t_in_cycle < ramp_up_len {
        if ramp_up_len > 0.0 {
            t_in_cycle / ramp_up_len
        } else {
            0.0
        }
    } else if t_in_cycle >= stable_end {
        if ramp_up_len > 0.0 {
            1.0 - (t_in_cycle - stable_end) / ramp_up_len
        } else {
            0.0
        }
    } else {
        1.0
    }
}

/// Sine wave with adjustable up/down symmetry.
/// `phase_fraction` should be in the range `[0, 1)` representing the
/// position within one cycle. `skew` ranges from -1.0 (long upswing) to
/// 1.0 (short upswing). A value of 0.0 yields a standard sine shape.
pub fn skewed_sine_phase(phase_fraction: f32, skew: f32) -> f32 {
    let mut frac = 0.5 + 0.5 * skew;
    if frac <= 0.0 {
        frac = 1e-9;
    }
    if frac >= 1.0 {
        frac = 1.0 - 1e-9;
    }

    if phase_fraction < frac {
        let local = phase_fraction / frac;
        (std::f32::consts::PI * local).sin()
    } else {
        let local = (phase_fraction - frac) / (1.0 - frac);
        (std::f32::consts::PI * (1.0 + local)).sin()
    }
}

/// Triangle wave with adjustable up/down symmetry.
/// `phase_fraction` should be in the range `[0, 1)` representing the
/// position within one cycle. `skew` ranges from -1.0 (long upswing) to
/// 1.0 (short upswing). A value of 0.0 yields a standard triangle shape.
pub fn skewed_triangle_phase(phase_fraction: f32, skew: f32) -> f32 {
    let mut frac = 0.5 + 0.5 * skew;
    if frac <= 0.0 {
        frac = 1e-9;
    }
    if frac >= 1.0 {
        frac = 1.0 - 1e-9;
    }

    if phase_fraction < frac {
        let local = phase_fraction / frac;
        -1.0 + 2.0 * local
    } else {
        let local = (phase_fraction - frac) / (1.0 - frac);
        1.0 - 2.0 * local
    }
}

pub fn sine_wave_varying(freq_array: &[f32], t: &[f32], _sample_rate: f32) -> Vec<f32> {
    if t.len() == 0 || freq_array.len() != t.len() {
        return Vec::new();
    }
    let n = t.len();
    let mut out = Vec::with_capacity(n);
    let mut phase = 2.0 * std::f32::consts::PI * freq_array[0].max(1e-9) * t[0];
    out.push(sine_wave(0.0, 0.0, phase));
    for i in 1..n {
        let dt = t[i] - t[i - 1];
        phase += 2.0 * std::f32::consts::PI * freq_array[i].max(1e-9) * dt;
        out.push(sine_wave(0.0, 0.0, phase));
    }
    out
}

pub fn linen_envelope(t: &[f32], attack: f32, release: f32) -> Vec<f32> {
    let total_samples = t.len();
    if total_samples == 0 {
        return Vec::new();
    }
    let duration = t[total_samples - 1] - t[0] + if total_samples > 1 { t[1] - t[0] } else { 0.0 };
    let sr = if duration > 0.0 { total_samples as f32 / duration } else { 44100.0 };

    let mut attack_s = (attack.max(0.0) * sr) as usize;
    let mut release_s = (release.max(0.0) * sr) as usize;
    if attack_s + release_s > total_samples && attack_s + release_s > 0 {
        let scale = total_samples as f32 / (attack_s + release_s) as f32;
        attack_s = (attack_s as f32 * scale) as usize;
        release_s = total_samples.saturating_sub(attack_s);
    }
    let sustain_s = total_samples.saturating_sub(attack_s + release_s);

    let mut env = Vec::with_capacity(total_samples);
    for i in 0..attack_s {
        env.push(i as f32 / attack_s as f32);
    }
    for _ in 0..sustain_s {
        env.push(1.0);
    }
    for i in 0..release_s {
        env.push(1.0 - (i as f32 / release_s as f32));
    }
    env.truncate(total_samples);
    env
}

pub fn create_linear_fade_envelope(
    total_duration: f32,
    sample_rate: u32,
    fade_duration: f32,
    start_amp: f32,
    end_amp: f32,
    fade_type: &str,
) -> Vec<f32> {
    let total_samples = (total_duration * sample_rate as f32) as usize;
    if total_samples == 0 {
        return Vec::new();
    }
    let fade_samples = (fade_duration * sample_rate as f32).min(total_samples as f32) as usize;
    let mut env = vec![1.0f32; total_samples];
    if fade_samples == 0 {
        let fill = if fade_type == "in" { end_amp } else { start_amp };
        env.fill(fill);
        return env;
    }

    match fade_type {
        "in" => {
            for i in 0..fade_samples {
                let alpha = i as f32 / fade_samples as f32;
                env[i] = start_amp + alpha * (end_amp - start_amp);
            }
            for v in env.iter_mut().skip(fade_samples) {
                *v = end_amp;
            }
        }
        "out" => {
            let sustain = total_samples.saturating_sub(fade_samples);
            for i in 0..sustain {
                env[i] = start_amp;
            }
            for i in 0..fade_samples {
                let alpha = i as f32 / fade_samples as f32;
                env[sustain + i] = start_amp + alpha * (end_amp - start_amp);
            }
        }
        _ => {}
    }
    env
}

pub fn calculate_transition_alpha(
    total_duration: f32,
    sample_rate: f32,
    initial_offset: f32,
    post_offset: f32,
    curve: &str,
) -> Vec<f32> {
    let n = (total_duration * sample_rate) as usize;
    if n == 0 {
        return Vec::new();
    }
    let mut alpha = Vec::with_capacity(n);
    let dt = 1.0 / sample_rate;
    for i in 0..n {
        let t = i as f32 * dt;
        let val = if t < initial_offset {
            0.0
        } else if t > total_duration - post_offset {
            1.0
        } else {
            let span = total_duration - initial_offset - post_offset;
            if span > 0.0 {
                (t - initial_offset) / span
            } else {
                1.0
            }
        };
        alpha.push(val);
    }

    match curve {
        "logarithmic" => {
            for a in &mut alpha {
                *a = 1.0 - (1.0 - *a).powi(2);
            }
        }
        "exponential" => {
            for a in &mut alpha {
                *a = a.powi(2);
            }
        }
        _ => {}
    }

    alpha
}

use crate::models::VolumeEnvelope;

/// Build a volume envelope vector from a `VolumeEnvelope` description.
/// If the type is unknown or parameters are missing, a flat envelope is returned.
pub fn build_volume_envelope(
    env: &VolumeEnvelope,
    duration: f32,
    sample_rate: u32,
) -> Vec<f32> {
    let total_samples = (duration * sample_rate as f32) as usize;
    if total_samples == 0 {
        return Vec::new();
    }

    match env.envelope_type.as_str() {
        "linear_fade" => {
            let fade_duration = env
                .params
                .get("fade_duration")
                .copied()
                .unwrap_or(0.0) as f32;
            let start_amp = env
                .params
                .get("start_amp")
                .copied()
                .unwrap_or(0.0) as f32;
            let end_amp = env
                .params
                .get("end_amp")
                .copied()
                .unwrap_or(1.0) as f32;
            let fade_type = "in"; // only 'in' currently used
            create_linear_fade_envelope(duration, sample_rate, fade_duration, start_amp, end_amp, fade_type)
        }
        "adsr" => {
            let attack = env.params.get("attack").copied().unwrap_or(0.01) as f32;
            let decay = env.params.get("decay").copied().unwrap_or(0.1) as f32;
            let sustain_level = env
                .params
                .get("sustain_level")
                .copied()
                .unwrap_or(0.8) as f32;
            let release = env.params.get("release").copied().unwrap_or(0.1) as f32;
            let dt = 1.0 / sample_rate as f32;
            let t: Vec<f32> = (0..total_samples).map(|i| i as f32 * dt).collect();
            adsr_envelope(&t, attack, decay, sustain_level, release)
        }
        "linen" => {
            let attack = env.params.get("attack").copied().unwrap_or(0.01) as f32;
            let release = env.params.get("release").copied().unwrap_or(0.1) as f32;
            let dt = 1.0 / sample_rate as f32;
            let t: Vec<f32> = (0..total_samples).map(|i| i as f32 * dt).collect();
            linen_envelope(&t, attack, release)
        }
        _ => vec![1.0; total_samples],
    }
}
