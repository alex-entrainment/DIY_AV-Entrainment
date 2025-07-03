use std::f32::consts::PI;

use super::{generate_brown_noise_samples, generate_pink_noise_samples};

#[derive(Clone, Copy)]
struct BiquadState {
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl BiquadState {
    fn new() -> Self {
        Self { x1: 0.0, x2: 0.0, y1: 0.0, y2: 0.0 }
    }

    fn process(&mut self, x: f32, coeffs: &Coeffs) -> f32 {
        let y = coeffs.b0 * x
            + coeffs.b1 * self.x1
            + coeffs.b2 * self.x2
            - coeffs.a1 * self.y1
            - coeffs.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

#[derive(Clone, Copy)]
struct Coeffs {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

fn notch_coeffs(freq: f32, q: f32, sample_rate: f32) -> Coeffs {
    let w0 = 2.0 * PI * freq / sample_rate;
    let cos_w0 = crate::dsp::trig::cos_lut(w0);
    let alpha = crate::dsp::trig::sin_lut(w0) / (2.0 * q.max(0.001));
    let b0 = 1.0;
    let b1 = -2.0 * cos_w0;
    let b2 = 1.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;
    Coeffs {
        b0: b0 / a0,
        b1: b1 / a0,
        b2: b2 / a0,
        a1: a1 / a0,
        a2: a2 / a0,
    }
}

fn triangle_wave(phase: f32) -> f32 {
    let t = (phase / (2.0 * PI)).rem_euclid(1.0);
    2.0 * (2.0 * (t - (t + 0.5).floor())).abs() - 1.0
}

fn apply_deep_swept_notches_single_phase(
    input: &[f32],
    sample_rate: f32,
    lfo_freq: f32,
    filter_sweeps: &[(f32, f32)],
    notch_q: &[f32],
    cascade_count: &[usize],
    phase_offset: f32,
    lfo_waveform: &str,
) -> Vec<f32> {
    let n = input.len();
    let mut out = vec![0.0f32; n];
    let mut states: Vec<Vec<BiquadState>> = filter_sweeps
        .iter()
        .enumerate()
        .map(|(i, _)| vec![BiquadState::new(); cascade_count[i]])
        .collect();

    for (idx, sample) in input.iter().enumerate() {
        let t = idx as f32 / sample_rate;
        let phase = 2.0 * PI * lfo_freq * t + phase_offset;
        let lfo = if lfo_waveform.eq_ignore_ascii_case("triangle") {
            triangle_wave(phase)
        } else {
            crate::dsp::trig::cos_lut(phase)
        };

        let mut val = *sample;
        for (i, sweep) in filter_sweeps.iter().enumerate() {
            let center = (sweep.0 + sweep.1) * 0.5;
            let range = (sweep.1 - sweep.0) * 0.5;
            let freq = center + range * lfo;
            if freq >= sample_rate * 0.49 {
                continue;
            }
            let coeffs = notch_coeffs(freq, notch_q[i], sample_rate);
            for state in &mut states[i] {
                val = state.process(val, &coeffs);
            }
        }
        out[idx] = val;
    }
    out
}

pub fn apply_deep_swept_notches(
    input: &[f32],
    sample_rate: f32,
    lfo_freq: f32,
    filter_sweeps: &[(f32, f32)],
    notch_q: &[f32],
    cascade_count: &[usize],
    phase_offset: f32,
    extra_phase_offset: f32,
    lfo_waveform: &str,
) -> Vec<f32> {
    let mut out = apply_deep_swept_notches_single_phase(
        input,
        sample_rate,
        lfo_freq,
        filter_sweeps,
        notch_q,
        cascade_count,
        phase_offset,
        lfo_waveform,
    );
    if extra_phase_offset != 0.0 {
        out = apply_deep_swept_notches_single_phase(
            &out,
            sample_rate,
            lfo_freq,
            filter_sweeps,
            notch_q,
            cascade_count,
            phase_offset + extra_phase_offset,
            lfo_waveform,
        );
    }
    out
}

pub fn generate_swept_notch_noise(
    duration_seconds: f32,
    sample_rate: u32,
    lfo_freq: f32,
    filter_sweeps: &[(f32, f32)],
    notch_q: &[f32],
    cascade_count: &[usize],
    lfo_phase_offset_deg: f32,
    intra_phase_offset_deg: f32,
    noise_type: &str,
    lfo_waveform: &str,
) -> Vec<f32> {
    let n_samples = (duration_seconds * sample_rate as f32) as usize;
    let base_noise = match noise_type.to_lowercase().as_str() {
        "brown" => generate_brown_noise_samples(n_samples),
        _ => generate_pink_noise_samples(n_samples),
    };

    let phase_offset = lfo_phase_offset_deg.to_radians();
    let intra_offset = intra_phase_offset_deg.to_radians();

    let left = apply_deep_swept_notches(
        &base_noise,
        sample_rate as f32,
        lfo_freq,
        filter_sweeps,
        notch_q,
        cascade_count,
        0.0,
        intra_offset,
        lfo_waveform,
    );
    let right = apply_deep_swept_notches(
        &base_noise,
        sample_rate as f32,
        lfo_freq,
        filter_sweeps,
        notch_q,
        cascade_count,
        phase_offset,
        intra_offset,
        lfo_waveform,
    );

    let rms_in: f32 = {
        let s: f32 = base_noise.iter().map(|v| v * v).sum();
        (s / base_noise.len() as f32).sqrt()
    };

    let mut stereo: Vec<f32> = Vec::with_capacity(n_samples * 2);
    let mut rms_left = 0.0f32;
    let mut rms_right = 0.0f32;
    for i in 0..n_samples {
        rms_left += left[i] * left[i];
        rms_right += right[i] * right[i];
    }
    rms_left = (rms_left / n_samples as f32).sqrt();
    rms_right = (rms_right / n_samples as f32).sqrt();

    let gain_l = if rms_left > 1e-8 { rms_in / rms_left } else { 1.0 };
    let gain_r = if rms_right > 1e-8 { rms_in / rms_right } else { 1.0 };

    let mut max_val = 0.0f32;
    for i in 0..n_samples {
        let l = left[i] * gain_l;
        let r = right[i] * gain_r;
        stereo.push(l);
        stereo.push(r);
        max_val = max_val.max(l.abs().max(r.abs()));
    }

    if max_val > 0.95 {
        for v in &mut stereo {
            *v = v.clamp(-0.95, 0.95);
        }
    } else if max_val > 0.0 {
        let norm = 0.95 / max_val;
        for v in &mut stereo {
            *v *= norm;
        }
    }

    stereo
}

