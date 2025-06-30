use rand::Rng;

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
    (2.0 * std::f32::consts::PI * freq * t + phase).sin()
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
    let left = angle.cos() * signal;
    let right = angle.sin() * signal;
    (left, right)
}
