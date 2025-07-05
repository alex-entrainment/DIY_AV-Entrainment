use rand::Rng;
use crate::noise_params::NoiseParams;

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
    let w0 = 2.0 * std::f32::consts::PI * freq / sample_rate;
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
    let t = (phase / (2.0 * std::f32::consts::PI)).rem_euclid(1.0);
    2.0 * (2.0 * (t - (t + 0.5).floor())).abs() - 1.0
}

fn lfo_value(phase: f32, waveform: &str) -> f32 {
    if waveform.eq_ignore_ascii_case("triangle") {
        triangle_wave(phase)
    } else {
        crate::dsp::trig::cos_lut(phase)
    }
}

pub struct StreamingNoise {
    sample_rate: f32,
    lfo_freq: f32,
    sweeps: Vec<(f32, f32)>,
    qs: Vec<f32>,
    cascades: Vec<usize>,
    lfo_phase: f32,
    lfo_phase_offset: f32,
    intra_offset: f32,
    lfo_waveform: String,
    noise_type: String,
    // states for pink noise
    b0: f32,
    b1: f32,
    b2: f32,
    b3: f32,
    b4: f32,
    b5: f32,
    // state for brown noise
    brown: f32,
    rng: rand::rngs::ThreadRng,
    // filter states
    states_main_l: Vec<Vec<BiquadState>>,
    states_extra_l: Vec<Vec<BiquadState>>,
    states_main_r: Vec<Vec<BiquadState>>,
    states_extra_r: Vec<Vec<BiquadState>>,
}

impl StreamingNoise {
    pub fn new(params: &NoiseParams, sample_rate: u32) -> Self {
        let lfo_freq = if params.transition {
            params.start_lfo_freq
        } else if params.lfo_freq != 0.0 {
            params.lfo_freq
        } else {
            1.0 / 12.0
        };
        let sweeps: Vec<(f32, f32)> = if !params.sweeps.is_empty() {
            params
                .sweeps
                .iter()
                .map(|sw| {
                    let min = if sw.start_min > 0.0 { sw.start_min } else { 1000.0 };
                    let max = if sw.start_max > 0.0 {
                        sw.start_max.max(min + 1.0)
                    } else {
                        (min + 1.0).max(min)
                    };
                    (min, max)
                })
                .collect()
        } else {
            vec![(1000.0, 10000.0)]
        };
        let qs: Vec<f32> = if !params.sweeps.is_empty() {
            params
                .sweeps
                .iter()
                .map(|sw| if sw.start_q > 0.0 { sw.start_q } else { 25.0 })
                .collect()
        } else {
            vec![25.0; sweeps.len()]
        };
        let casc: Vec<usize> = if !params.sweeps.is_empty() {
            params
                .sweeps
                .iter()
                .map(|sw| if sw.start_casc > 0 { sw.start_casc } else { 10 })
                .collect()
        } else {
            vec![10usize; sweeps.len()]
        };
        let mk_states = |casc: &Vec<usize>| -> (Vec<Vec<BiquadState>>, Vec<Vec<BiquadState>>) {
            let main: Vec<Vec<BiquadState>> = casc.iter().map(|c| vec![BiquadState::new(); *c]).collect();
            let extra: Vec<Vec<BiquadState>> = casc.iter().map(|c| vec![BiquadState::new(); *c]).collect();
            (main, extra)
        };
        let (states_main_l, states_extra_l) = mk_states(&casc);
        let (states_main_r, states_extra_r) = mk_states(&casc);
        Self {
            sample_rate: sample_rate as f32,
            lfo_freq,
            sweeps,
            qs,
            cascades: casc,
            lfo_phase: 0.0,
            lfo_phase_offset: params.start_lfo_phase_offset_deg.to_radians(),
            intra_offset: params.start_intra_phase_offset_deg.to_radians(),
            lfo_waveform: params.lfo_waveform.clone(),
            noise_type: params.noise_type.clone(),
            b0: 0.0,
            b1: 0.0,
            b2: 0.0,
            b3: 0.0,
            b4: 0.0,
            b5: 0.0,
            brown: 0.0,
            rng: rand::thread_rng(),
            states_main_l,
            states_extra_l,
            states_main_r,
            states_extra_r,
        }
    }

    pub fn skip_samples(&mut self, n: usize) {
        let mut scratch = vec![0.0f32; n * 2];
        self.generate(&mut scratch);
    }

    fn apply_pass(
        mut sample: f32,
        states: &mut [Vec<BiquadState>],
        sweeps: &[(f32, f32)],
        qs: &[f32],
        sample_rate: f32,
        lfo_waveform: &str,
        phase: f32,
    ) -> f32 {
        let lfo = lfo_value(phase, lfo_waveform);
        for (i, sweep) in sweeps.iter().enumerate() {
            let center = (sweep.0 + sweep.1) * 0.5;
            let range = (sweep.1 - sweep.0) * 0.5;
            let freq = center + range * lfo;
            if freq >= sample_rate * 0.49 {
                continue;
            }
            let coeffs = notch_coeffs(freq, qs[i], sample_rate);
            for state in &mut states[i] {
                sample = state.process(sample, &coeffs);
            }
        }
        sample
    }

    fn next_base(&mut self) -> f32 {
        match self.noise_type.to_lowercase().as_str() {
            "brown" => {
                self.brown += self.rng.gen::<f32>() - 0.5;
                self.brown
            }
            _ => {
                let w = self.rng.gen::<f32>();
                self.b0 = 0.99886 * self.b0 + w * 0.0555179;
                self.b1 = 0.99332 * self.b1 + w * 0.0750759;
                self.b2 = 0.96900 * self.b2 + w * 0.1538520;
                self.b3 = 0.86650 * self.b3 + w * 0.3104856;
                self.b4 = 0.55000 * self.b4 + w * 0.5329522;
                self.b5 = -0.7616 * self.b5 - w * 0.0168980;
                (self.b0 + self.b1 + self.b2 + self.b3 + self.b4 + self.b5) * 0.11
            }
        }
    }

    pub fn generate(&mut self, out: &mut [f32]) {
        let frames = out.len() / 2;
        let sweeps = self.sweeps.clone();
        let qs = self.qs.clone();
        let sample_rate = self.sample_rate;
        let lfo_waveform = self.lfo_waveform.clone();
        let lfo_phase_offset = self.lfo_phase_offset;
        let intra_offset = self.intra_offset;

        for i in 0..frames {
            let base = self.next_base();
            let l_phase = self.lfo_phase;
            let r_phase = self.lfo_phase + lfo_phase_offset;
            let mut l = Self::apply_pass(base, &mut self.states_main_l, &sweeps, &qs, sample_rate, &lfo_waveform, l_phase);
            if intra_offset != 0.0 {
                l = Self::apply_pass(l, &mut self.states_extra_l, &sweeps, &qs, sample_rate, &lfo_waveform, l_phase + intra_offset);
            }
            let mut r = Self::apply_pass(base, &mut self.states_main_r, &sweeps, &qs, sample_rate, &lfo_waveform, r_phase);
            if intra_offset != 0.0 {
                r = Self::apply_pass(r, &mut self.states_extra_r, &sweeps, &qs, sample_rate, &lfo_waveform, r_phase + intra_offset);
            }
            out[i * 2] = l;
            out[i * 2 + 1] = r;
            self.lfo_phase += 2.0 * std::f32::consts::PI * self.lfo_freq / sample_rate;
        }
    }
}

