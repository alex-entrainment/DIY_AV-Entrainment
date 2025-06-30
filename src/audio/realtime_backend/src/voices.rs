use std::collections::HashMap;
use serde_json::Value;

use crate::dsp::{sine_wave};
use crate::scheduler::Voice;

pub struct BinauralBeatVoice {
    amp_l: f32,
    amp_r: f32,
    freq_l: f32,
    freq_r: f32,
    phase_l: f32,
    phase_r: f32,
    sample_rate: f32,
    remaining_samples: usize,
}

impl BinauralBeatVoice {
    pub fn new(params: &HashMap<String, Value>, duration: f32, sample_rate: f32) -> Self {
        let amp_l = params.get("ampL").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
        let amp_r = params.get("ampR").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
        let base_freq = params.get("baseFreq").and_then(|v| v.as_f64()).unwrap_or(200.0) as f32;
        let beat_freq = params.get("beatFreq").and_then(|v| v.as_f64()).unwrap_or(4.0) as f32;
        let force_mono = params.get("forceMono").and_then(|v| v.as_bool()).unwrap_or(false);
        let freq_l = base_freq;
        let freq_r = if force_mono { base_freq } else { base_freq + beat_freq };
        let total_samples = (duration * sample_rate) as usize;
        Self {
            amp_l,
            amp_r,
            freq_l,
            freq_r,
            phase_l: 0.0,
            phase_r: 0.0,
            sample_rate,
            remaining_samples: total_samples,
        }
    }
}

impl Voice for BinauralBeatVoice {
    fn process(&mut self, output: &mut [f32]) {
        let channels = 2;
        let frames = output.len() / channels;
        for i in 0..frames {
            if self.remaining_samples == 0 { break; }
            let t = 1.0 / self.sample_rate;
            let sample_l = sine_wave(self.freq_l, 0.0, self.phase_l) * self.amp_l;
            let sample_r = sine_wave(self.freq_r, 0.0, self.phase_r) * self.amp_r;
            self.phase_l += 2.0 * std::f32::consts::PI * self.freq_l * t;
            self.phase_r += 2.0 * std::f32::consts::PI * self.freq_r * t;
            output[i * 2] += sample_l;
            output[i * 2 + 1] += sample_r;
            self.remaining_samples -= 1;
        }
    }

    fn is_finished(&self) -> bool {
        self.remaining_samples == 0
    }
}

use crate::models::{StepData, VoiceData};

pub fn voices_for_step(step: &StepData, sample_rate: f32) -> Vec<Box<dyn Voice>> {
    let mut out: Vec<Box<dyn Voice>> = Vec::new();
    for voice in &step.voices {
        if let Some(v) = create_voice(voice, step.duration as f32, sample_rate) {
            out.push(v);
        }
    }
    out
}

fn create_voice(data: &VoiceData, duration: f32, sample_rate: f32) -> Option<Box<dyn Voice>> {
    match data.synth_function_name.as_str() {
        "binaural_beat" => Some(Box::new(BinauralBeatVoice::new(&data.params, duration, sample_rate))),
        _ => None,
    }
}
