use serde_json::Value;
use std::collections::HashMap;

use crate::dsp::sine_wave;
use crate::scheduler::Voice;

fn get_f32(params: &HashMap<String, Value>, key: &str, default: f32) -> f32 {
    params
        .get(key)
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(default)
}

fn get_bool(params: &HashMap<String, Value>, key: &str, default: bool) -> bool {
    params.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
}

#[derive(Clone, Copy)]
enum TransitionCurve {
    Linear,
    Logarithmic,
    Exponential,
}

impl TransitionCurve {
    fn from_str(s: &str) -> Self {
        match s {
            "logarithmic" => TransitionCurve::Logarithmic,
            "exponential" => TransitionCurve::Exponential,
            _ => TransitionCurve::Linear,
        }
    }

    fn apply(self, alpha: f32) -> f32 {
        match self {
            TransitionCurve::Linear => alpha,
            TransitionCurve::Logarithmic => 1.0 - (1.0 - alpha).powi(2),
            TransitionCurve::Exponential => alpha.powi(2),
        }
    }
}

pub struct BinauralBeatVoice {
    amp_l: f32,
    amp_r: f32,
    base_freq: f32,
    beat_freq: f32,
    force_mono: bool,
    start_phase_l: f32,
    start_phase_r: f32,
    amp_osc_depth_l: f32,
    amp_osc_freq_l: f32,
    amp_osc_depth_r: f32,
    amp_osc_freq_r: f32,
    freq_osc_range_l: f32,
    freq_osc_freq_l: f32,
    freq_osc_range_r: f32,
    freq_osc_freq_r: f32,
    amp_osc_phase_offset_l: f32,
    amp_osc_phase_offset_r: f32,
    phase_osc_freq: f32,
    phase_osc_range: f32,
    phase_l: f32,
    phase_r: f32,
    sample_rate: f32,
    remaining_samples: usize,
    elapsed: f32,
}

pub struct BinauralBeatTransitionVoice {
    start_amp_l: f32,
    end_amp_l: f32,
    start_amp_r: f32,
    end_amp_r: f32,
    start_base_freq: f32,
    end_base_freq: f32,
    start_beat_freq: f32,
    end_beat_freq: f32,
    start_force_mono: bool,
    end_force_mono: bool,
    start_start_phase_l: f32,
    end_start_phase_l: f32,
    start_start_phase_r: f32,
    end_start_phase_r: f32,
    start_phase_osc_freq: f32,
    end_phase_osc_freq: f32,
    start_phase_osc_range: f32,
    end_phase_osc_range: f32,
    start_amp_osc_depth_l: f32,
    end_amp_osc_depth_l: f32,
    start_amp_osc_freq_l: f32,
    end_amp_osc_freq_l: f32,
    start_amp_osc_depth_r: f32,
    end_amp_osc_depth_r: f32,
    start_amp_osc_freq_r: f32,
    end_amp_osc_freq_r: f32,
    start_amp_osc_phase_offset_l: f32,
    end_amp_osc_phase_offset_l: f32,
    start_amp_osc_phase_offset_r: f32,
    end_amp_osc_phase_offset_r: f32,
    start_freq_osc_range_l: f32,
    end_freq_osc_range_l: f32,
    start_freq_osc_freq_l: f32,
    end_freq_osc_freq_l: f32,
    start_freq_osc_range_r: f32,
    end_freq_osc_range_r: f32,
    start_freq_osc_freq_r: f32,
    end_freq_osc_freq_r: f32,
    curve: TransitionCurve,
    initial_offset: f32,
    post_offset: f32,
    sample_rate: f32,
    remaining_samples: usize,
    phase_l: f32,
    phase_r: f32,
    elapsed: f32,
    duration: f32,
}
impl BinauralBeatVoice {
    pub fn new(params: &HashMap<String, Value>, duration: f32, sample_rate: f32) -> Self {
        let amp_l = get_f32(params, "ampL", 0.5);
        let amp_r = get_f32(params, "ampR", 0.5);
        let base_freq = get_f32(params, "baseFreq", 200.0);
        let beat_freq = get_f32(params, "beatFreq", 4.0);
        let force_mono = get_bool(params, "forceMono", false);
        let start_phase_l = get_f32(params, "startPhaseL", 0.0);
        let start_phase_r = get_f32(params, "startPhaseR", 0.0);
        let amp_osc_depth_l = get_f32(params, "ampOscDepthL", 0.0);
        let amp_osc_freq_l = get_f32(params, "ampOscFreqL", 0.0);
        let amp_osc_depth_r = get_f32(params, "ampOscDepthR", 0.0);
        let amp_osc_freq_r = get_f32(params, "ampOscFreqR", 0.0);
        let freq_osc_range_l = get_f32(params, "freqOscRangeL", 0.0);
        let freq_osc_freq_l = get_f32(params, "freqOscFreqL", 0.0);
        let freq_osc_range_r = get_f32(params, "freqOscRangeR", 0.0);
        let freq_osc_freq_r = get_f32(params, "freqOscFreqR", 0.0);
        let amp_osc_phase_offset_l = get_f32(params, "ampOscPhaseOffsetL", 0.0);
        let amp_osc_phase_offset_r = get_f32(params, "ampOscPhaseOffsetR", 0.0);
        let phase_osc_freq = get_f32(params, "phaseOscFreq", 0.0);
        let phase_osc_range = get_f32(params, "phaseOscRange", 0.0);

        let total_samples = (duration * sample_rate) as usize;
        Self {
            amp_l,
            amp_r,
            base_freq,
            beat_freq,
            force_mono,
            start_phase_l,
            start_phase_r,
            amp_osc_depth_l,
            amp_osc_freq_l,
            amp_osc_depth_r,
            amp_osc_freq_r,
            freq_osc_range_l,
            freq_osc_freq_l,
            freq_osc_range_r,
            freq_osc_freq_r,
            amp_osc_phase_offset_l,
            amp_osc_phase_offset_r,
            phase_osc_freq,
            phase_osc_range,
            phase_l: start_phase_l,
            phase_r: start_phase_r,
            sample_rate,
            remaining_samples: total_samples,
            elapsed: 0.0,
        }
    }
}

impl BinauralBeatTransitionVoice {
    pub fn new(params: &HashMap<String, Value>, duration: f32, sample_rate: f32) -> Self {
        let start_amp_l = get_f32(params, "startAmpL", get_f32(params, "ampL", 0.5));
        let end_amp_l = get_f32(params, "endAmpL", start_amp_l);
        let start_amp_r = get_f32(params, "startAmpR", get_f32(params, "ampR", 0.5));
        let end_amp_r = get_f32(params, "endAmpR", start_amp_r);
        let start_base_freq = get_f32(params, "startBaseFreq", get_f32(params, "baseFreq", 200.0));
        let end_base_freq = get_f32(params, "endBaseFreq", start_base_freq);
        let start_beat_freq = get_f32(params, "startBeatFreq", get_f32(params, "beatFreq", 4.0));
        let end_beat_freq = get_f32(params, "endBeatFreq", start_beat_freq);
        let start_force_mono = get_bool(
            params,
            "startForceMono",
            get_bool(params, "forceMono", false),
        );
        let end_force_mono = get_bool(params, "endForceMono", start_force_mono);
        let start_start_phase_l = get_f32(
            params,
            "startStartPhaseL",
            get_f32(params, "startPhaseL", 0.0),
        );
        let end_start_phase_l = get_f32(params, "endStartPhaseL", start_start_phase_l);
        let start_start_phase_r = get_f32(
            params,
            "startStartPhaseR",
            get_f32(params, "startPhaseR", 0.0),
        );
        let end_start_phase_r = get_f32(params, "endStartPhaseR", start_start_phase_r);
        let start_phase_osc_freq = get_f32(
            params,
            "startPhaseOscFreq",
            get_f32(params, "phaseOscFreq", 0.0),
        );
        let end_phase_osc_freq = get_f32(params, "endPhaseOscFreq", start_phase_osc_freq);
        let start_phase_osc_range = get_f32(
            params,
            "startPhaseOscRange",
            get_f32(params, "phaseOscRange", 0.0),
        );
        let end_phase_osc_range = get_f32(params, "endPhaseOscRange", start_phase_osc_range);
        let start_amp_osc_depth_l = get_f32(
            params,
            "startAmpOscDepthL",
            get_f32(params, "ampOscDepthL", 0.0),
        );
        let end_amp_osc_depth_l = get_f32(params, "endAmpOscDepthL", start_amp_osc_depth_l);
        let start_amp_osc_freq_l = get_f32(
            params,
            "startAmpOscFreqL",
            get_f32(params, "ampOscFreqL", 0.0),
        );
        let end_amp_osc_freq_l = get_f32(params, "endAmpOscFreqL", start_amp_osc_freq_l);
        let start_amp_osc_depth_r = get_f32(
            params,
            "startAmpOscDepthR",
            get_f32(params, "ampOscDepthR", 0.0),
        );
        let end_amp_osc_depth_r = get_f32(params, "endAmpOscDepthR", start_amp_osc_depth_r);
        let start_amp_osc_freq_r = get_f32(
            params,
            "startAmpOscFreqR",
            get_f32(params, "ampOscFreqR", 0.0),
        );
        let end_amp_osc_freq_r = get_f32(params, "endAmpOscFreqR", start_amp_osc_freq_r);
        let start_amp_osc_phase_offset_l = get_f32(
            params,
            "startAmpOscPhaseOffsetL",
            get_f32(params, "ampOscPhaseOffsetL", 0.0),
        );
        let end_amp_osc_phase_offset_l = get_f32(
            params,
            "endAmpOscPhaseOffsetL",
            start_amp_osc_phase_offset_l,
        );
        let start_amp_osc_phase_offset_r = get_f32(
            params,
            "startAmpOscPhaseOffsetR",
            get_f32(params, "ampOscPhaseOffsetR", 0.0),
        );
        let end_amp_osc_phase_offset_r = get_f32(
            params,
            "endAmpOscPhaseOffsetR",
            start_amp_osc_phase_offset_r,
        );
        let start_freq_osc_range_l = get_f32(
            params,
            "startFreqOscRangeL",
            get_f32(params, "freqOscRangeL", 0.0),
        );
        let end_freq_osc_range_l = get_f32(params, "endFreqOscRangeL", start_freq_osc_range_l);
        let start_freq_osc_freq_l = get_f32(
            params,
            "startFreqOscFreqL",
            get_f32(params, "freqOscFreqL", 0.0),
        );
        let end_freq_osc_freq_l = get_f32(params, "endFreqOscFreqL", start_freq_osc_freq_l);
        let start_freq_osc_range_r = get_f32(
            params,
            "startFreqOscRangeR",
            get_f32(params, "freqOscRangeR", 0.0),
        );
        let end_freq_osc_range_r = get_f32(params, "endFreqOscRangeR", start_freq_osc_range_r);
        let start_freq_osc_freq_r = get_f32(
            params,
            "startFreqOscFreqR",
            get_f32(params, "freqOscFreqR", 0.0),
        );
        let end_freq_osc_freq_r = get_f32(params, "endFreqOscFreqR", start_freq_osc_freq_r);

        let curve = TransitionCurve::from_str(
            params
                .get("transition_curve")
                .and_then(|v| v.as_str())
                .unwrap_or("linear"),
        );
        let initial_offset = get_f32(params, "initial_offset", 0.0);
        let post_offset = get_f32(params, "post_offset", 0.0);

        let total_samples = (duration * sample_rate) as usize;

        Self {
            start_amp_l,
            end_amp_l,
            start_amp_r,
            end_amp_r,
            start_base_freq,
            end_base_freq,
            start_beat_freq,
            end_beat_freq,
            start_force_mono,
            end_force_mono,
            start_start_phase_l,
            end_start_phase_l,
            start_start_phase_r,
            end_start_phase_r,
            start_phase_osc_freq,
            end_phase_osc_freq,
            start_phase_osc_range,
            end_phase_osc_range,
            start_amp_osc_depth_l,
            end_amp_osc_depth_l,
            start_amp_osc_freq_l,
            end_amp_osc_freq_l,
            start_amp_osc_depth_r,
            end_amp_osc_depth_r,
            start_amp_osc_freq_r,
            end_amp_osc_freq_r,
            start_amp_osc_phase_offset_l,
            end_amp_osc_phase_offset_l,
            start_amp_osc_phase_offset_r,
            end_amp_osc_phase_offset_r,
            start_freq_osc_range_l,
            end_freq_osc_range_l,
            start_freq_osc_freq_l,
            end_freq_osc_freq_l,
            start_freq_osc_range_r,
            end_freq_osc_range_r,
            start_freq_osc_freq_r,
            end_freq_osc_freq_r,
            curve,
            initial_offset,
            post_offset,
            sample_rate,
            remaining_samples: total_samples,
            phase_l: start_start_phase_l,
            phase_r: start_start_phase_r,
            elapsed: 0.0,
            duration,
        }
    }
}

impl Voice for BinauralBeatTransitionVoice {
    fn process(&mut self, output: &mut [f32]) {
        let channels = 2;
        let frames = output.len() / channels;
        for i in 0..frames {
            if self.remaining_samples == 0 {
                break;
            }
            let dt = 1.0 / self.sample_rate;
            let t = self.elapsed;
            let alpha = if t < self.initial_offset {
                0.0
            } else if t > self.duration - self.post_offset {
                1.0
            } else {
                let span = self.duration - self.initial_offset - self.post_offset;
                if span > 0.0 {
                    (t - self.initial_offset) / span
                } else {
                    1.0
                }
            };
            let alpha = self.curve.apply(alpha.clamp(0.0, 1.0));

            let amp_l = self.start_amp_l + (self.end_amp_l - self.start_amp_l) * alpha;
            let amp_r = self.start_amp_r + (self.end_amp_r - self.start_amp_r) * alpha;
            let base_freq =
                self.start_base_freq + (self.end_base_freq - self.start_base_freq) * alpha;
            let beat_freq =
                self.start_beat_freq + (self.end_beat_freq - self.start_beat_freq) * alpha;
            let force_mono = if self.start_force_mono == self.end_force_mono {
                self.start_force_mono
            } else {
                alpha >= 0.5
            };
            let phase_osc_freq = self.start_phase_osc_freq
                + (self.end_phase_osc_freq - self.start_phase_osc_freq) * alpha;
            let phase_osc_range = self.start_phase_osc_range
                + (self.end_phase_osc_range - self.start_phase_osc_range) * alpha;
            let amp_osc_depth_l = self.start_amp_osc_depth_l
                + (self.end_amp_osc_depth_l - self.start_amp_osc_depth_l) * alpha;
            let amp_osc_freq_l = self.start_amp_osc_freq_l
                + (self.end_amp_osc_freq_l - self.start_amp_osc_freq_l) * alpha;
            let amp_osc_depth_r = self.start_amp_osc_depth_r
                + (self.end_amp_osc_depth_r - self.start_amp_osc_depth_r) * alpha;
            let amp_osc_freq_r = self.start_amp_osc_freq_r
                + (self.end_amp_osc_freq_r - self.start_amp_osc_freq_r) * alpha;
            let amp_osc_phase_offset_l = self.start_amp_osc_phase_offset_l
                + (self.end_amp_osc_phase_offset_l - self.start_amp_osc_phase_offset_l) * alpha;
            let amp_osc_phase_offset_r = self.start_amp_osc_phase_offset_r
                + (self.end_amp_osc_phase_offset_r - self.start_amp_osc_phase_offset_r) * alpha;
            let freq_osc_range_l = self.start_freq_osc_range_l
                + (self.end_freq_osc_range_l - self.start_freq_osc_range_l) * alpha;
            let freq_osc_freq_l = self.start_freq_osc_freq_l
                + (self.end_freq_osc_freq_l - self.start_freq_osc_freq_l) * alpha;
            let freq_osc_range_r = self.start_freq_osc_range_r
                + (self.end_freq_osc_range_r - self.start_freq_osc_range_r) * alpha;
            let freq_osc_freq_r = self.start_freq_osc_freq_r
                + (self.end_freq_osc_freq_r - self.start_freq_osc_freq_r) * alpha;

            // instantaneous frequencies
            let half_beat = beat_freq * 0.5;
            let mut freq_l = base_freq - half_beat
                + (freq_osc_range_l * 0.5)
                    * (2.0 * std::f32::consts::PI * freq_osc_freq_l * t).sin();
            let mut freq_r = base_freq
                + half_beat
                + (freq_osc_range_r * 0.5)
                    * (2.0 * std::f32::consts::PI * freq_osc_freq_r * t).sin();

            if force_mono || beat_freq == 0.0 {
                freq_l = base_freq.max(0.0);
                freq_r = base_freq.max(0.0);
            } else {
                if freq_l < 0.0 {
                    freq_l = 0.0;
                }
                if freq_r < 0.0 {
                    freq_r = 0.0;
                }
            }

            self.phase_l += 2.0 * std::f32::consts::PI * freq_l * dt;
            self.phase_r += 2.0 * std::f32::consts::PI * freq_r * dt;
            let mut ph_l = self.phase_l;
            let mut ph_r = self.phase_r;
            if phase_osc_freq != 0.0 || phase_osc_range != 0.0 {
                let dphi = (phase_osc_range * 0.5)
                    * (2.0 * std::f32::consts::PI * phase_osc_freq * t).sin();
                ph_l -= dphi;
                ph_r += dphi;
            }

            let env_l = 1.0
                - amp_osc_depth_l
                    * (0.5
                        * (1.0
                            + (2.0 * std::f32::consts::PI * amp_osc_freq_l * t
                                + amp_osc_phase_offset_l)
                                .sin()));
            let env_r = 1.0
                - amp_osc_depth_r
                    * (0.5
                        * (1.0
                            + (2.0 * std::f32::consts::PI * amp_osc_freq_r * t
                                + amp_osc_phase_offset_r)
                                .sin()));

            let sample_l = ph_l.sin() * env_l * amp_l;
            let sample_r = ph_r.sin() * env_r * amp_r;

            output[i * 2] += sample_l;
            output[i * 2 + 1] += sample_r;

            self.elapsed += dt;
            self.remaining_samples -= 1;
        }
    }

    fn is_finished(&self) -> bool {
        self.remaining_samples == 0
    }
}
impl Voice for BinauralBeatVoice {
    fn process(&mut self, output: &mut [f32]) {
        let channels = 2;
        let frames = output.len() / channels;
        for i in 0..frames {
            if self.remaining_samples == 0 {
                break;
            }
            let t = self.elapsed;

            // Instantaneous frequency with vibrato
            let half_beat = self.beat_freq * 0.5;
            let mut freq_l = self.base_freq - half_beat
                + (self.freq_osc_range_l * 0.5)
                    * (2.0 * std::f32::consts::PI * self.freq_osc_freq_l * t).sin();
            let mut freq_r = self.base_freq
                + half_beat
                + (self.freq_osc_range_r * 0.5)
                    * (2.0 * std::f32::consts::PI * self.freq_osc_freq_r * t).sin();

            if self.force_mono || self.beat_freq == 0.0 {
                freq_l = self.base_freq.max(0.0);
                freq_r = self.base_freq.max(0.0);
            } else {
                if freq_l < 0.0 {
                    freq_l = 0.0;
                }
                if freq_r < 0.0 {
                    freq_r = 0.0;
                }
            }

            // Advance phase
            let dt = 1.0 / self.sample_rate;
            self.phase_l += 2.0 * std::f32::consts::PI * freq_l * dt;
            self.phase_r += 2.0 * std::f32::consts::PI * freq_r * dt;

            // Phase modulation
            let mut ph_l = self.phase_l;
            let mut ph_r = self.phase_r;
            if self.phase_osc_freq != 0.0 || self.phase_osc_range != 0.0 {
                let dphi = (self.phase_osc_range * 0.5)
                    * (2.0 * std::f32::consts::PI * self.phase_osc_freq * t).sin();
                ph_l -= dphi;
                ph_r += dphi;
            }

            // Amplitude envelopes
            let env_l = 1.0
                - self.amp_osc_depth_l
                    * (0.5
                        * (1.0
                            + (2.0 * std::f32::consts::PI * self.amp_osc_freq_l * t
                                + self.amp_osc_phase_offset_l)
                                .sin()));
            let env_r = 1.0
                - self.amp_osc_depth_r
                    * (0.5
                        * (1.0
                            + (2.0 * std::f32::consts::PI * self.amp_osc_freq_r * t
                                + self.amp_osc_phase_offset_r)
                                .sin()));

            let sample_l = ph_l.sin() * env_l * self.amp_l;
            let sample_r = ph_r.sin() * env_r * self.amp_r;

            output[i * 2] += sample_l;
            output[i * 2 + 1] += sample_r;

            self.elapsed += dt;
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
        "binaural_beat" => Some(Box::new(BinauralBeatVoice::new(
            &data.params,
            duration,
            sample_rate,
        ))),
        "binaural_beat_transition" => Some(Box::new(BinauralBeatTransitionVoice::new(
            &data.params,
            duration,
            sample_rate,
        ))),
        _ => None,
    }
}
