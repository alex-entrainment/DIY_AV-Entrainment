use crate::dsp::noise_flanger::generate_swept_notch_noise;
use crate::dsp::{generate_brown_noise_samples, generate_pink_noise_samples};
use crate::models::{StepData, TrackData};
use crate::voices::voices_for_step;
use std::fs::File;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_probe};
pub trait Voice: Send + Sync {
    fn process(&mut self, output: &mut [f32]);
    fn is_finished(&self) -> bool;
}

#[derive(Clone, Copy)]
pub enum CrossfadeCurve {
    Linear,
    EqualPower,
}

impl CrossfadeCurve {
    fn gains(self, ratio: f32) -> (f32, f32) {
        match self {
            CrossfadeCurve::Linear => (1.0 - ratio, ratio),
            CrossfadeCurve::EqualPower => {
                let theta = ratio * std::f32::consts::FRAC_PI_2;
                (theta.cos(), theta.sin())
            }
        }
    }
}

fn steps_have_continuous_voices(a: &StepData, b: &StepData) -> bool {
    if a.voices.len() != b.voices.len() {
        return false;
    }

    for (va, vb) in a.voices.iter().zip(&b.voices) {
        if va.synth_function_name != vb.synth_function_name {
            return false;
        }
        if va.params != vb.params {
            return false;
        }
        if va.is_transition != vb.is_transition {
            return false;
        }
    }

    true
}

pub struct TrackScheduler {
    pub track: TrackData,
    pub current_sample: usize,
    pub current_step: usize,
    pub active_voices: Vec<Box<dyn Voice>>,
    pub next_voices: Vec<Box<dyn Voice>>,
    pub sample_rate: f32,
    pub crossfade_samples: usize,
    pub current_crossfade_samples: usize,
    pub crossfade_curve: CrossfadeCurve,
    pub crossfade_envelope: Vec<f32>,
    pub next_step_sample: usize,
    pub crossfade_active: bool,
    pub absolute_sample: usize,
    pub clips: Vec<LoadedClip>,
    pub background_noise: Option<BackgroundNoise>,
}

pub struct LoadedClip {
    samples: Vec<f32>,
    start_sample: usize,
    position: usize,
    gain: f32,
}

pub struct BackgroundNoise {
    samples: Vec<f32>,
    position: usize,
    gain: f32,
}

fn load_clip_file(path: &str, sample_rate: u32) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let probed = get_probe().format(
        &Hint::new(),
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;
    let track = format.default_track().ok_or("no default track")?;
    let mut decoder = get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let src_rate = track
        .codec_params
        .sample_rate
        .ok_or("unknown sample rate")?;
    let channels = track
        .codec_params
        .channels
        .ok_or("unknown channel count")?
        .count();

    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    let mut samples: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(_)) => break,
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(e) => return Err(Box::new(e)),
        };
        let decoded = decoder.decode(&packet)?;
        if sample_buf.is_none() {
            sample_buf = Some(SampleBuffer::<f32>::new(
                decoded.capacity() as u64,
                *decoded.spec(),
            ));
        }
        let sbuf = sample_buf.as_mut().unwrap();
        sbuf.copy_interleaved_ref(decoded);
        let data = sbuf.samples();
        for frame in data.chunks(channels) {
            let l = frame[0];
            let r = if channels > 1 { frame[1] } else { frame[0] };
            samples.push(l);
            samples.push(r);
        }
    }
    if src_rate != sample_rate {
        samples = resample_linear_stereo(&samples, src_rate, sample_rate);
    }
    Ok(samples)
}

fn resample_linear_stereo(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || input.is_empty() {
        return input.to_vec();
    }
    let frames = input.len() / 2;
    let duration = frames as f64 / src_rate as f64;
    let out_frames = (duration * dst_rate as f64).round() as usize;
    let mut out = vec![0.0f32; out_frames * 2];
    for i in 0..out_frames {
        let t = i as f64 / dst_rate as f64;
        let pos = t * src_rate as f64;
        let idx = pos.floor() as usize;
        let frac = pos - idx as f64;
        let idx2 = if idx + 1 < frames { idx + 1 } else { idx };
        for ch in 0..2 {
            let x0 = input[idx * 2 + ch];
            let x1 = input[idx2 * 2 + ch];
            out[i * 2 + ch] = ((1.0 - frac) * x0 as f64 + frac * x1 as f64) as f32;
        }
    }
    out
}

impl TrackScheduler {
    pub fn new(track: TrackData) -> Self {
        let sample_rate = track.global_settings.sample_rate as f32;
        let crossfade_samples =
            (track.global_settings.crossfade_duration * sample_rate as f64) as usize;
        let crossfade_curve = match track.global_settings.crossfade_curve.as_str() {
            "equal_power" => CrossfadeCurve::EqualPower,
            _ => CrossfadeCurve::Linear,
        };
        let mut clips = Vec::new();
        for c in &track.clips {
            if let Ok(samples) = load_clip_file(&c.file_path, track.global_settings.sample_rate) {
                clips.push(LoadedClip {
                    samples,
                    start_sample: (c.start * sample_rate as f64) as usize,
                    position: 0,
                    gain: c.amp,
                });
            }
        }

        let background_noise = if let Some(noise_cfg) = &track.background_noise {
            let total_duration: f64 = track.steps.iter().map(|s| s.duration).sum();
            let total_samples = (total_duration * sample_rate as f64) as usize;
            let samples = match noise_cfg.noise_type.to_lowercase().as_str() {
                "brown" => generate_brown_noise_samples(total_samples),
                "swept_notch" => generate_swept_notch_noise(
                    total_duration as f32,
                    track.global_settings.sample_rate,
                    1.0 / 12.0,
                    &[(1000.0, 10000.0)],
                    &[25.0],
                    &[10],
                    90.0,
                    0.0,
                    "pink",
                    "sine",
                ),
                _ => generate_pink_noise_samples(total_samples),
            };
            let mut stereo = Vec::with_capacity(samples.len() * 2);
            for s in samples {
                stereo.push(s);
                stereo.push(s);
            }
            Some(BackgroundNoise {
                samples: stereo,
                position: 0,
                gain: noise_cfg.amp,
            })
        } else {
            None
        };

        Self {
            track,
            current_sample: 0,
            current_step: 0,
            active_voices: Vec::new(),
            next_voices: Vec::new(),
            sample_rate,
            crossfade_samples,
            current_crossfade_samples: 0,
            crossfade_curve,
            crossfade_envelope: Vec::new(),
            next_step_sample: 0,
            crossfade_active: false,
            absolute_sample: 0,
            clips,
            background_noise,
        }
    }

    /// Replace the current track data while preserving playback progress.
    pub fn update_track(&mut self, track: TrackData) {
        let abs_samples = self.absolute_sample;

        self.sample_rate = track.global_settings.sample_rate as f32;
        self.crossfade_samples =
            (track.global_settings.crossfade_duration * self.sample_rate as f64) as usize;
        self.crossfade_curve = match track.global_settings.crossfade_curve.as_str() {
            "equal_power" => CrossfadeCurve::EqualPower,
            _ => CrossfadeCurve::Linear,
        };

        self.track = track.clone();

        self.clips.clear();
        for c in &track.clips {
            if let Ok(samples) = load_clip_file(&c.file_path, track.global_settings.sample_rate) {
                let start_sample = (c.start * self.sample_rate as f64) as usize;
                let position = if abs_samples > start_sample {
                    (abs_samples - start_sample) * 2
                } else {
                    0
                };
                self.clips.push(LoadedClip {
                    samples,
                    start_sample,
                    position,
                    gain: c.amp,
                });
            }
        }

        self.background_noise = if let Some(noise_cfg) = &track.background_noise {
            let total_duration: f64 = track.steps.iter().map(|s| s.duration).sum();
            let total_samples = (total_duration * self.sample_rate as f64) as usize;
            let samples = match noise_cfg.noise_type.to_lowercase().as_str() {
                "brown" => generate_brown_noise_samples(total_samples),
                "swept_notch" => generate_swept_notch_noise(
                    total_duration as f32,
                    track.global_settings.sample_rate,
                    1.0 / 12.0,
                    &[(1000.0, 10000.0)],
                    &[25.0],
                    &[10],
                    90.0,
                    0.0,
                    "pink",
                    "sine",
                ),
                _ => generate_pink_noise_samples(total_samples),
            };
            let mut stereo = Vec::with_capacity(samples.len() * 2);
            for s in samples {
                stereo.push(s);
                stereo.push(s);
            }
            let pos = (abs_samples * 2).min(stereo.len());
            Some(BackgroundNoise {
                samples: stereo,
                position: pos,
                gain: noise_cfg.amp,
            })
        } else {
            None
        };

        let mut remaining = abs_samples;
        self.current_step = 0;
        self.current_sample = 0;
        for (idx, step) in track.steps.iter().enumerate() {
            let step_samples = (step.duration * self.sample_rate as f64) as usize;
            if remaining < step_samples {
                self.current_step = idx;
                self.current_sample = remaining;
                break;
            }
            remaining = remaining.saturating_sub(step_samples);
        }

        self.active_voices.clear();
        self.next_voices.clear();
        self.crossfade_active = false;
        self.current_crossfade_samples = 0;
        self.next_step_sample = 0;
    }

    pub fn process_block(&mut self, buffer: &mut [f32]) {
        buffer.fill(0.0);

        if self.current_step >= self.track.steps.len() {
            return;
        }

        if self.active_voices.is_empty() && !self.crossfade_active {
            let step = &self.track.steps[self.current_step];
            self.active_voices = voices_for_step(step, self.sample_rate);
        }

        // Check if we need to start crossfade into the next step
        if !self.crossfade_active
            && self.crossfade_samples > 0
            && self.current_step + 1 < self.track.steps.len()
        {
            let step = &self.track.steps[self.current_step];
            let next_step = &self.track.steps[self.current_step + 1];
            if !steps_have_continuous_voices(step, next_step) {
                let step_samples = (step.duration * self.sample_rate as f64) as usize;
                let fade_len = self.crossfade_samples.min(step_samples);
                if self.current_sample >= step_samples.saturating_sub(fade_len) {
                    self.next_voices = voices_for_step(next_step, self.sample_rate);
                    self.crossfade_active = true;
                    self.next_step_sample = 0;
                    let next_samples = (next_step.duration * self.sample_rate as f64) as usize;
                    self.current_crossfade_samples = self.crossfade_samples.min(step_samples).min(next_samples);
                    self.crossfade_envelope = if self.current_crossfade_samples <= 1 {
                        vec![0.0; self.current_crossfade_samples]
                    } else {
                        (0..self.current_crossfade_samples)
                            .map(|i| i as f32 / (self.current_crossfade_samples - 1) as f32)
                            .collect()
                    };
                }
            }
        }

        if self.crossfade_active {
            let len = buffer.len();
            let frames = len / 2;
            let mut prev_buf = vec![0.0f32; len];
            let mut next_buf = vec![0.0f32; len];

            for v in &mut self.active_voices {
                v.process(&mut prev_buf);
            }
            for v in &mut self.next_voices {
                v.process(&mut next_buf);
            }

            let out_gain = if self.active_voices.is_empty() {
                0.0
            } else {
                1.0 / self.active_voices.len() as f32
            };
            let in_gain = if self.next_voices.is_empty() {
                0.0
            } else {
                1.0 / self.next_voices.len() as f32
            };

            for i in 0..frames {
                let idx = i * 2;
                let progress = self.next_step_sample + i;
                if progress < self.current_crossfade_samples {
                    let ratio = if progress < self.crossfade_envelope.len() {
                        self.crossfade_envelope[progress]
                    } else {
                        progress as f32 / (self.current_crossfade_samples - 1) as f32
                    };
                    let (g_out, g_in) = self.crossfade_curve.gains(ratio);
                    buffer[idx] = prev_buf[idx] * g_out * out_gain + next_buf[idx] * g_in * in_gain;
                    buffer[idx + 1] =
                        prev_buf[idx + 1] * g_out * out_gain + next_buf[idx + 1] * g_in * in_gain;
                } else {
                    buffer[idx] = next_buf[idx] * in_gain;
                    buffer[idx + 1] = next_buf[idx + 1] * in_gain;
                }
            }

            self.current_sample += frames;
            self.next_step_sample += frames;

            self.active_voices.retain(|v| !v.is_finished());
            self.next_voices.retain(|v| !v.is_finished());

            if self.next_step_sample >= self.current_crossfade_samples {
                self.current_step += 1;
                self.current_sample = self.next_step_sample;
                self.next_step_sample = 0;
                self.active_voices = std::mem::take(&mut self.next_voices);
                self.crossfade_active = false;
                self.crossfade_envelope.clear();
                self.current_crossfade_samples = 0;
            }
        } else {
            let gain = if self.active_voices.is_empty() {
                0.0
            } else {
                1.0 / self.active_voices.len() as f32
            };
            let mut voice_buf = vec![0.0f32; buffer.len()];
            for voice in &mut self.active_voices {
                voice_buf.fill(0.0);
                voice.process(&mut voice_buf);
                for (out, sample) in buffer.iter_mut().zip(&voice_buf) {
                    *out += sample * gain;
                }
            }
            self.active_voices.retain(|v| !v.is_finished());
            let frames = buffer.len() / 2;
            self.current_sample += frames;
            let step = &self.track.steps[self.current_step];
            let step_samples = (step.duration * self.sample_rate as f64) as usize;
            if self.current_sample >= step_samples {
                self.current_step += 1;
                self.current_sample = 0;
                self.active_voices.clear();
            }
        }

        let frames = buffer.len() / 2;

        if let Some(noise) = &mut self.background_noise {
            for i in 0..frames {
                if noise.position + 1 >= noise.samples.len() {
                    break;
                }
                buffer[i * 2] += noise.samples[noise.position] * noise.gain;
                buffer[i * 2 + 1] += noise.samples[noise.position + 1] * noise.gain;
                noise.position += 2;
            }
        }

        let start_sample = self.absolute_sample;
        for clip in &mut self.clips {
            if start_sample + frames < clip.start_sample {
                continue;
            }
            let mut pos = clip.position;
            if start_sample < clip.start_sample {
                let offset = clip.start_sample - start_sample;
                pos += offset * 2;
            }
            for i in 0..frames {
                let global_idx = start_sample + i;
                if global_idx < clip.start_sample {
                    continue;
                }
                if pos + 1 >= clip.samples.len() {
                    break;
                }
                buffer[i * 2] += clip.samples[pos] * clip.gain;
                buffer[i * 2 + 1] += clip.samples[pos + 1] * clip.gain;
                pos += 2;
            }
            clip.position = pos;
        }

        // Apply a simple hard limiter for safety
        for sample in buffer.iter_mut() {
            *sample = sample.clamp(-0.95, 0.95);
        }

        self.absolute_sample += frames;
    }
}

#[cfg(test)]
mod tests {
    use super::CrossfadeCurve;

    #[test]
    fn test_fade_curves_match_python() {
        let samples = 5;
        for curve in [CrossfadeCurve::Linear, CrossfadeCurve::EqualPower] {
            for i in 0..samples {
                let ratio = i as f32 / (samples - 1) as f32;
                let (g_out, g_in) = curve.gains(ratio);
                let (exp_out, exp_in) = match curve {
                    CrossfadeCurve::Linear => (1.0 - ratio, ratio),
                    CrossfadeCurve::EqualPower => {
                        let theta = ratio * std::f32::consts::FRAC_PI_2;
                        (theta.cos(), theta.sin())
                    }
                };
                assert!((g_out - exp_out).abs() < 1e-6);
                assert!((g_in - exp_in).abs() < 1e-6);
            }
        }
    }
}
