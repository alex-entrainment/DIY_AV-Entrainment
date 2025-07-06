use crate::streaming_noise::StreamingNoise;
use crate::models::{StepData, TrackData};
use crate::voices::{voices_for_step, VoiceKind};
use crate::gpu::GpuMixer;
use crate::config::CONFIG;
use std::fs::File;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSourceStream, MediaSource};
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
                (crate::dsp::trig::cos_lut(theta), crate::dsp::trig::sin_lut(theta))
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
    pub active_voices: Vec<VoiceKind>,
    pub next_voices: Vec<VoiceKind>,
    pub sample_rate: f32,
    pub crossfade_samples: usize,
    pub current_crossfade_samples: usize,
    pub crossfade_curve: CrossfadeCurve,
    pub crossfade_envelope: Vec<f32>,
    crossfade_prev: Vec<f32>,
    crossfade_next: Vec<f32>,
    pub next_step_sample: usize,
    pub crossfade_active: bool,
    pub absolute_sample: u64,
    /// Whether playback is paused
    pub paused: bool,
    pub clips: Vec<LoadedClip>,
    pub background_noise: Option<BackgroundNoise>,
    pub scratch: Vec<f32>,
    /// Whether GPU accelerated mixing should be used when available
    pub gpu_enabled: bool,
    pub voice_gain: f32,
    pub noise_gain: f32,
    pub clip_gain: f32,
    #[cfg(feature = "gpu")]
    pub gpu: GpuMixer,
}

pub enum ClipSamples {
    Static(Vec<f32>),
    Streaming { data: Vec<f32>, finished: bool },
}

pub struct LoadedClip {
    samples: ClipSamples,
    start_sample: usize,
    position: usize,
    gain: f32,
}

pub struct BackgroundNoise {
    generator: StreamingNoise,
    gain: f32,
}

use crate::command::Command;
use std::io::Cursor;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;

fn decode_clip_reader<R: MediaSource + 'static>(reader: R, sample_rate: u32) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mss = MediaSourceStream::new(Box::new(reader), Default::default());
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

fn load_clip_bytes(data: &[u8], sample_rate: u32) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let cursor = Cursor::new(data.to_vec());
    decode_clip_reader(cursor, sample_rate)
}
fn load_clip_file(path: &str, sample_rate: u32) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if path.starts_with("data:") {
        if let Some(idx) = path.find(',') {
            let (_, b64) = path.split_at(idx + 1);
            let bytes = BASE64.decode(b64.trim())?;
            return load_clip_bytes(&bytes, sample_rate);
        } else {
            return Err("invalid data url".into());
        }
    }

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
    pub fn new(track: TrackData, device_rate: u32) -> Self {
        Self::new_with_start(track, device_rate, 0.0)
    }

    pub fn new_with_start(track: TrackData, device_rate: u32, start_time: f64) -> Self {
        let sample_rate = device_rate as f32;
        let crossfade_samples =
            (track.global_settings.crossfade_duration * sample_rate as f64) as usize;
        let crossfade_curve = match track.global_settings.crossfade_curve.as_str() {
            "equal_power" => CrossfadeCurve::EqualPower,
            _ => CrossfadeCurve::Linear,
        };
        let mut clips = Vec::new();
        let cfg = &CONFIG;
        for c in &track.clips {
            let clip_samples = match load_clip_file(&c.file_path, device_rate) {
                Ok(samples) => ClipSamples::Static(samples),
                Err(_) => ClipSamples::Streaming { data: Vec::new(), finished: false },
            };
            clips.push(LoadedClip {
                samples: clip_samples,
                start_sample: (c.start * sample_rate as f64) as usize,
                position: 0,
                gain: c.amp * cfg.clip_gain,
            });
        }

        let background_noise = if let Some(noise_cfg) = &track.background_noise {
            if !noise_cfg.file_path.is_empty() && noise_cfg.file_path.ends_with(".noise") {
                if let Ok(params) = crate::noise_params::load_noise_params(&noise_cfg.file_path) {
                    let gen = StreamingNoise::new(&params, device_rate);
                    Some(BackgroundNoise { generator: gen, gain: noise_cfg.amp * cfg.noise_gain })
                } else {
                    None
                }
            } else if let Some(params) = &noise_cfg.params {
                let gen = StreamingNoise::new(params, device_rate);
                Some(BackgroundNoise { generator: gen, gain: noise_cfg.amp * cfg.noise_gain })
            } else {
                None
            }
        } else {
            None
        };

        let mut sched = Self {
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
            crossfade_prev: Vec::new(),
            crossfade_next: Vec::new(),
            next_step_sample: 0,
            crossfade_active: false,
            absolute_sample: 0,
            paused: false,
            clips,
            background_noise,
            scratch: Vec::new(),
            gpu_enabled: cfg.gpu,
            voice_gain: cfg.voice_gain,
            noise_gain: cfg.noise_gain,
            clip_gain: cfg.clip_gain,
            #[cfg(feature = "gpu")]
            gpu: GpuMixer::new(),
        };

        let start_samples = (start_time * sample_rate as f64) as usize;
        sched.seek_samples(start_samples);
        sched
    }

    fn seek_samples(&mut self, abs_samples: usize) {
        self.absolute_sample = abs_samples as u64;

        for clip in &mut self.clips {
            clip.position = if abs_samples > clip.start_sample {
                (abs_samples - clip.start_sample) * 2
            } else {
                0
            };
        }



        let mut remaining = abs_samples;
        self.current_step = 0;
        self.current_sample = 0;
        for (idx, step) in self.track.steps.iter().enumerate() {
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
        self.crossfade_prev.clear();
        self.crossfade_next.clear();
        if let Some(noise) = &mut self.background_noise {
            noise.generator.skip_samples(abs_samples);
        }
    }

    /// Replace the current track data while preserving playback progress.
    pub fn update_track(&mut self, track: TrackData) {
        let abs_samples = self.absolute_sample as usize;

        self.crossfade_samples =
            (track.global_settings.crossfade_duration * self.sample_rate as f64) as usize;
        self.crossfade_curve = match track.global_settings.crossfade_curve.as_str() {
            "equal_power" => CrossfadeCurve::EqualPower,
            _ => CrossfadeCurve::Linear,
        };

        self.track = track.clone();

        self.clips.clear();
        for c in &track.clips {
            let clip_samples = match load_clip_file(&c.file_path, self.sample_rate as u32) {
                Ok(samples) => ClipSamples::Static(samples),
                Err(_) => ClipSamples::Streaming { data: Vec::new(), finished: false },
            };
            self.clips.push(LoadedClip {
                samples: clip_samples,
                start_sample: (c.start * self.sample_rate as f64) as usize,
                position: 0,
                gain: c.amp * self.clip_gain,
            });
        }

        self.background_noise = if let Some(noise_cfg) = &track.background_noise {
            if !noise_cfg.file_path.is_empty() && noise_cfg.file_path.ends_with(".noise") {
                if let Ok(params) = crate::noise_params::load_noise_params(&noise_cfg.file_path) {
                    let gen = StreamingNoise::new(&params, self.sample_rate as u32);
                    Some(BackgroundNoise { generator: gen, gain: noise_cfg.amp * self.noise_gain })
                } else {
                    None
                }
            } else if let Some(params) = &noise_cfg.params {
                let gen = StreamingNoise::new(params, self.sample_rate as u32);
                Some(BackgroundNoise { generator: gen, gain: noise_cfg.amp * self.noise_gain })
            } else {
                None
            }
        } else {
            None
        };

        self.seek_samples(abs_samples);
        self.crossfade_prev.clear();
        self.crossfade_next.clear();
        #[cfg(feature = "gpu")]
        {
            self.gpu = GpuMixer::new();
        }
    }

    pub fn handle_command(&mut self, cmd: Command) {
        match cmd {
            Command::UpdateTrack(t) => self.update_track(t),
            Command::EnableGpu(enable) => {
                self.gpu_enabled = enable;
            }
            Command::SetPaused(p) => {
                if p {
                    self.pause();
                } else {
                    self.resume();
                }
            }
            Command::StartFrom(time) => {
                let samples = (time * self.sample_rate as f64) as usize;
                self.seek_samples(samples);
            }
            Command::PushClipSamples { index, data, finished } => {
                if let Some(clip) = self.clips.get_mut(index) {
                    if let ClipSamples::Streaming { data: buf, finished: fin } = &mut clip.samples {
                        buf.extend_from_slice(&data);
                        if finished {
                            *fin = true;
                        }
                    }
                }
            }
        }
    }

    pub fn pause(&mut self) {
        self.paused = true;
    }

    pub fn resume(&mut self) {
        self.paused = false;
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    pub fn current_step_index(&self) -> usize {
        self.current_step
    }

    pub fn elapsed_samples(&self) -> u64 {
        self.absolute_sample
    }

    pub fn process_block(&mut self, buffer: &mut [f32]) {
        let frame_count = buffer.len() / 2;
        buffer.fill(0.0);

        if self.paused {
            return;
        }

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
                    self.current_crossfade_samples =
                        self.crossfade_samples.min(step_samples).min(next_samples);
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
            if self.crossfade_prev.len() != len {
                self.crossfade_prev.resize(len, 0.0);
            }
            if self.crossfade_next.len() != len {
                self.crossfade_next.resize(len, 0.0);
            }
            let prev_buf = &mut self.crossfade_prev;
            let next_buf = &mut self.crossfade_next;
            prev_buf.fill(0.0);
            next_buf.fill(0.0);

            if self.gpu_enabled {
                #[cfg(feature = "gpu")]
                {
                    let mut prev_locals: Vec<Vec<f32>> = Vec::with_capacity(self.active_voices.len());
                    for voice in &mut self.active_voices {
                        let mut local = vec![0.0f32; buffer.len()];
                        voice.process(&mut local);
                        prev_locals.push(local);
                    }
                    let prev_refs: Vec<&[f32]> = prev_locals.iter().map(|b| b.as_slice()).collect();
                    self.gpu.mix(&prev_refs, prev_buf);

                    let mut next_locals: Vec<Vec<f32>> = Vec::with_capacity(self.next_voices.len());
                    for voice in &mut self.next_voices {
                        let mut local = vec![0.0f32; buffer.len()];
                        voice.process(&mut local);
                        next_locals.push(local);
                    }
                    let next_refs: Vec<&[f32]> = next_locals.iter().map(|b| b.as_slice()).collect();
                    self.gpu.mix(&next_refs, next_buf);
                }
                #[cfg(not(feature = "gpu"))]
                {
                    for v in &mut self.active_voices {
                        v.process(prev_buf);
                    }
                    for v in &mut self.next_voices {
                        v.process(next_buf);
                    }
                }
            } else {
                for v in &mut self.active_voices {
                    v.process(prev_buf);
                }
                for v in &mut self.next_voices {
                    v.process(next_buf);
                }
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
            // --- EFFICIENT GAIN STAGING FOR NORMAL PLAYBACK ---
            let num_voices = self.active_voices.len();
            if num_voices > 0 {
                if self.scratch.len() != buffer.len() {
                    self.scratch.resize(buffer.len(), 0.0);
                }
                if self.gpu_enabled {
                    #[cfg(feature = "gpu")]
                    {
                        let mut voice_bufs: Vec<Vec<f32>> = Vec::with_capacity(num_voices);
                        for voice in &mut self.active_voices {
                            let mut local = vec![0.0f32; buffer.len()];
                            voice.process(&mut local);
                            voice_bufs.push(local);
                        }
                        let refs: Vec<&[f32]> = voice_bufs.iter().map(|b| b.as_slice()).collect();
                        self.gpu.mix(&refs, buffer);
                    }
                    #[cfg(not(feature = "gpu"))]
                    {
                        // fallback to CPU mixing when GPU support is unavailable
                        let gain = 1.0 / num_voices as f32;
                        for voice in &mut self.active_voices {
                            self.scratch.fill(0.0);
                            voice.process(&mut self.scratch);
                            for i in 0..buffer.len() {
                                buffer[i] += self.scratch[i] * gain;
                            }
                        }
                    }
                } else {
                    let gain = 1.0 / num_voices as f32;
                    for voice in &mut self.active_voices {
                        self.scratch.fill(0.0);
                        voice.process(&mut self.scratch);
                        for i in 0..buffer.len() {
                            buffer[i] += self.scratch[i] * gain;
                        }
                    }
                }
            }

            self.active_voices.retain(|v| !v.is_finished());
            self.current_sample += frame_count;
            let step = &self.track.steps[self.current_step];
            let step_samples = (step.duration * self.sample_rate as f64) as usize;
            if self.current_sample >= step_samples {
                self.current_step += 1;
                self.current_sample = 0;
                self.active_voices.clear();
            }
        }

        for v in &mut buffer[..] {
            *v *= self.voice_gain;
        }

        let frames = frame_count;

        if let Some(noise) = &mut self.background_noise {
            if self.scratch.len() != buffer.len() {
                self.scratch.resize(buffer.len(), 0.0);
            }
            self.scratch.fill(0.0);
            noise.generator.generate(&mut self.scratch);
            for i in 0..buffer.len() {
                buffer[i] += self.scratch[i] * noise.gain;
            }
        }

        let start_sample = self.absolute_sample as usize;
        for clip in &mut self.clips {
            if start_sample + frames < clip.start_sample {
                continue;
            }
            let mut pos = clip.position;
            if start_sample < clip.start_sample {
                let offset = clip.start_sample - start_sample;
                pos += offset * 2;
            }
            match &mut clip.samples {
                ClipSamples::Static(data) => {
                    for i in 0..frames {
                        let global_idx = start_sample + i;
                        if global_idx < clip.start_sample {
                            continue;
                        }
                        if pos + 1 >= data.len() {
                            break;
                        }
                        buffer[i * 2] += data[pos] * clip.gain;
                        buffer[i * 2 + 1] += data[pos + 1] * clip.gain;
                        pos += 2;
                    }
                }
                ClipSamples::Streaming { data, finished } => {
                    for i in 0..frames {
                        let global_idx = start_sample + i;
                        if global_idx < clip.start_sample {
                            continue;
                        }
                        if pos + 1 >= data.len() {
                            break;
                        }
                        buffer[i * 2] += data[pos] * clip.gain;
                        buffer[i * 2 + 1] += data[pos + 1] * clip.gain;
                        pos += 2;
                    }
                    if *finished && pos >= data.len() {
                        pos = data.len();
                    }
                    // Remove consumed samples to free memory
                    if pos > 4096 {
                        data.drain(0..pos);
                        clip.start_sample += pos / 2;
                        pos = 0;
                    }
                }
            }
            clip.position = pos;
        }

        // Normalize including noise and overlay clips to avoid clipping
        const THRESH: f32 = 0.95;
        let mut max_val = 0.0f32;
        for &s in buffer.iter() {
            if s.abs() > max_val {
                max_val = s.abs();
            }
        }
        if max_val > THRESH {
            let norm = THRESH / max_val;
            for v in buffer.iter_mut() {
                *v *= norm;
            }
        }

        self.absolute_sample += frame_count as u64;
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
                        (
                            crate::dsp::trig::cos_lut(theta),
                            crate::dsp::trig::sin_lut(theta),
                        )
                    }
                };
                assert!((g_out - exp_out).abs() < 1e-6);
                assert!((g_in - exp_in).abs() < 1e-6);
            }
        }
    }
}
