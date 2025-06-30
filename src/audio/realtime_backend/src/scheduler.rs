use crate::models::TrackData;
use crate::voices::voices_for_step;

#[derive(Clone, Copy)]
enum CrossfadeCurve {
    Linear,
    EqualPower,
}

impl CrossfadeCurve {
    fn gains(&self, progress: f32) -> (f32, f32) {
        let p = progress.clamp(0.0, 1.0);
        match self {
            CrossfadeCurve::Linear => (1.0 - p, p),
            CrossfadeCurve::EqualPower => {
                let theta = p * std::f32::consts::FRAC_PI_2;
                (theta.cos(), theta.sin())
            }
        }
    }
}

pub trait Voice: Send + Sync {
    fn process(&mut self, output: &mut [f32]);
    fn is_finished(&self) -> bool;
}

pub struct TrackScheduler {
    pub track: TrackData,
    pub current_sample: usize,
    pub current_step: usize,
    pub active_voices: Vec<Box<dyn Voice>>,
    pub sample_rate: f32,
    crossfade_curve: CrossfadeCurve,
    crossfade_samples: usize,
    fade_out_voices: Vec<Box<dyn Voice>>,
    fade_in_voices: Vec<Box<dyn Voice>>,
    crossfade_pos: usize,
    in_crossfade: bool,
}

impl TrackScheduler {
    pub fn new(track: TrackData) -> Self {
        let sample_rate = track.global_settings.sample_rate as f32;
        let curve = match track.global_settings.crossfade_curve.as_str() {
            "equal_power" => CrossfadeCurve::EqualPower,
            _ => CrossfadeCurve::Linear,
        };
        let cf_samples = (track.global_settings.crossfade_duration * sample_rate as f64) as usize;
        Self {
            track,
            current_sample: 0,
            current_step: 0,
            active_voices: Vec::new(),
            sample_rate,
            crossfade_curve: curve,
            crossfade_samples: cf_samples,
            fade_out_voices: Vec::new(),
            fade_in_voices: Vec::new(),
            crossfade_pos: 0,
            in_crossfade: false,
        }
    }

    pub fn process_block(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = 0.0;
        }

        let channels = 2;
        let frames = buffer.len() / channels;

        if !self.in_crossfade && self.current_step >= self.track.steps.len() {
            return;
        }

        if !self.in_crossfade && self.active_voices.is_empty() {
            if let Some(step) = self.track.steps.get(self.current_step) {
                self.active_voices = voices_for_step(step, self.sample_rate);
            }
        }

        if self.in_crossfade {
            self.process_crossfade(buffer, frames);
            return;
        }

        for voice in &mut self.active_voices {
            voice.process(buffer);
        }
        self.active_voices.retain(|v| !v.is_finished());

        if self.current_step + 1 < self.track.steps.len() && self.crossfade_samples > 0 {
            let step = &self.track.steps[self.current_step];
            let step_samples = (step.duration * self.sample_rate as f64) as usize;
            let cf_start = step_samples.saturating_sub(self.crossfade_samples);
            if self.current_sample >= cf_start {
                self.start_crossfade();
            }
        }

        self.current_sample += frames;
        if let Some(step) = self.track.steps.get(self.current_step) {
            let step_samples = (step.duration * self.sample_rate as f64) as usize;
            if self.current_sample >= step_samples {
                self.current_step += 1;
                self.current_sample = 0;
                self.active_voices.clear();
            }
        }
    }

    fn start_crossfade(&mut self) {
        if self.in_crossfade || self.current_step + 1 >= self.track.steps.len() {
            return;
        }
        self.fade_out_voices = std::mem::take(&mut self.active_voices);
        let next_step = &self.track.steps[self.current_step + 1];
        self.fade_in_voices = voices_for_step(next_step, self.sample_rate);
        self.in_crossfade = true;
        self.crossfade_pos = 0;
    }

    fn process_crossfade(&mut self, buffer: &mut [f32], frames: usize) {
        let mut out_buf = vec![0.0f32; buffer.len()];
        let mut in_buf = vec![0.0f32; buffer.len()];

        for v in &mut self.fade_out_voices {
            v.process(&mut out_buf);
        }
        for v in &mut self.fade_in_voices {
            v.process(&mut in_buf);
        }

        for i in 0..frames {
            let pos = self.crossfade_pos + i;
            let progress = (pos as f32) / (self.crossfade_samples as f32);
            let (fo, fi) = self.crossfade_curve.gains(progress);
            let li = i * 2;
            buffer[li] = out_buf[li] * fo + in_buf[li] * fi;
            buffer[li + 1] = out_buf[li + 1] * fo + in_buf[li + 1] * fi;
        }

        self.crossfade_pos += frames;
        self.fade_out_voices.retain(|v| !v.is_finished());
        self.fade_in_voices.retain(|v| !v.is_finished());

        if self.crossfade_pos >= self.crossfade_samples {
            let leftover = self.crossfade_pos - self.crossfade_samples;
            self.in_crossfade = false;
            self.active_voices = std::mem::take(&mut self.fade_in_voices);
            self.fade_out_voices.clear();
            self.current_step += 1;
            self.current_sample = leftover;
            self.crossfade_pos = 0;
        }
    }
}
