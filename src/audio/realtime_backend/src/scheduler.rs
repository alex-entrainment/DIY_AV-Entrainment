use crate::models::TrackData;
use crate::voices::voices_for_step;

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

pub struct TrackScheduler {
    pub track: TrackData,
    pub current_sample: usize,
    pub current_step: usize,
    pub active_voices: Vec<Box<dyn Voice>>,
    pub next_voices: Vec<Box<dyn Voice>>,
    pub sample_rate: f32,
    pub crossfade_samples: usize,
    pub crossfade_curve: CrossfadeCurve,
    pub next_step_sample: usize,
    pub crossfade_active: bool,
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
        Self {
            track,
            current_sample: 0,
            current_step: 0,
            active_voices: Vec::new(),
            next_voices: Vec::new(),
            sample_rate,
            crossfade_samples,
            crossfade_curve,
            next_step_sample: 0,
            crossfade_active: false,
        }
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
            let step_samples = (step.duration * self.sample_rate as f64) as usize;
            if self.current_sample >= step_samples.saturating_sub(self.crossfade_samples) {
                let next_step = &self.track.steps[self.current_step + 1];
                self.next_voices = voices_for_step(next_step, self.sample_rate);
                self.crossfade_active = true;
                self.next_step_sample = 0;
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

            for i in 0..frames {
                let idx = i * 2;
                let progress = self.next_step_sample + i;
                if progress < self.crossfade_samples {
                    let ratio = progress as f32 / self.crossfade_samples as f32;
                    let (g_out, g_in) = self.crossfade_curve.gains(ratio);
                    buffer[idx] = prev_buf[idx] * g_out + next_buf[idx] * g_in;
                    buffer[idx + 1] = prev_buf[idx + 1] * g_out + next_buf[idx + 1] * g_in;
                } else {
                    buffer[idx] = next_buf[idx];
                    buffer[idx + 1] = next_buf[idx + 1];
                }
            }

            self.current_sample += frames;
            self.next_step_sample += frames;

            self.active_voices.retain(|v| !v.is_finished());
            self.next_voices.retain(|v| !v.is_finished());

            if self.next_step_sample >= self.crossfade_samples {
                self.current_step += 1;
                self.current_sample = self.next_step_sample;
                self.next_step_sample = 0;
                self.active_voices = std::mem::take(&mut self.next_voices);
                self.crossfade_active = false;
            }
        } else {
            for voice in &mut self.active_voices {
                voice.process(buffer);
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
    }
}
