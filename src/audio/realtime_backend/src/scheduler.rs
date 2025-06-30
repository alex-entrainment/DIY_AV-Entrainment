use crate::models::{TrackData};
use crate::voices::voices_for_step;

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
}

impl TrackScheduler {
    pub fn new(track: TrackData) -> Self {
        let sample_rate = track.global_settings.sample_rate as f32;
        Self {
            track,
            current_sample: 0,
            current_step: 0,
            active_voices: Vec::new(),
            sample_rate,
        }
    }

    pub fn process_block(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = 0.0;
        }
        if self.current_step >= self.track.steps.len() {
            return;
        }
        let step = &self.track.steps[self.current_step];
        if self.current_sample == 0 {
            let new_voices = voices_for_step(step, self.sample_rate);
            self.active_voices.extend(new_voices);
        }
        for voice in &mut self.active_voices {
            voice.process(buffer);
        }
        self.active_voices.retain(|v| !v.is_finished());
        // TODO crossfade and step transition logic
        self.current_sample += buffer.len();
        if self.current_sample as f32 / self.sample_rate >= step.duration as f32 {
            self.current_step += 1;
            self.current_sample = 0;
            // active_voices will be populated on next call when current_sample == 0
        }
    }
}
