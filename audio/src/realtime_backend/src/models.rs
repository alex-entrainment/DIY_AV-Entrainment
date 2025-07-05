use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

fn default_amp() -> f32 {
    1.0
}

fn default_crossfade_duration() -> f64 {
    3.0
}
fn default_crossfade_curve() -> String {
    "linear".to_string()
}

#[derive(Deserialize, Debug, Clone)]
pub struct VolumeEnvelope {
    #[serde(rename = "type")]
    pub envelope_type: String,
    pub params: HashMap<String, f64>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct VoiceData {
    #[serde(alias = "synthFunctionName", alias = "synth_function")]
    pub synth_function_name: String,
    #[serde(alias = "parameters", default)]
    pub params: HashMap<String, serde_json::Value>,
    #[serde(alias = "volumeEnvelope")]
    pub volume_envelope: Option<VolumeEnvelope>,
    #[serde(default, alias = "isTransition")]
    pub is_transition: bool,
    #[serde(default)]
    pub description: String,
}

#[derive(Deserialize, Debug, Clone)]
pub struct StepData {
    #[serde(alias = "Duration", alias = "durationSeconds", alias = "stepDuration")]
    pub duration: f64,
    #[serde(default)]
    pub description: String,
    pub voices: Vec<VoiceData>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct GlobalSettings {
    #[serde(alias = "sampleRate")]
    pub sample_rate: u32,
    #[serde(default = "default_crossfade_duration", alias = "crossfadeDuration")]
    pub crossfade_duration: f64,
    #[serde(default = "default_crossfade_curve", alias = "crossfadeCurve")]
    pub crossfade_curve: String,
    #[serde(default, alias = "outputFilename")]
    pub output_filename: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct TrackData {
    #[serde(alias = "globalSettings", alias = "global")]
    pub global_settings: GlobalSettings,
    #[serde(alias = "progression")]
    pub steps: Vec<StepData>,
    #[serde(default, alias = "overlay_clips")]
    pub clips: Vec<ClipData>,
    #[serde(default, alias = "noise")]
    pub background_noise: Option<BackgroundNoiseData>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ClipData {
    #[serde(alias = "path", alias = "file")]
    pub file_path: String,
    #[serde(default, alias = "start_time")]
    pub start: f64,
    #[serde(default = "default_amp", alias = "gain")]
    pub amp: f32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct BackgroundNoiseData {
    #[serde(default, alias = "file", alias = "file_path", alias = "params_path")]
    pub file_path: String,
    #[serde(default, rename = "type")]
    pub noise_type: String,
    #[serde(default = "default_amp", alias = "gain", alias = "amp")]
    pub amp: f32,
    #[serde(default)]
    pub params: Option<crate::noise_params::NoiseParams>,
}

impl TrackData {
    /// Resolve clip and noise file paths relative to the provided base directory.
    pub fn resolve_relative_paths<P: AsRef<Path>>(&mut self, base: P) {
        let base = base.as_ref();
        if let Some(noise) = &mut self.background_noise {
            if !noise.file_path.is_empty() {
                let p = Path::new(&noise.file_path);
                if p.is_relative() {
                    noise.file_path = base.join(p).to_string_lossy().into_owned();
                }
            }
        }
        for clip in &mut self.clips {
            if !clip.file_path.is_empty() {
                let p = Path::new(&clip.file_path);
                if p.is_relative() {
                    clip.file_path = base.join(p).to_string_lossy().into_owned();
                }
            }
        }
    }
}
