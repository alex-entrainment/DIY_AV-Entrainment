use serde::Deserialize;
use std::collections::HashMap;

fn default_crossfade_duration() -> f64 { 1.0 }
fn default_crossfade_curve() -> String { "linear".to_string() }

#[derive(Deserialize, Debug, Clone)]
pub struct VolumeEnvelope {
    #[serde(rename = "type")]
    pub envelope_type: String,
    pub params: HashMap<String, f64>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct VoiceData {
    pub synth_function_name: String,
    pub params: HashMap<String, serde_json::Value>,
    pub volume_envelope: Option<VolumeEnvelope>,
    #[serde(default)]
    pub is_transition: bool,
    #[serde(default)]
    pub description: String,
}

#[derive(Deserialize, Debug, Clone)]
pub struct StepData {
    pub duration: f64,
    #[serde(default)]
    pub description: String,
    pub voices: Vec<VoiceData>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct GlobalSettings {
    pub sample_rate: u32,
    #[serde(default = "default_crossfade_duration")]
    pub crossfade_duration: f64,
    #[serde(default = "default_crossfade_curve")]
    pub crossfade_curve: String,
    #[serde(default)]
    pub output_filename: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct TrackData {
    pub global_settings: GlobalSettings,
    pub steps: Vec<StepData>,
    #[serde(default)]
    pub clips: Vec<ClipData>,
    #[serde(default)]
    pub background_noise: Option<BackgroundNoiseData>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ClipData {
    #[serde(alias = "path", alias = "file")]
    pub file_path: String,
    #[serde(default, alias = "start_time")]
    pub start: f64,
    #[serde(default, alias = "gain")]
    pub amp: f32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct BackgroundNoiseData {
    #[serde(default, alias = "file", alias = "file_path", alias = "params_path")]
    pub file_path: String,
    #[serde(default, rename = "type")]
    pub noise_type: String,
    #[serde(default, alias = "gain", alias = "amp")]
    pub amp: f32,
}
