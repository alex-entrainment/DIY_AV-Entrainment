use once_cell::sync::Lazy;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize, Debug, Clone)]
pub struct BackendConfig {
    #[serde(default = "default_output_dir")]
    pub output_dir: PathBuf,
    #[serde(default)]
    pub gpu: bool,
    #[serde(default = "default_gain")]
    pub voice_gain: f32,
    #[serde(default = "default_gain")]
    pub noise_gain: f32,
    #[serde(default = "default_gain")]
    pub clip_gain: f32,
}

fn default_output_dir() -> PathBuf {
    PathBuf::from("output")
}

fn default_gain() -> f32 {
    1.0
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            output_dir: default_output_dir(),
            gpu: false,
            voice_gain: 1.0,
            noise_gain: 1.0,
            clip_gain: 1.0,
        }
    }
}

pub static CONFIG: Lazy<BackendConfig> = Lazy::new(|| {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config.toml");
    if let Ok(txt) = std::fs::read_to_string(&path) {
        toml::from_str(&txt).unwrap_or_default()
    } else {
        BackendConfig::default()
    }
});
