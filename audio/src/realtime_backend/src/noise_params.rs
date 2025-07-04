use serde::Deserialize;

#[derive(Deserialize, Debug, Clone, Default)]
pub struct NoiseSweep {
    #[serde(default)]
    pub start_min: f32,
    #[serde(default)]
    pub end_min: f32,
    #[serde(default)]
    pub start_max: f32,
    #[serde(default)]
    pub end_max: f32,
    #[serde(default)]
    pub start_q: f32,
    #[serde(default)]
    pub end_q: f32,
    #[serde(default)]
    pub start_casc: usize,
    #[serde(default)]
    pub end_casc: usize,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct NoiseParams {
    #[serde(default)]
    pub duration_seconds: f32,
    #[serde(default)]
    pub sample_rate: u32,
    #[serde(default)]
    pub noise_type: String,
    #[serde(default)]
    pub lfo_waveform: String,
    #[serde(default)]
    pub transition: bool,
    #[serde(default)]
    pub lfo_freq: f32,
    #[serde(default)]
    pub start_lfo_freq: f32,
    #[serde(default)]
    pub end_lfo_freq: f32,
    #[serde(default)]
    pub sweeps: Vec<NoiseSweep>,
    #[serde(default)]
    pub start_lfo_phase_offset_deg: f32,
    #[serde(default)]
    pub end_lfo_phase_offset_deg: f32,
    #[serde(default)]
    pub start_intra_phase_offset_deg: f32,
    #[serde(default)]
    pub end_intra_phase_offset_deg: f32,
    #[serde(default)]
    pub initial_offset: f32,
    #[serde(default)]
    pub post_offset: f32,
    #[serde(default)]
    pub input_audio_path: String,
}

pub fn load_noise_params(path: &str) -> Result<NoiseParams, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let params: NoiseParams = serde_json::from_reader(file)?;
    Ok(params)
}

pub fn load_noise_params_from_str(data: &str) -> Result<NoiseParams, serde_json::Error> {
    serde_json::from_str(data)
}
