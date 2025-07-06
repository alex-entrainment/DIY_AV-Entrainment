use crate::models::TrackData;

#[derive(Debug)]
pub enum Command {
    UpdateTrack(TrackData),
    /// Enable or disable GPU accelerated mixing
    EnableGpu(bool),
    /// Pause or resume playback
    SetPaused(bool),
    /// Seek to a new playback position in seconds
    StartFrom(f64),
    /// Feed audio samples to a streaming overlay clip
    PushClipSamples { index: usize, data: Vec<f32>, finished: bool },
}
