use crate::models::TrackData;

#[derive(Debug)]
pub enum Command {
    UpdateTrack(TrackData),
    /// Enable or disable GPU accelerated mixing
    EnableGpu(bool),
}
