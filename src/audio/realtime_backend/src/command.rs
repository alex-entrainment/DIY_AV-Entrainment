use crate::models::TrackData;

#[derive(Debug)]
pub enum Command {
    UpdateTrack(TrackData),
}
