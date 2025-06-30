mod audio_io;
mod dsp;
mod models;
mod scheduler;
mod voices;

use models::TrackData;
use pyo3::prelude::*;
use scheduler::TrackScheduler;
use parking_lot::Mutex;
use std::sync::Arc;
use once_cell::sync::Lazy;

static ENGINE_STATE: Lazy<Mutex<Option<Arc<Mutex<TrackScheduler>>>>> = Lazy::new(|| Mutex::new(None));

#[pyfunction]
fn start_stream(track_json_str: String) -> PyResult<()> {
    let track_data: TrackData = serde_json::from_str(&track_json_str)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let scheduler = Arc::new(Mutex::new(TrackScheduler::new(track_data)));
    *ENGINE_STATE.lock() = Some(scheduler.clone());
    std::thread::spawn(move || {
        audio_io::run_audio_stream(scheduler);
    });
    Ok(())
}

#[pyfunction]
fn stop_stream() -> PyResult<()> {
    *ENGINE_STATE.lock() = None;
    audio_io::stop_audio_stream();
    Ok(())
}

#[pymodule]
fn realtime_backend(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start_stream, m)?)?;
    m.add_function(wrap_pyfunction!(stop_stream, m)?)?;
    Ok(())
}
