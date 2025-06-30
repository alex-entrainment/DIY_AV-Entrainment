#![allow(dead_code)]
#![allow(unused_imports)]

#[cfg(feature = "python")]
mod audio_io;
mod dsp;
mod models;
mod scheduler;
mod voices;

use models::TrackData;
use scheduler::TrackScheduler;
use parking_lot::Mutex;
use once_cell::sync::Lazy;
use std::sync::Arc;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use crossbeam::channel::{unbounded, Sender};
#[cfg(feature = "web")]
use wasm_bindgen::prelude::*;

static ENGINE_STATE: Lazy<Mutex<Option<Arc<Mutex<TrackScheduler>>>>> = Lazy::new(|| Mutex::new(None));
#[cfg(feature = "python")]
static STOP_SENDER: Lazy<Mutex<Option<Sender<()>>>> = Lazy::new(|| Mutex::new(None));

#[cfg(feature = "python")]
#[pyfunction]
fn start_stream(track_json_str: String) -> PyResult<()> {
    let track_data: TrackData = serde_json::from_str(&track_json_str)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let scheduler = Arc::new(Mutex::new(TrackScheduler::new(track_data)));
    *ENGINE_STATE.lock() = Some(scheduler.clone());

    let (tx, rx) = unbounded();
    *STOP_SENDER.lock() = Some(tx);

    std::thread::spawn(move || {
        audio_io::run_audio_stream(scheduler, rx);
    });
    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction]
fn stop_stream() -> PyResult<()> {
    *ENGINE_STATE.lock() = None;
    if let Some(tx) = STOP_SENDER.lock().take() {
        audio_io::stop_audio_stream(&tx);
    }
    Ok(())
}

#[cfg(feature = "web")]
#[wasm_bindgen]
pub fn start_stream(track_json_str: &str) {
    let track_data: TrackData = serde_json::from_str(track_json_str).unwrap();
    let scheduler = Arc::new(Mutex::new(TrackScheduler::new(track_data)));
    *ENGINE_STATE.lock() = Some(scheduler);
}

#[cfg(feature = "web")]
#[wasm_bindgen]
pub fn process_block(frame_count: usize) -> js_sys::Float32Array {
    let mut buf = vec![0.0f32; frame_count];
    if let Some(engine) = &*ENGINE_STATE.lock() {
        engine.lock().process_block(&mut buf);
    }
    js_sys::Float32Array::from(buf.as_slice())
}

#[cfg(feature = "web")]
#[wasm_bindgen]
pub fn stop_stream() {
    *ENGINE_STATE.lock() = None;
}

#[cfg(feature = "python")]
#[pymodule]
fn realtime_backend(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start_stream, m)?)?;
    m.add_function(wrap_pyfunction!(stop_stream, m)?)?;
    Ok(())
}
