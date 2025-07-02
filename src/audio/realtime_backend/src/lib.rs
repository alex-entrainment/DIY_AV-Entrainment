#![allow(dead_code)]
#![allow(unused_imports)]

#[cfg(feature = "python")]
pub mod audio_io;
pub mod dsp;
pub mod models;
pub mod scheduler;
pub mod command;
pub mod voices;

use models::TrackData;
use scheduler::TrackScheduler;
use parking_lot::Mutex;
use once_cell::sync::Lazy;
use ringbuf::{HeapRb, HeapProd, HeapCons};
use ringbuf::traits::{Producer, Consumer, Split};
#[cfg(feature = "python")]
use cpal::traits::HostTrait;
#[cfg(feature = "python")]
use cpal::traits::DeviceTrait;
use command::Command;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::prelude::Bound;
#[cfg(feature = "python")]
use crossbeam::channel::{unbounded, Sender};
#[cfg(feature = "web")]
use wasm_bindgen::prelude::*;

static ENGINE_STATE: Lazy<Mutex<Option<HeapProd<Command>>>> = Lazy::new(|| Mutex::new(None));
#[cfg(feature = "python")]
static STOP_SENDER: Lazy<Mutex<Option<Sender<()>>>> = Lazy::new(|| Mutex::new(None));
#[cfg(feature = "web")]
thread_local! {
    static WASM_SCHED: std::cell::RefCell<Option<(TrackScheduler, HeapCons<Command>)>> = std::cell::RefCell::new(None);
}

#[cfg(feature = "python")]
#[pyfunction]
fn start_stream(track_json_str: String) -> PyResult<()> {
    let track_data: TrackData = serde_json::from_str(&track_json_str)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let host = cpal::default_host();
    let device = host.default_output_device().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("no output device"))?;
    let cfg = device.default_output_config().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let stream_rate = cfg.sample_rate().0;

    let scheduler = TrackScheduler::new(track_data, stream_rate);
    let rb = HeapRb::<Command>::new(1024);
    let (prod, cons) = rb.split();
    *ENGINE_STATE.lock() = Some(prod);

    let (tx, rx) = unbounded();
    *STOP_SENDER.lock() = Some(tx);

    std::thread::spawn(move || {
        audio_io::run_audio_stream(scheduler, cons, rx);
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

#[cfg(feature = "python")]
#[pyfunction]
fn update_track(track_json_str: String) -> PyResult<()> {
    let track_data: TrackData = serde_json::from_str(&track_json_str)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    if let Some(prod) = &mut *ENGINE_STATE.lock() {
        let _ = prod.try_push(Command::UpdateTrack(track_data));
    }
    Ok(())
}

#[cfg(feature = "web")]
#[wasm_bindgen]
pub fn start_stream(track_json_str: &str, sample_rate: u32) {
    let track_data: TrackData = serde_json::from_str(track_json_str).unwrap();
    let scheduler = TrackScheduler::new(track_data, sample_rate);
    let rb = HeapRb::<Command>::new(1024);
    let (prod, cons) = rb.split();
    *ENGINE_STATE.lock() = Some(prod);
    // In wasm mode we don't spawn a thread; scheduler is stored globally for pull processing
    WASM_SCHED.with(|s| *s.borrow_mut() = Some((scheduler, cons)));
}

#[cfg(feature = "web")]
#[wasm_bindgen]
pub fn update_track(track_json_str: &str) {
    if let Some(prod) = &mut *ENGINE_STATE.lock() {
        if let Ok(track_data) = serde_json::from_str(track_json_str) {
            let _ = prod.try_push(Command::UpdateTrack(track_data));
        }
    }
}

#[cfg(feature = "web")]
#[wasm_bindgen]
pub fn process_block(frame_count: usize) -> js_sys::Float32Array {
    let mut buf = vec![0.0f32; frame_count];
    WASM_SCHED.with(|s| {
        if let Some((sched, cons)) = &mut *s.borrow_mut() {
            while let Some(cmd) = cons.try_pop() {
                sched.handle_command(cmd);
            }
            sched.process_block(&mut buf);
        }
    });
    js_sys::Float32Array::from(buf.as_slice())
}

#[cfg(feature = "web")]
#[wasm_bindgen]
pub fn stop_stream() {
    *ENGINE_STATE.lock() = None;
    WASM_SCHED.with(|s| *s.borrow_mut() = None);
}

#[cfg(feature = "python")]
#[pymodule]
fn realtime_backend(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start_stream, m)?)?;
    m.add_function(wrap_pyfunction!(stop_stream, m)?)?;
    m.add_function(wrap_pyfunction!(update_track, m)?)?;
    Ok(())
}
