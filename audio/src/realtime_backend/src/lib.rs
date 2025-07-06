#![allow(dead_code)]
#![allow(unused_imports)]

#[cfg(feature = "python")]
pub mod audio_io;
pub mod dsp;
pub mod models;
pub mod noise_params;
pub mod streaming_noise;
pub mod scheduler;
pub mod command;
pub mod voices;
pub mod gpu;
pub mod config;

use config::CONFIG;

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
#[cfg(feature = "python")]
use hound;
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
#[pyo3(signature = (track_json_str, start_time=None))]
fn start_stream(track_json_str: String, start_time: Option<f64>) -> PyResult<()> {
    let track_data: TrackData = serde_json::from_str(&track_json_str)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let host = cpal::default_host();
    let device = host.default_output_device().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("no output device"))?;
    let cfg = device.default_output_config().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let stream_rate = cfg.sample_rate().0;

    let start_secs = start_time.unwrap_or(0.0);
    let mut scheduler = TrackScheduler::new_with_start(track_data, stream_rate, start_secs);
    // Disable GPU usage for realtime playback. GPU acceleration is reserved
    // for offline rendering to avoid unnecessary overhead during streaming.
    scheduler.gpu_enabled = false;
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

#[cfg(feature = "python")]
#[pyfunction]
fn enable_gpu(enable: bool) -> PyResult<()> {
    if let Some(prod) = &mut *ENGINE_STATE.lock() {
        let _ = prod.try_push(Command::EnableGpu(enable));
    }
    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction]
fn pause_stream() -> PyResult<()> {
    if let Some(prod) = &mut *ENGINE_STATE.lock() {
        let _ = prod.try_push(Command::SetPaused(true));
    }
    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction]
fn resume_stream() -> PyResult<()> {
    if let Some(prod) = &mut *ENGINE_STATE.lock() {
        let _ = prod.try_push(Command::SetPaused(false));
    }
    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction]
fn start_from(position: f64) -> PyResult<()> {
    if let Some(prod) = &mut *ENGINE_STATE.lock() {
        let _ = prod.try_push(Command::StartFrom(position));
    }
    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (index, samples, finished=false))]
fn push_clip_samples(index: usize, samples: Vec<f32>, finished: bool) -> PyResult<()> {
    if let Some(prod) = &mut *ENGINE_STATE.lock() {
        let _ = prod.try_push(Command::PushClipSamples { index, data: samples, finished });
    }
    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction]
fn render_sample_wav(track_json_str: String, out_path: String) -> PyResult<()> {
    use hound::{WavSpec, WavWriter, SampleFormat};
    let track_data: TrackData = serde_json::from_str(&track_json_str)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let sample_rate = track_data.global_settings.sample_rate;
    let mut scheduler = TrackScheduler::new(track_data.clone(), sample_rate);
    // Use GPU acceleration when rendering to a file if available.
    // Streaming paths keep GPU disabled.
    scheduler.gpu_enabled = true;
    let track_frames: usize = track_data
        .steps
        .iter()
        .map(|s| (s.duration * sample_rate as f64) as usize)
        .sum();
    let target_frames = (sample_rate as usize * 60).min(track_frames);

    let spec = WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let output_path = if std::path::Path::new(&out_path).is_absolute() {
        std::path::PathBuf::from(&out_path)
    } else {
        CONFIG.output_dir.join(&out_path)
    };

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    }

    let mut writer = WavWriter::create(&output_path, spec)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let mut remaining = target_frames;
    let mut buffer = vec![0.0f32; 512 * 2];
    while remaining > 0 {
        let frames = 512.min(remaining);
        buffer.resize(frames * 2, 0.0);
        scheduler.process_block(&mut buffer);
        for sample in &buffer[..frames * 2] {
            let s = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer.write_sample(s).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        }
        remaining -= frames;
    }

    writer.finalize().map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction]
fn render_full_wav(track_json_str: String, out_path: String) -> PyResult<()> {
    use hound::{WavSpec, WavWriter, SampleFormat};
    let track_data: TrackData = serde_json::from_str(&track_json_str)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let sample_rate = track_data.global_settings.sample_rate;
    let mut scheduler = TrackScheduler::new(track_data.clone(), sample_rate);
    // Enable GPU acceleration during full track rendering if available.
    scheduler.gpu_enabled = true;
    let target_frames: usize = track_data
        .steps
        .iter()
        .map(|s| (s.duration * sample_rate as f64) as usize)
        .sum();

    let spec = WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let output_path = if std::path::Path::new(&out_path).is_absolute() {
        std::path::PathBuf::from(&out_path)
    } else {
        CONFIG.output_dir.join(&out_path)
    };

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    }

    let mut writer = WavWriter::create(&output_path, spec)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    let start_time = std::time::Instant::now();

    let mut remaining = target_frames;
    let mut buffer = vec![0.0f32; 512 * 2];
    while remaining > 0 {
        let frames = 512.min(remaining);
        buffer.resize(frames * 2, 0.0);
        scheduler.process_block(&mut buffer);
        for sample in &buffer[..frames * 2] {
            let s = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer.write_sample(s).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        }
        remaining -= frames;
    }

    writer.finalize().map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    let elapsed = start_time.elapsed().as_secs_f32();
    println!("Total generation time: {:.2}s", elapsed);
    Ok(())
}

#[cfg(feature = "web")]
#[wasm_bindgen]
pub fn start_stream(track_json_str: &str, sample_rate: u32, start_time: f64) {
    let track_data: TrackData = serde_json::from_str(track_json_str).unwrap();
    let scheduler = TrackScheduler::new_with_start(track_data, sample_rate, start_time);
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
pub fn enable_gpu(enable: bool) {
    if let Some(prod) = &mut *ENGINE_STATE.lock() {
        let _ = prod.try_push(Command::EnableGpu(enable));
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
pub fn current_step() -> usize {
    let mut step = 0usize;
    WASM_SCHED.with(|s| {
        if let Some((sched, _)) = &*s.borrow() {
            step = sched.current_step_index();
        }
    });
    step
}

#[cfg(feature = "web")]
#[wasm_bindgen]
pub fn elapsed_samples() -> u64 {
    let mut samples = 0u64;
    WASM_SCHED.with(|s| {
        if let Some((sched, _)) = &*s.borrow() {
            samples = sched.elapsed_samples();
        }
    });
    samples
}

#[cfg(feature = "web")]
#[wasm_bindgen]
pub fn pause_stream() {
    if let Some(prod) = &mut *ENGINE_STATE.lock() {
        let _ = prod.try_push(Command::SetPaused(true));
    }
}

#[cfg(feature = "web")]
#[wasm_bindgen]
pub fn resume_stream() {
    if let Some(prod) = &mut *ENGINE_STATE.lock() {
        let _ = prod.try_push(Command::SetPaused(false));
    }
}

#[cfg(feature = "web")]
#[wasm_bindgen]
pub fn start_from(position: f64) {
    if let Some(prod) = &mut *ENGINE_STATE.lock() {
        let _ = prod.try_push(Command::StartFrom(position));
    }
}

#[cfg(feature = "web")]
#[wasm_bindgen]
pub fn push_clip_samples(index: usize, samples: &js_sys::Float32Array, finished: bool) {
    let mut vec = vec![0.0f32; samples.length() as usize];
    samples.copy_to(&mut vec);
    if let Some(prod) = &mut *ENGINE_STATE.lock() {
        let _ = prod.try_push(Command::PushClipSamples { index, data: vec, finished });
    }
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
    m.add_function(wrap_pyfunction!(pause_stream, m)?)?;
    m.add_function(wrap_pyfunction!(resume_stream, m)?)?;
    m.add_function(wrap_pyfunction!(start_from, m)?)?;
    m.add_function(wrap_pyfunction!(push_clip_samples, m)?)?;
    m.add_function(wrap_pyfunction!(update_track, m)?)?;
    m.add_function(wrap_pyfunction!(render_sample_wav, m)?)?;
    m.add_function(wrap_pyfunction!(render_full_wav, m)?)?;
    m.add_function(wrap_pyfunction!(enable_gpu, m)?)?;
    Ok(())
}
