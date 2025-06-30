use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use parking_lot::Mutex;
use std::sync::Arc;

use crate::scheduler::TrackScheduler;

pub fn run_audio_stream(scheduler: Arc<Mutex<TrackScheduler>>) {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("no output device available");
    let supported_config = device.default_output_config().expect("no default config");
    let sample_format = supported_config.sample_format();
    let config: StreamConfig = supported_config.into();

    let audio_callback = move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
        let mut scheduler = scheduler.lock();
        scheduler.process_block(data);
    };
    let err_fn = |err| eprintln!("stream error: {err}");

    let stream = match sample_format {
        SampleFormat::F32 => device
            .build_output_stream(&config, audio_callback, err_fn, None)
            .unwrap(),
        _ => panic!("Unsupported sample format"),
    };
    stream.play().unwrap();

    // Keep the stream alive until the thread is killed
    loop {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

pub fn stop_audio_stream() {
    // TODO: Implement a mechanism to stop the audio thread
}
