use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use crossbeam::channel::Receiver;
use ringbuf::traits::Consumer;

use crate::command::Command;

use crate::scheduler::TrackScheduler;

pub fn run_audio_stream<C>(scheduler: TrackScheduler, cmd_rx: C, stop_rx: Receiver<()>)
where
    C: Consumer<Item = Command> + Send + 'static,
{
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("no output device available");
    let supported_config = device
        .default_output_config()
        .expect("no default config");
    let sample_format = supported_config.sample_format();
    let mut config: StreamConfig = supported_config.clone().into();

    // Use the scheduler's sample rate if it differs from the device default.
    let desired_rate = scheduler.sample_rate as u32;
    if desired_rate != config.sample_rate.0 {
        if let Ok(mut ranges) = device.supported_output_configs() {
            if let Some(range) = ranges.find(|r| {
                r.channels() == config.channels
                    && r.sample_format() == sample_format
                    && r.min_sample_rate().0 <= desired_rate
                    && desired_rate <= r.max_sample_rate().0
            }) {
                config = range
                    .with_sample_rate(cpal::SampleRate(desired_rate))
                    .config();
            } else {
                eprintln!(
                    "Sample rate {} not supported, using {}",
                    desired_rate, config.sample_rate.0
                );
            }
        } else {
            eprintln!("Could not query supported output configs; using default");
        }
    }

    let mut sched = scheduler;
    let mut cmds = cmd_rx;
    let audio_callback = move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
        while let Some(cmd) = cmds.try_pop() {
            sched.handle_command(cmd);
        }
        sched.process_block(data);
    };

    let stream = match sample_format {
        SampleFormat::F32 => device
            .build_output_stream(&config, audio_callback, |err| eprintln!("stream error: {err}"), None)
            .expect("failed to build output stream"),
        _ => panic!("Unsupported sample format"),
    };
    stream.play().unwrap();

    // Keep the stream alive until a stop signal is received
    while stop_rx.recv_timeout(std::time::Duration::from_millis(100)).is_err() {}
}

// The actual stop logic is handled via the channel in `run_audio_stream`.
pub fn stop_audio_stream(sender: &crossbeam::channel::Sender<()>) {
    let _ = sender.send(());
}
