use clap::Parser;
use realtime_backend::models::TrackData;
use realtime_backend::scheduler::TrackScheduler;
use realtime_backend::command::Command;
use realtime_backend::audio_io;
use realtime_backend::config::CONFIG;
use ringbuf::HeapRb;
use ringbuf::traits::Split;
use crossbeam::channel::unbounded;
use cpal::traits::{DeviceTrait, HostTrait};

/// Simple CLI to play a track JSON file using the realtime backend
#[derive(Parser)]
struct Args {
    /// Path to the track JSON file
    track_file: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let json_str = std::fs::read_to_string(&args.track_file)?;
    let track_data: TrackData = serde_json::from_str(&json_str)?;

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or("no output device")?;
    let cfg = device.default_output_config()?;
    let stream_rate = cfg.sample_rate().0;

    let mut scheduler = TrackScheduler::new(track_data, stream_rate);
    scheduler.gpu_enabled = CONFIG.gpu;
    let rb = HeapRb::<Command>::new(1024);
    let (_prod, cons) = rb.split();
    let (tx, rx) = unbounded();

    std::thread::spawn(move || {
        audio_io::run_audio_stream(scheduler, cons, rx);
    });

    println!("Playing {}... press Ctrl+C to stop", args.track_file);
    ctrlc::set_handler(move || {
        let _ = tx.send(());
    })?;

    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
