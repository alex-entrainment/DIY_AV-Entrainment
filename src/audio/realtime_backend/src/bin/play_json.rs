use clap::Parser;
use realtime_backend::models::TrackData;
use realtime_backend::scheduler::TrackScheduler;
use realtime_backend::command::Command;
use realtime_backend::audio_io;
use realtime_backend::config::CONFIG;
use ringbuf::HeapRb;
use ringbuf::traits::{Split, Producer};
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
    let mut track_data: TrackData = serde_json::from_str(&json_str)?;
    if let Some(dir) = std::path::Path::new(&args.track_file).parent() {
        track_data.resolve_relative_paths(dir);
    }

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or("no output device")?;
    let cfg = device.default_output_config()?;
    let stream_rate = cfg.sample_rate().0;

    let mut scheduler = TrackScheduler::new(track_data, stream_rate);
    scheduler.gpu_enabled = CONFIG.gpu;
    let rb = HeapRb::<Command>::new(1024);
    let (mut prod, cons) = rb.split();
    let (tx, rx) = unbounded();
    let rx_thread = rx.clone();

    std::thread::spawn(move || {
        audio_io::run_audio_stream(scheduler, cons, rx_thread);
    });
    println!("Playing {}...", args.track_file);
    println!("Controls: p = toggle pause/resume, q = quit");
    ctrlc::set_handler({
        let tx = tx.clone();
        move || {
            let _ = tx.send(());
        }
    })?;

    let input_thread = std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut paused = false;
        loop {
            let mut buf = String::new();
            if stdin.read_line(&mut buf).is_err() {
                continue;
            }
            match buf.trim() {
                "p" => {
                    paused = !paused;
                    let _ = prod.try_push(Command::SetPaused(paused));
                    if paused {
                        println!("Paused");
                    } else {
                        println!("Resumed");
                    }
                }
                "q" => {
                    let _ = tx.send(());
                    break;
                }
                _ => {
                    println!("p = pause/resume, q = quit");
                }
            }
        }
    });

    let _ = rx.recv();
    let _ = input_thread.join();
    Ok(())
}
