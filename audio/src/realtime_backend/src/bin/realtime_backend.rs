use clap::{Parser, Subcommand, Args as ClapArgs};
use realtime_backend::models::TrackData;
use realtime_backend::scheduler::TrackScheduler;
use realtime_backend::command::Command;
use realtime_backend::audio_io;
use realtime_backend::config::{CONFIG, BackendConfig};
use ringbuf::HeapRb;
use ringbuf::traits::{Split, Producer};
use crossbeam::channel::unbounded;
use cpal::traits::{DeviceTrait, HostTrait};

/// CLI for streaming or rendering a track using the realtime backend
#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Stream or render a track JSON file
    Run(RunArgs),
    /// Generate a default config file and exit
    GenerateConfig(ConfigArgs),
}

#[derive(ClapArgs)]
struct RunArgs {
    /// Path to the track JSON file
    #[arg(long)]
    path: String,
    /// Generate the full track to the output file instead of streaming
    #[arg(long, default_value_t = false)]
    generate: bool,
    /// Enable GPU accelerated mixing (requires building with `--features gpu`)
    #[arg(long, default_value_t = false)]
    gpu: bool,
    /// Start playback from this time in seconds
    #[arg(long, default_value_t = 0.0)]
    start: f64,
}

#[derive(ClapArgs)]
struct ConfigArgs {
    /// Output path for the generated configuration
    #[arg(long, default_value = "config.toml")]
    out: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => run_command(args)?,
        Commands::GenerateConfig(cfg) => {
            BackendConfig::generate_default(&cfg.out)?;
            println!("Generated default config at {}", cfg.out);
        }
    }
    Ok(())
}

fn run_command(args: RunArgs) -> Result<(), Box<dyn std::error::Error>> {
    let json_str = std::fs::read_to_string(&args.path)?;
    let mut track_data: TrackData = serde_json::from_str(&json_str)?;
    if let Some(dir) = std::path::Path::new(&args.path).parent() {
        track_data.resolve_relative_paths(dir);
    }

    if args.generate {
        let out_name = track_data
            .global_settings
            .output_filename
            .clone()
            .ok_or("outputFilename missing in global settings")?;
        let out_path = if std::path::Path::new(&out_name).is_absolute() {
            std::path::PathBuf::from(&out_name)
        } else {
            CONFIG.output_dir.join(&out_name)
        };
        render_full_wav(track_data, out_path.to_str().unwrap(), args.gpu)?;
        println!("Generated full track at {}", out_path.display());
        return Ok(());
    }

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or("no output device")?;
    let cfg = device.default_output_config()?;
    let stream_rate = cfg.sample_rate().0;


    let mut scheduler = TrackScheduler::new(track_data, stream_rate);
    // GPU acceleration is reserved for file generation. Disable it during
    // realtime streaming to avoid extra overhead.
    scheduler.gpu_enabled = false;
    let rb = HeapRb::<Command>::new(1024);
    let (mut prod, cons) = rb.split();
    let (tx, rx) = unbounded();
    let rx_thread = rx.clone();

    std::thread::spawn(move || {
        audio_io::run_audio_stream(scheduler, cons, rx_thread);
    });

    println!("Streaming {}...", args.path);
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

fn render_full_wav(
    track_data: TrackData,
    out_path: &str,
    gpu: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use hound::{WavSpec, WavWriter, SampleFormat};
    let sample_rate = track_data.global_settings.sample_rate;
    let mut scheduler = TrackScheduler::new(track_data.clone(), sample_rate);
    // Use the GPU when rendering if requested by the caller. Streaming never
    // enables GPU acceleration.
    scheduler.gpu_enabled = gpu;
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

    let output_path = if std::path::Path::new(out_path).is_absolute() {
        std::path::PathBuf::from(out_path)
    } else {
        CONFIG.output_dir.join(out_path)
    };

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut writer = WavWriter::create(&output_path, spec)?;
    let start_time = std::time::Instant::now();
    let mut remaining = target_frames;
    let mut buffer = vec![0.0f32; 512 * 2];
    while remaining > 0 {
        let frames = 512.min(remaining);
        buffer.resize(frames * 2, 0.0);
        scheduler.process_block(&mut buffer);
        for sample in &buffer[..frames * 2] {
            let s = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer.write_sample(s)?;
        }
        remaining -= frames;
    }

    writer.finalize()?;
    let elapsed = start_time.elapsed().as_secs_f32();
    println!("Total generation time: {:.2}s", elapsed);
    Ok(())
}
