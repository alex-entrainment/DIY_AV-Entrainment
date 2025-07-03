use clap::{Parser, Subcommand, Args as ClapArgs};
use realtime_backend::models::TrackData;
use realtime_backend::scheduler::TrackScheduler;
use realtime_backend::command::Command;
use realtime_backend::audio_io;
use realtime_backend::config::{CONFIG, BackendConfig};
use ringbuf::HeapRb;
use ringbuf::traits::Split;
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
    let track_data: TrackData = serde_json::from_str(&json_str)?;

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
    scheduler.gpu_enabled = if args.gpu { true } else { CONFIG.gpu };
    let rb = HeapRb::<Command>::new(1024);
    let (_prod, cons) = rb.split();
    let (tx, rx) = unbounded();

    std::thread::spawn(move || {
        audio_io::run_audio_stream(scheduler, cons, rx);
    });

    println!("Streaming {}... press Ctrl+C to stop", args.path);
    ctrlc::set_handler(move || {
        let _ = tx.send(());
    })?;

    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

fn render_full_wav(
    track_data: TrackData,
    out_path: &str,
    gpu: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use hound::{WavSpec, WavWriter, SampleFormat};
    let sample_rate = track_data.global_settings.sample_rate;
    let mut scheduler = TrackScheduler::new(track_data.clone(), sample_rate);
    scheduler.gpu_enabled = if gpu { true } else { CONFIG.gpu };
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
