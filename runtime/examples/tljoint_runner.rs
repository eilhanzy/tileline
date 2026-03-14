use std::env;
use std::path::PathBuf;

use runtime::{
    compile_tljoint_scene_from_path, TlscriptShowcaseConfig, TlscriptShowcaseControlInput,
    TlscriptShowcaseFrameInput,
};

#[derive(Debug, Clone)]
struct Cli {
    joint_path: PathBuf,
    scene: String,
    frames: u64,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            joint_path: PathBuf::from("docs/demos/tlapp/bounce_showcase.tljoint"),
            scene: "main".to_string(),
            frames: 8,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = parse_cli()?;
    let outcome = compile_tljoint_scene_from_path(
        &cli.joint_path,
        &cli.scene,
        TlscriptShowcaseConfig::default(),
    );

    if !outcome.diagnostics.is_empty() {
        println!("diagnostics:");
        for d in &outcome.diagnostics {
            println!("  - {:?} line {}: {}", d.level, d.line, d.message);
        }
    }

    let Some(bundle) = outcome.bundle else {
        return Err("failed to compile .tljoint scene bundle".into());
    };

    println!(
        "tljoint: {} | scene={} | scripts={} | sprites={} | merged_sprite={}",
        bundle.manifest_path.display(),
        bundle.scene_name,
        bundle.scripts.len(),
        bundle.sprite_paths.len(),
        bundle
            .merged_sprite_program
            .as_ref()
            .map(|p| p.sprites().len())
            .unwrap_or(0)
    );

    for frame_index in 0..cli.frames {
        let frame = bundle.evaluate_frame(
            TlscriptShowcaseFrameInput {
                frame_index,
                live_balls: (frame_index as usize) * 120,
                spawned_this_tick: 64,
                key_f_down: false,
            },
            TlscriptShowcaseControlInput::default(),
        );
        println!(
            "frame={:>4} spawn_per_tick={:?} target_ball_count={:?} gravity={:?}",
            frame_index,
            frame.patch.spawn_per_tick,
            frame.patch.target_ball_count,
            frame.patch.gravity
        );
    }

    Ok(())
}

fn parse_cli() -> Result<Cli, Box<dyn std::error::Error>> {
    let mut cli = Cli::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--joint" => {
                let value = args
                    .next()
                    .ok_or_else(|| String::from("missing value for --joint"))?;
                cli.joint_path = PathBuf::from(value);
            }
            "--scene" => {
                let value = args
                    .next()
                    .ok_or_else(|| String::from("missing value for --scene"))?;
                cli.scene = value;
            }
            "--frames" => {
                let value = args
                    .next()
                    .ok_or_else(|| String::from("missing value for --frames"))?;
                cli.frames = value
                    .parse::<u64>()
                    .map_err(|_| format!("invalid --frames value: {value}"))?;
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg '{other}'").into()),
        }
    }
    Ok(cli)
}

fn print_usage() {
    println!("Tileline .tljoint Runner");
    println!("Usage: cargo run -p runtime --example tljoint_runner -- [options]");
    println!("Options:");
    println!("  --joint <path>     .tljoint manifest path (default: docs/demos/tlapp/bounce_showcase.tljoint)");
    println!("  --scene <name>     scene id in manifest (default: main)");
    println!("  --frames <N>       number of demo frames to evaluate (default: 8)");
    println!("  -h, --help         show help");
}
