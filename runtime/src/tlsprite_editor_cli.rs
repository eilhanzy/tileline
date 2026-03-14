//! Runtime-owned CLI flow for list-mode `.tlsprite` editor usage.
//!
//! This keeps command behavior in `src/` so both binaries and examples can reuse the same logic.

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

use crate::{
    TlspriteDiagnostic, TlspriteDiagnosticLevel, TlspriteEditorTheme, TlspriteHotReloadConfig,
    TlspriteHotReloadEvent, TlspriteListDocument, TlspriteWatchConfig, TlspriteWatchReloader,
};

#[derive(Debug, Clone)]
struct Cli {
    file: PathBuf,
    markdown: bool,
    watch_ms: Option<u64>,
    strict: bool,
    strict_warnings: bool,
    write_markdown: Option<PathBuf>,
    init_if_missing: bool,
    clear_on_update: bool,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            file: PathBuf::from("docs/demos/tlapp/bounce_hud.tlsprite"),
            markdown: false,
            watch_ms: None,
            strict: false,
            strict_warnings: false,
            write_markdown: None,
            init_if_missing: false,
            clear_on_update: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RenderSummary {
    rows: usize,
    errors: usize,
    warnings: usize,
}

/// Run the `.tlsprite` editor CLI using process arguments.
pub fn run_from_env() -> Result<(), Box<dyn Error>> {
    let cli = parse_cli()?;
    ensure_input_exists(&cli)?;

    let theme = TlspriteEditorTheme::DarkLavender;
    let palette = theme.palette();
    println!(
        "tlsprite-editor | file={} | theme=dark_lavender | accent=rgba({:.2},{:.2},{:.2},{:.2})",
        cli.file.display(),
        palette.accent[0],
        palette.accent[1],
        palette.accent[2],
        palette.accent[3]
    );

    if let Some(ms) = cli.watch_ms {
        run_watch_loop(&cli, ms)?;
        return Ok(());
    }

    let summary = render_once(&cli)?;
    enforce_strict(&cli, summary)?;
    Ok(())
}

fn run_watch_loop(cli: &Cli, watch_ms: u64) -> Result<(), Box<dyn Error>> {
    let mut reloader = TlspriteWatchReloader::with_configs(
        cli.file.clone(),
        TlspriteHotReloadConfig::default(),
        TlspriteWatchConfig {
            prefer_notify_backend: true,
            poll_interval_ms: watch_ms.max(50),
        },
    );
    println!(
        "watch: backend={:?} | interval_ms={}",
        reloader.backend(),
        watch_ms.max(50)
    );
    if let Some(warning) = reloader.init_warning() {
        eprintln!("watch: {warning}");
    }

    loop {
        match reloader.reload_if_needed() {
            TlspriteHotReloadEvent::Unchanged => {
                thread::sleep(Duration::from_millis((watch_ms / 4).max(16)));
            }
            TlspriteHotReloadEvent::Applied {
                sprite_count,
                warning_count,
            } => {
                println!(
                    "watch-event: applied | sprites={} | warnings={}",
                    sprite_count, warning_count
                );
                let summary = render_once(cli)?;
                enforce_strict(cli, summary)?;
            }
            TlspriteHotReloadEvent::Rejected {
                error_count,
                warning_count,
                kept_last_program,
            } => {
                eprintln!(
                    "watch-event: rejected | errors={} | warnings={} | keep_last={}",
                    error_count, warning_count, kept_last_program
                );
                let summary = render_once(cli)?;
                enforce_strict(cli, summary)?;
            }
            TlspriteHotReloadEvent::SourceError { message } => {
                eprintln!("watch-event: source-error | {message}");
                thread::sleep(Duration::from_millis((watch_ms / 2).max(50)));
            }
        }
    }
}

fn render_once(cli: &Cli) -> Result<RenderSummary, Box<dyn Error>> {
    if cli.clear_on_update {
        // ANSI clear + cursor home for readable watch mode.
        print!("\x1b[2J\x1b[H");
    }

    let document = TlspriteListDocument::from_path(&cli.file)?;
    let errors = count_errors(document.diagnostics());
    let warnings = count_warnings(document.diagnostics());

    if errors > 0 {
        eprintln!("compile: ERRORS={errors} WARNINGS={warnings}");
    } else {
        println!(
            "compile: ok | rows={} | warnings={warnings}",
            document.rows().len()
        );
    }

    if cli.markdown || cli.write_markdown.is_some() {
        let table = document.to_markdown_table();
        if cli.markdown {
            println!("{table}");
        }
        if let Some(path) = &cli.write_markdown {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).map_err(|err| {
                    format!(
                        "failed to create markdown output directory '{}': {err}",
                        parent.display()
                    )
                })?;
            }
            fs::write(path, table).map_err(|err| {
                format!(
                    "failed to write markdown output '{}': {err}",
                    path.display()
                )
            })?;
            println!("markdown: wrote {}", path.display());
        }
    } else {
        for row in document.rows() {
            println!(
                "#{:>3} [{}] kind={} slot={} layer={} pos=({:.2},{:.2},{:.2}) size=({:.2},{:.2}) fbx={}",
                row.index,
                row.name,
                row.kind_label(),
                row.texture_slot,
                row.layer,
                row.position[0],
                row.position[1],
                row.position[2],
                row.size[0],
                row.size[1],
                row.fbx_source.as_deref().unwrap_or("-")
            );
        }
    }

    if !document.diagnostics().is_empty() {
        println!("diagnostics:");
        for diagnostic in document.diagnostics() {
            println!(
                "  - {:?} line {}: {}",
                diagnostic.level, diagnostic.line, diagnostic.message
            );
        }
    }

    Ok(RenderSummary {
        rows: document.rows().len(),
        errors,
        warnings,
    })
}

fn enforce_strict(cli: &Cli, summary: RenderSummary) -> Result<(), Box<dyn Error>> {
    if cli.strict_warnings && (summary.errors > 0 || summary.warnings > 0) {
        return Err(format!(
            "strict-warnings violation: rows={} errors={} warnings={}",
            summary.rows, summary.errors, summary.warnings
        )
        .into());
    }
    if cli.strict && summary.errors > 0 {
        return Err(format!(
            "strict violation: rows={} errors={} warnings={}",
            summary.rows, summary.errors, summary.warnings
        )
        .into());
    }
    Ok(())
}

fn ensure_input_exists(cli: &Cli) -> Result<(), Box<dyn Error>> {
    if cli.file.exists() {
        return Ok(());
    }
    if !cli.init_if_missing {
        return Err(format!(
            "input file '{}' does not exist (use --init-if-missing to create a starter file)",
            cli.file.display()
        )
        .into());
    }

    if let Some(parent) = cli.file.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create '{}': {err}", parent.display()))?;
    }
    fs::write(&cli.file, default_tlsprite_template()).map_err(|err| {
        format!(
            "failed to write starter tlsprite '{}': {err}",
            cli.file.display()
        )
    })?;
    println!("init: created starter file {}", cli.file.display());
    Ok(())
}

fn parse_cli() -> Result<Cli, Box<dyn Error>> {
    let mut cli = Cli::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--file" => {
                let value = args
                    .next()
                    .ok_or_else(|| String::from("missing value for --file"))?;
                cli.file = PathBuf::from(value);
            }
            "--markdown" => cli.markdown = true,
            "--watch-ms" => {
                let value = args
                    .next()
                    .ok_or_else(|| String::from("missing value for --watch-ms"))?;
                let parsed = value
                    .parse::<u64>()
                    .map_err(|_| format!("invalid --watch-ms value: {value}"))?;
                cli.watch_ms = Some(parsed.max(50));
            }
            "--strict" => cli.strict = true,
            "--strict-warnings" => cli.strict_warnings = true,
            "--write-markdown" => {
                let value = args
                    .next()
                    .ok_or_else(|| String::from("missing value for --write-markdown"))?;
                cli.write_markdown = Some(PathBuf::from(value));
            }
            "--init-if-missing" => cli.init_if_missing = true,
            "--clear" => cli.clear_on_update = true,
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
    println!("Tileline .tlsprite Editor (List Mode)");
    println!("Usage: cargo run -p runtime --bin tlsprite_editor -- [options]");
    println!("Options:");
    println!("  --file <path>            Input .tlsprite file (default: docs/demos/tlapp/bounce_hud.tlsprite)");
    println!("  --markdown               Print list output as markdown table");
    println!("  --write-markdown <path>  Write markdown table output to a file");
    println!("  --watch-ms <N>           Hot-reload with notify+poll backend (minimum 50 ms)");
    println!("  --clear                  Clear screen before each watch render");
    println!("  --strict                 Exit non-zero if compile has any error");
    println!("  --strict-warnings        Exit non-zero if compile has any warning/error");
    println!("  --init-if-missing        Create starter .tlsprite file when input is missing");
    println!("  -h, --help               Show help");
}

fn count_errors(diags: &[TlspriteDiagnostic]) -> usize {
    diags
        .iter()
        .filter(|d| d.level == TlspriteDiagnosticLevel::Error)
        .count()
}

fn count_warnings(diags: &[TlspriteDiagnostic]) -> usize {
    diags
        .iter()
        .filter(|d| d.level == TlspriteDiagnosticLevel::Warning)
        .count()
}

fn default_tlsprite_template() -> &'static str {
    "tlsprite_v1
[hud.default]
sprite_id = 1
kind = hud
texture_slot = 0
layer = 100
position = 0.0, 0.0, 0.0
size = 0.24, 0.06
rotation_rad = 0.0
color = 1.0, 1.0, 1.0, 1.0
"
}
