use std::env;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

use runtime::{TlspriteEditorTheme, TlspriteListDocument};

#[derive(Debug, Clone)]
struct Cli {
    file: PathBuf,
    markdown: bool,
    watch_ms: Option<u64>,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            file: PathBuf::from("docs/demos/tlapp/bounce_hud.tlsprite"),
            markdown: false,
            watch_ms: None,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = parse_cli()?;
    let theme = TlspriteEditorTheme::DarkLavender;
    let palette = theme.palette();
    println!(
        "tlsprite-list-editor | theme=dark_lavender | accent=rgba({:.2},{:.2},{:.2},{:.2})",
        palette.accent[0], palette.accent[1], palette.accent[2], palette.accent[3]
    );

    if let Some(ms) = cli.watch_ms {
        loop {
            render_once(&cli)?;
            thread::sleep(Duration::from_millis(ms.max(50)));
            println!("---");
        }
    } else {
        render_once(&cli)?;
    }
    Ok(())
}

fn render_once(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let document = TlspriteListDocument::from_path(&cli.file)?;
    if document.has_errors() {
        eprintln!(
            "compile: ERRORS={} WARNINGS={}",
            count_errors(document.diagnostics()),
            count_warnings(document.diagnostics())
        );
    } else {
        println!(
            "compile: ok | rows={} | warnings={}",
            document.rows().len(),
            count_warnings(document.diagnostics())
        );
    }

    if cli.markdown {
        println!("{}", document.to_markdown_table());
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
    Ok(())
}

fn parse_cli() -> Result<Cli, Box<dyn std::error::Error>> {
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
    println!("Tileline .tlsprite List Editor (Alpha scaffold)");
    println!("Usage: cargo run -p runtime --example tlsprite_list_editor -- [options]");
    println!("Options:");
    println!("  --file <path>       Input .tlsprite file (default: docs/demos/tlapp/bounce_hud.tlsprite)");
    println!("  --markdown          Emit list output as markdown table");
    println!("  --watch-ms <N>      Poll-reload file and print list every N ms");
    println!("  -h, --help          Show help");
}

fn count_errors(diags: &[runtime::TlspriteDiagnostic]) -> usize {
    diags
        .iter()
        .filter(|d| d.level == runtime::TlspriteDiagnosticLevel::Error)
        .count()
}

fn count_warnings(diags: &[runtime::TlspriteDiagnostic]) -> usize {
    diags
        .iter()
        .filter(|d| d.level == runtime::TlspriteDiagnosticLevel::Warning)
        .count()
}
