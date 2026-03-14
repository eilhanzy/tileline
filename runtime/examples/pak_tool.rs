use std::env;
use std::path::PathBuf;

use runtime::{create_pak_from_dir, list_pak, unpack_pak};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        print_usage();
        return Ok(());
    };

    match command.as_str() {
        "pack" => {
            let mut src = PathBuf::from("docs/demos");
            let mut out = PathBuf::from("dist/prebeta/tileline-assets-prebeta.pak");
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--src" => {
                        src = PathBuf::from(
                            args.next()
                                .ok_or_else(|| String::from("missing value for --src"))?,
                        );
                    }
                    "--out" => {
                        out = PathBuf::from(
                            args.next()
                                .ok_or_else(|| String::from("missing value for --out"))?,
                        );
                    }
                    "-h" | "--help" => {
                        print_usage();
                        return Ok(());
                    }
                    other => {
                        return Err(format!("unknown argument for pack: {other}").into());
                    }
                }
            }

            let report = create_pak_from_dir(&src, &out)?;
            println!(
                "packed: {} files, {} bytes -> {}",
                report.file_count,
                report.total_payload_bytes,
                report.output_path.display()
            );
        }
        "list" => {
            let mut pak = PathBuf::from("dist/prebeta/tileline-assets-prebeta.pak");
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--pak" => {
                        pak = PathBuf::from(
                            args.next()
                                .ok_or_else(|| String::from("missing value for --pak"))?,
                        );
                    }
                    "-h" | "--help" => {
                        print_usage();
                        return Ok(());
                    }
                    other => {
                        return Err(format!("unknown argument for list: {other}").into());
                    }
                }
            }

            let index = list_pak(&pak)?;
            println!(
                "pak: {} | files={} | data_bytes={} | toc_bytes={}",
                pak.display(),
                index.entries.len(),
                index.data_size,
                index.toc_size
            );
            for entry in &index.entries {
                println!(
                    "- {} ({} bytes, checksum {:016x})",
                    entry.path, entry.size, entry.checksum_fnv64
                );
            }
        }
        "unpack" => {
            let mut pak = PathBuf::from("dist/prebeta/tileline-assets-prebeta.pak");
            let mut out = PathBuf::from("dist/prebeta/unpacked-assets");
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--pak" => {
                        pak = PathBuf::from(
                            args.next()
                                .ok_or_else(|| String::from("missing value for --pak"))?,
                        );
                    }
                    "--out" => {
                        out = PathBuf::from(
                            args.next()
                                .ok_or_else(|| String::from("missing value for --out"))?,
                        );
                    }
                    "-h" | "--help" => {
                        print_usage();
                        return Ok(());
                    }
                    other => {
                        return Err(format!("unknown argument for unpack: {other}").into());
                    }
                }
            }

            let report = unpack_pak(&pak, &out)?;
            println!(
                "unpacked: {} files, {} bytes -> {}",
                report.file_count,
                report.total_payload_bytes,
                report.output_dir.display()
            );
        }
        "-h" | "--help" => {
            print_usage();
        }
        other => {
            return Err(format!("unknown command: {other}").into());
        }
    }

    Ok(())
}

fn print_usage() {
    println!("Tileline .pak Tool (Pre-Beta)");
    println!("Usage:");
    println!(
        "  cargo run -p runtime --example pak_tool -- pack   [--src <dir>] [--out <file.pak>]"
    );
    println!("  cargo run -p runtime --example pak_tool -- list   [--pak <file.pak>]");
    println!(
        "  cargo run -p runtime --example pak_tool -- unpack [--pak <file.pak>] [--out <dir>]"
    );
}
