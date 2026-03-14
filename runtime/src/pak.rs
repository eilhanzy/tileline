//! Tileline `.pak` archive support (Pre-Beta scaffold).
//!
//! Goals:
//! - deterministic pack order
//! - simple binary format (no external dependencies)
//! - safe unpack path handling
//! - checksum validation for asset integrity

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};

const PAK_MAGIC: [u8; 8] = *b"TLPAKV1\0";
const PAK_HEADER_SIZE: usize = 56;

/// One archive index entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PakEntry {
    pub path: String,
    pub offset: u64,
    pub size: u64,
    pub checksum_fnv64: u64,
}

/// Header/TOC summary for a loaded `.pak`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PakIndex {
    pub entries: Vec<PakEntry>,
    pub data_offset: u64,
    pub data_size: u64,
    pub toc_offset: u64,
    pub toc_size: u64,
}

/// Build report for packing operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PakBuildReport {
    pub source_dir: PathBuf,
    pub output_path: PathBuf,
    pub file_count: usize,
    pub total_payload_bytes: u64,
}

/// Unpack report for extraction operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PakUnpackReport {
    pub archive_path: PathBuf,
    pub output_dir: PathBuf,
    pub file_count: usize,
    pub total_payload_bytes: u64,
}

#[derive(Debug, Clone)]
struct BuildEntry {
    relative_path: String,
    payload: Vec<u8>,
    checksum_fnv64: u64,
}

/// Build a deterministic `.pak` archive from a source directory.
pub fn create_pak_from_dir(
    source_dir: &Path,
    output_path: &Path,
) -> Result<PakBuildReport, String> {
    if !source_dir.exists() {
        return Err(format!(
            "source directory does not exist: {}",
            source_dir.display()
        ));
    }
    if !source_dir.is_dir() {
        return Err(format!(
            "source path is not a directory: {}",
            source_dir.display()
        ));
    }

    let mut entries = collect_files(source_dir, source_dir)?;
    entries.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));

    let output_parent = output_path
        .parent()
        .ok_or_else(|| format!("invalid output path: {}", output_path.display()))?;
    fs::create_dir_all(output_parent).map_err(|err| {
        format!(
            "failed to create output parent '{}': {err}",
            output_parent.display()
        )
    })?;

    let mut out = File::create(output_path)
        .map_err(|err| format!("failed to create pak '{}': {err}", output_path.display()))?;
    write_header_placeholder(&mut out)?;

    let data_offset = PAK_HEADER_SIZE as u64;
    let mut running_offset = 0u64;
    let mut toc_entries = Vec::with_capacity(entries.len());
    let mut total_payload_bytes = 0u64;

    for entry in &entries {
        out.write_all(&entry.payload)
            .map_err(|err| format!("failed to write payload '{}': {err}", entry.relative_path))?;
        let size = entry.payload.len() as u64;
        toc_entries.push(PakEntry {
            path: entry.relative_path.clone(),
            offset: running_offset,
            size,
            checksum_fnv64: entry.checksum_fnv64,
        });
        running_offset = running_offset.saturating_add(size);
        total_payload_bytes = total_payload_bytes.saturating_add(size);
    }

    let toc_offset = data_offset.saturating_add(total_payload_bytes);
    let toc_size = write_toc(&mut out, &toc_entries)?;
    let data_size = total_payload_bytes;
    write_header(
        &mut out,
        toc_entries.len() as u32,
        data_offset,
        data_size,
        toc_offset,
        toc_size,
    )?;

    Ok(PakBuildReport {
        source_dir: source_dir.to_path_buf(),
        output_path: output_path.to_path_buf(),
        file_count: toc_entries.len(),
        total_payload_bytes,
    })
}

/// Load archive index (TOC) for listing/inspection.
pub fn list_pak(archive_path: &Path) -> Result<PakIndex, String> {
    let mut file = File::open(archive_path)
        .map_err(|err| format!("failed to open pak '{}': {err}", archive_path.display()))?;
    let (entry_count, data_offset, data_size, toc_offset, toc_size) = read_header(&mut file)?;

    file.seek(SeekFrom::Start(toc_offset))
        .map_err(|err| format!("failed to seek TOC in '{}': {err}", archive_path.display()))?;
    let entries = read_toc(&mut file, entry_count as usize, toc_size)?;

    Ok(PakIndex {
        entries,
        data_offset,
        data_size,
        toc_offset,
        toc_size,
    })
}

/// Extract all archive entries into output directory with checksum verification.
pub fn unpack_pak(archive_path: &Path, output_dir: &Path) -> Result<PakUnpackReport, String> {
    let index = list_pak(archive_path)?;
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create output directory '{}': {err}",
            output_dir.display()
        )
    })?;

    let mut file = File::open(archive_path)
        .map_err(|err| format!("failed to open pak '{}': {err}", archive_path.display()))?;

    let mut total = 0u64;
    for entry in &index.entries {
        ensure_safe_relative_path(&entry.path)?;
        file.seek(SeekFrom::Start(
            index.data_offset.saturating_add(entry.offset),
        ))
        .map_err(|err| {
            format!(
                "failed to seek payload '{}' in '{}': {err}",
                entry.path,
                archive_path.display()
            )
        })?;
        let mut payload = vec![0u8; entry.size as usize];
        file.read_exact(&mut payload).map_err(|err| {
            format!(
                "failed reading payload '{}' in '{}': {err}",
                entry.path,
                archive_path.display()
            )
        })?;
        let checksum = fnv1a64(&payload);
        if checksum != entry.checksum_fnv64 {
            return Err(format!(
                "checksum mismatch for '{}': expected {:016x}, got {:016x}",
                entry.path, entry.checksum_fnv64, checksum
            ));
        }

        let out_path = output_dir.join(&entry.path);
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed creating output directory '{}': {err}",
                    parent.display()
                )
            })?;
        }
        fs::write(&out_path, &payload).map_err(|err| {
            format!(
                "failed writing unpacked file '{}': {err}",
                out_path.display()
            )
        })?;
        total = total.saturating_add(entry.size);
    }

    Ok(PakUnpackReport {
        archive_path: archive_path.to_path_buf(),
        output_dir: output_dir.to_path_buf(),
        file_count: index.entries.len(),
        total_payload_bytes: total,
    })
}

/// Read one file payload from archive by exact relative path.
pub fn read_file_from_pak(archive_path: &Path, entry_path: &str) -> Result<Vec<u8>, String> {
    let index = list_pak(archive_path)?;
    let Some(entry) = index.entries.iter().find(|entry| entry.path == entry_path) else {
        return Err(format!(
            "entry '{}' not found in '{}'",
            entry_path,
            archive_path.display()
        ));
    };

    let mut file = File::open(archive_path)
        .map_err(|err| format!("failed to open pak '{}': {err}", archive_path.display()))?;
    file.seek(SeekFrom::Start(
        index.data_offset.saturating_add(entry.offset),
    ))
    .map_err(|err| format!("failed to seek entry '{}': {err}", entry.path))?;
    let mut payload = vec![0u8; entry.size as usize];
    file.read_exact(&mut payload)
        .map_err(|err| format!("failed to read entry '{}': {err}", entry.path))?;
    let checksum = fnv1a64(&payload);
    if checksum != entry.checksum_fnv64 {
        return Err(format!(
            "checksum mismatch for '{}': expected {:016x}, got {:016x}",
            entry.path, entry.checksum_fnv64, checksum
        ));
    }
    Ok(payload)
}

fn collect_files(base_dir: &Path, cursor: &Path) -> Result<Vec<BuildEntry>, String> {
    let mut out = Vec::new();
    let entries = fs::read_dir(cursor)
        .map_err(|err| format!("failed to read directory '{}': {err}", cursor.display()))?;
    for entry in entries {
        let entry = entry.map_err(|err| {
            format!(
                "failed reading directory entry '{}': {err}",
                cursor.display()
            )
        })?;
        let path = entry.path();
        let metadata = entry
            .metadata()
            .map_err(|err| format!("failed to stat '{}': {err}", path.display()))?;
        if metadata.is_dir() {
            out.extend(collect_files(base_dir, &path)?);
            continue;
        }
        if !metadata.is_file() {
            continue;
        }

        let rel = path.strip_prefix(base_dir).map_err(|_| {
            format!(
                "failed to strip base '{}' from '{}'",
                base_dir.display(),
                path.display()
            )
        })?;
        let rel_str = normalize_rel_path(rel)?;
        ensure_safe_relative_path(&rel_str)?;
        let payload = fs::read(&path)
            .map_err(|err| format!("failed reading input file '{}': {err}", path.display()))?;
        let checksum_fnv64 = fnv1a64(&payload);
        out.push(BuildEntry {
            relative_path: rel_str,
            payload,
            checksum_fnv64,
        });
    }
    Ok(out)
}

fn normalize_rel_path(path: &Path) -> Result<String, String> {
    let mut out = String::new();
    for (i, component) in path.components().enumerate() {
        if i > 0 {
            out.push('/');
        }
        let Component::Normal(value) = component else {
            return Err(format!(
                "invalid relative path component in '{}'",
                path.display()
            ));
        };
        let value = value
            .to_str()
            .ok_or_else(|| format!("non-utf8 path component in '{}'", path.display()))?;
        out.push_str(value);
    }
    if out.is_empty() {
        return Err("empty relative path".to_string());
    }
    Ok(out)
}

fn ensure_safe_relative_path(path: &str) -> Result<(), String> {
    let candidate = Path::new(path);
    if candidate.is_absolute() {
        return Err(format!("absolute path is not allowed in pak entry: {path}"));
    }
    for component in candidate.components() {
        match component {
            Component::ParentDir => {
                return Err(format!(
                    "parent-dir component is not allowed in pak entry: {path}"
                ));
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(format!(
                    "root/prefix component is not allowed in pak entry: {path}"
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn write_header_placeholder(out: &mut File) -> Result<(), String> {
    let zeros = [0u8; PAK_HEADER_SIZE];
    out.write_all(&zeros)
        .map_err(|err| format!("failed writing pak header placeholder: {err}"))
}

fn write_header(
    out: &mut File,
    entry_count: u32,
    data_offset: u64,
    data_size: u64,
    toc_offset: u64,
    toc_size: u64,
) -> Result<(), String> {
    out.seek(SeekFrom::Start(0))
        .map_err(|err| format!("failed seeking pak header start: {err}"))?;

    let mut header = Vec::with_capacity(PAK_HEADER_SIZE);
    header.extend_from_slice(&PAK_MAGIC);
    header.extend_from_slice(&entry_count.to_le_bytes());
    header.extend_from_slice(&0u32.to_le_bytes()); // reserved
    header.extend_from_slice(&data_offset.to_le_bytes());
    header.extend_from_slice(&data_size.to_le_bytes());
    header.extend_from_slice(&toc_offset.to_le_bytes());
    header.extend_from_slice(&toc_size.to_le_bytes());
    header.extend_from_slice(&0u64.to_le_bytes()); // reserved
    if header.len() != PAK_HEADER_SIZE {
        return Err(format!(
            "internal header size mismatch: expected {PAK_HEADER_SIZE}, got {}",
            header.len()
        ));
    }
    out.write_all(&header)
        .map_err(|err| format!("failed writing pak header: {err}"))
}

fn read_header(file: &mut File) -> Result<(u32, u64, u64, u64, u64), String> {
    let mut buf = [0u8; PAK_HEADER_SIZE];
    file.seek(SeekFrom::Start(0))
        .map_err(|err| format!("failed seeking pak header: {err}"))?;
    file.read_exact(&mut buf)
        .map_err(|err| format!("failed reading pak header: {err}"))?;

    if buf[..8] != PAK_MAGIC {
        return Err("invalid pak magic/version".to_string());
    }

    let entry_count = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    let data_offset = u64::from_le_bytes([
        buf[16], buf[17], buf[18], buf[19], buf[20], buf[21], buf[22], buf[23],
    ]);
    let data_size = u64::from_le_bytes([
        buf[24], buf[25], buf[26], buf[27], buf[28], buf[29], buf[30], buf[31],
    ]);
    let toc_offset = u64::from_le_bytes([
        buf[32], buf[33], buf[34], buf[35], buf[36], buf[37], buf[38], buf[39],
    ]);
    let toc_size = u64::from_le_bytes([
        buf[40], buf[41], buf[42], buf[43], buf[44], buf[45], buf[46], buf[47],
    ]);

    Ok((entry_count, data_offset, data_size, toc_offset, toc_size))
}

fn write_toc(out: &mut File, entries: &[PakEntry]) -> Result<u64, String> {
    let mut bytes_written = 0u64;
    for entry in entries {
        let path_bytes = entry.path.as_bytes();
        if path_bytes.len() > u16::MAX as usize {
            return Err(format!(
                "path too long for pak entry '{}': {} bytes",
                entry.path,
                path_bytes.len()
            ));
        }
        out.write_all(&(path_bytes.len() as u16).to_le_bytes())
            .map_err(|err| format!("failed writing TOC path length '{}': {err}", entry.path))?;
        out.write_all(path_bytes)
            .map_err(|err| format!("failed writing TOC path '{}': {err}", entry.path))?;
        out.write_all(&entry.offset.to_le_bytes())
            .map_err(|err| format!("failed writing TOC offset '{}': {err}", entry.path))?;
        out.write_all(&entry.size.to_le_bytes())
            .map_err(|err| format!("failed writing TOC size '{}': {err}", entry.path))?;
        out.write_all(&entry.checksum_fnv64.to_le_bytes())
            .map_err(|err| format!("failed writing TOC checksum '{}': {err}", entry.path))?;
        bytes_written = bytes_written
            .saturating_add(2)
            .saturating_add(path_bytes.len() as u64)
            .saturating_add(8 + 8 + 8);
    }
    Ok(bytes_written)
}

fn read_toc(file: &mut File, entry_count: usize, toc_size: u64) -> Result<Vec<PakEntry>, String> {
    let mut entries = Vec::with_capacity(entry_count);
    let toc_start = file
        .stream_position()
        .map_err(|err| format!("failed reading TOC stream position: {err}"))?;
    for _ in 0..entry_count {
        let mut len_buf = [0u8; 2];
        file.read_exact(&mut len_buf)
            .map_err(|err| format!("failed reading TOC path length: {err}"))?;
        let path_len = u16::from_le_bytes(len_buf) as usize;
        let mut path_buf = vec![0u8; path_len];
        file.read_exact(&mut path_buf)
            .map_err(|err| format!("failed reading TOC path bytes: {err}"))?;
        let path = String::from_utf8(path_buf)
            .map_err(|err| format!("invalid utf8 path in TOC: {err}"))?;

        let mut u64_buf = [0u8; 8];
        file.read_exact(&mut u64_buf)
            .map_err(|err| format!("failed reading TOC offset for '{path}': {err}"))?;
        let offset = u64::from_le_bytes(u64_buf);
        file.read_exact(&mut u64_buf)
            .map_err(|err| format!("failed reading TOC size for '{path}': {err}"))?;
        let size = u64::from_le_bytes(u64_buf);
        file.read_exact(&mut u64_buf)
            .map_err(|err| format!("failed reading TOC checksum for '{path}': {err}"))?;
        let checksum_fnv64 = u64::from_le_bytes(u64_buf);

        entries.push(PakEntry {
            path,
            offset,
            size,
            checksum_fnv64,
        });
    }

    let toc_end = file
        .stream_position()
        .map_err(|err| format!("failed reading TOC end position: {err}"))?;
    let consumed = toc_end.saturating_sub(toc_start);
    if consumed != toc_size {
        return Err(format!(
            "TOC size mismatch: header says {toc_size} bytes, parsed {consumed} bytes"
        ));
    }
    Ok(entries)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x00000100000001B3;

    let mut hash = OFFSET_BASIS;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("tileline-pak-{name}-{nonce}"));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn packs_lists_and_unpacks_directory() {
        let src_dir = unique_temp_dir("src");
        let nested = src_dir.join("nested");
        std::fs::create_dir_all(&nested).expect("create nested");
        std::fs::write(src_dir.join("hello.txt"), b"hello world").expect("write hello");
        std::fs::write(nested.join("data.bin"), [1u8, 2, 3, 4, 5]).expect("write data");

        let pak_path = unique_temp_dir("out").join("demo.pak");
        let build = create_pak_from_dir(&src_dir, &pak_path).expect("build pak");
        assert_eq!(build.file_count, 2);

        let index = list_pak(&pak_path).expect("list pak");
        assert_eq!(index.entries.len(), 2);
        assert!(index.entries[0].path <= index.entries[1].path);

        let read_hello = read_file_from_pak(&pak_path, "hello.txt").expect("read hello");
        assert_eq!(read_hello, b"hello world");

        let out_dir = unique_temp_dir("extract");
        let unpack = unpack_pak(&pak_path, &out_dir).expect("unpack pak");
        assert_eq!(unpack.file_count, 2);
        assert_eq!(
            std::fs::read(out_dir.join("hello.txt")).expect("read unpacked hello"),
            b"hello world"
        );
        assert_eq!(
            std::fs::read(out_dir.join("nested/data.bin")).expect("read unpacked data"),
            vec![1u8, 2, 3, 4, 5]
        );
    }
}
