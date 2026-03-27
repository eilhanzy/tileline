# Runtime `.pak` Packaging (Pre-Beta)

`runtime` now includes a lightweight `.pak` archive implementation for deterministic asset packaging.

## Implementation

- `runtime/src/pak.rs`
- `runtime/examples/pak_tool.rs`

## Format

- Magic/version: `TLPAKV1\0`
- Header includes:
  - entry count
  - data offset/size
  - TOC offset/size
- Payload region stores file bytes concatenated in sorted path order.
- TOC stores per-entry:
  - UTF-8 relative path
  - payload offset
  - payload size
  - FNV-1a 64 checksum

## Safety

- Absolute paths are rejected.
- Parent-dir components (`..`) are rejected.
- Unpack validates checksums before writing files.

## API

- `create_pak_from_dir(source_dir, output_path)`
- `list_pak(archive_path)`
- `unpack_pak(archive_path, output_dir)`
- `read_file_from_pak(archive_path, entry_path)`

## CLI Example

Script wrapper:

```bash
./scripts/package_prebeta_pak.sh
```

Custom source/output:

```bash
./scripts/package_prebeta_pak.sh --src docs/demos --out dist/prebeta/custom-assets.pak
```

Pack:

```bash
cargo run -p runtime --example pak_tool -- \
  pack --src docs/demos --out dist/prebeta/tileline-assets-prebeta.pak
```

List:

```bash
cargo run -p runtime --example pak_tool -- \
  list --pak dist/prebeta/tileline-assets-prebeta.pak
```

Unpack:

```bash
cargo run -p runtime --example pak_tool -- \
  unpack --pak dist/prebeta/tileline-assets-prebeta.pak --out dist/prebeta/unpacked-assets
```

## `.tlpfile` Builder (5GB Numbered Shards)

For project-first packaging, use:

```bash
./scripts/build_tlpfile_pak.sh \
  --project docs/demos/tlapp/tlapp_project.tlpfile \
  --out-dir dist/pak \
  --base-name tlapp-assets \
  --max-shard-gb 5
```

This builder enforces:

- project root = directory that contains `.tlpfile`
- manifest paths must stay inside project root (no absolute paths, no `../`)
- numbered output shards:
  - `tlapp-assets.part0001.pak`
  - `tlapp-assets.part0002.pak`
  - ...
