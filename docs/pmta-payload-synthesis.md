# PMTA Payload Synthesis

This note documents the exploratory decoder that lives in
`mps/examples/payload_bruteforce.rs`.

## Scope

The example targets `lu:pmta:<payload>` records that decode into fixed-size 80 byte blobs:

- `0..32`: header / nonce source
- `32..64`: ciphertext
- `64..80`: trailing tag / opaque flags

The current working path assumes:

- `AES-128-CTR split-key` style decryption
- HKDF-SHA256 key derivation from a pinned candidate key
- per-record flag-driven alignment and timestamp recovery

## Pipeline

1. Parse each raw line with `parse_strict_key_record`.
2. Preserve outer-record metadata:
   - `raw_length`
   - `normalized_length`
   - `has_expected_length`
   - `payload.pad_added`
3. Decrypt the payload with a fixed candidate key and nonce resync sweep.
4. Synthesize `t/u/c` from the 32-byte plaintext.
5. Export a consistency table and `payload_synthesis_report.csv`.

## Why The Strict-Length Flag Is Separate

The strict parser keeps record-shape validation separate from payload decoding.
In the current dataset, several lines miss the original `121/129/133` normalized
length targets but still decode to structurally valid 80-byte payloads. That is
treated as an outer-record variation, not automatic corruption.

## Line 11 Surgical Path

Line 11 required extra recovery passes after the baseline synthesis failed.
The surgical runner uses progressively more expensive matrices:

1. Segment jitter on the timestamp bytes
2. AES nonce bit-flip and counter drift
3. Ciphertext window sliding and flag-based key / nonce mutation
4. Coding matrix:
   - endian reinterpretation
   - nibble / BCD scan
   - inverted-byte mirror pass
   - relaxed date filter
   - Line 10 delta / counter interpretation

The final successful rule for Line 11 came from the coding matrix via a delta
interpretation relative to Line 10.

## Outputs

The example prints:

- a dataset-wide consistency table
- strict-length anomaly summary
- a focused surgical report for Line 11
- a CSV export at `payload_synthesis_report.csv`

## Operational Notes

- `key.txt`, `candidate_keys.txt`, and `payload_synthesis_report.csv` are runtime
  artifacts and should stay out of source control unless explicitly needed.
- `runtime/src/tlscript_parallel.rs` is unrelated to this workflow and should not
  be mixed into commits for the PMTA extraction branch.
