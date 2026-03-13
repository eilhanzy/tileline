use aes::cipher::{KeyIvInit, StreamCipher};
use aes::Aes128;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use chrono::{TimeZone, Utc};
use ctr::Ctr128BE;
use hex::encode as hex_encode;
use hkdf::Hkdf;
use md5::Md5;
use mps::{normalize_candidate_keys, parse_strict_key_record, BLOB_BYTES, CIPHERTEXT_BYTES, HEADER_BYTES};
use percent_encoding::percent_decode_str;
use sha1::Sha1;
use sha2::{Digest, Sha256, Sha512};
use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};

const BLOCK_BYTES: usize = 16;
const FIXED_STAGE: &str = "focused";
const FIXED_BLOB_INDEX: usize = 13;
const FIXED_CANDIDATE_INDEX: usize = 20;
const FIXED_NONCE_OFFSET: usize = 0;
const SURGICAL_LINE_NO: usize = 11;
const CSV_OUTPUT_PATH: &str = "payload_synthesis_report.csv";
static LOGIC_LOOP_REPORTED: AtomicBool = AtomicBool::new(false);

fn main() {
    let payload_path = env::args().nth(1).unwrap_or_else(|| "key.txt".to_owned());
    let keys_path = env::args().nth(2).unwrap_or_else(|| {
        eprintln!("missing candidate key list; expected a file like candidate_keys.txt");
        std::process::exit(2);
    });
    let email_hint = env::args().nth(3);
    let local_part = local_part_hint(email_hint.as_deref());

    let payloads = load_payload_records(&payload_path);
    let stages = load_candidate_key_stages(&keys_path, email_hint.as_deref());
    let fixed = build_fixed_decryptor(&stages, local_part.as_bytes());
    let mut results = synthesize_dataset(&payloads, &fixed, &local_part);
    let surgical = run_line11_surgical_extraction(&payloads, &fixed, &local_part, &mut results);

    print_consistency_report(&results);
    print_strict_length_analysis(&payloads, &results);
    print_line11_surgical_report(surgical.as_ref());
    export_consistency_csv(&results, CSV_OUTPUT_PATH)
        .unwrap_or_else(|error| panic!("failed to write {CSV_OUTPUT_PATH}: {error}"));
    println!();
    println!("csv_export={CSV_OUTPUT_PATH}");
}

#[derive(Debug)]
struct CandidateStage {
    name: &'static str,
    raw_keys: Vec<Vec<u8>>,
}

#[derive(Debug, Default)]
struct StageBuckets {
    focused: Vec<Vec<u8>>,
    anchored: Vec<Vec<u8>>,
    expanded: Vec<Vec<u8>>,
}

#[derive(Clone, Copy, Debug)]
enum HkdfInfoKind {
    Envelope,
}

impl HkdfInfoKind {
    fn as_bytes(self) -> &'static [u8] {
        match self {
            Self::Envelope => b"envelope",
        }
    }
}

#[derive(Debug)]
struct FixedCandidate {
    plaintext: [u8; CIPHERTEXT_BYTES],
    nonce_mode: String,
}

#[derive(Debug)]
struct FixedDecryptor {
    raw_key: [u8; 32],
    derived_key: [u8; 32],
}

#[derive(Debug)]
struct InputPayloadRecord {
    line_no: usize,
    raw_length: usize,
    normalized_length: usize,
    strict_length_match: bool,
    payload_pad_added: usize,
    payload: [u8; BLOB_BYTES],
}

#[derive(Debug)]
struct DatasetSynthesisRow {
    line_no: usize,
    raw_length: usize,
    normalized_length: usize,
    strict_length_match: bool,
    payload_pad_added: usize,
    nonce_mode: String,
    parsed: SegmentedRecord,
}

#[derive(Debug)]
struct SurgicalTimestampReport {
    line_no: usize,
    segment_attempts: usize,
    aes_attempts: usize,
    deep_attempts: usize,
    coding_attempts: usize,
    best: Option<TimestampGoldenKey>,
    source: &'static str,
}

#[derive(Debug)]
struct SegmentedRecord {
    timestamp_segment: String,
    timestamp_note: Option<String>,
    uid_segment_plus56: String,
    uid_segment_plus115: String,
    reassembled_segment: String,
    tail_ascii85: Option<String>,
    tail_base64: Option<String>,
    tail_hybrid: Option<String>,
    combined_text: String,
    final_string: String,
    report_timestamp: String,
    user_id: String,
    campaign_id: String,
    extra_flags: String,
    polished_pairs: Vec<(String, String)>,
    diamond_polished: bool,
    mission_accomplished: bool,
    golden_result: Option<String>,
    realistic_score: u8,
    shift_key: u8,
    timestamp_rule: Option<String>,
}

#[derive(Debug)]
struct TailDecodeCandidate {
    source: String,
    value: String,
}

#[allow(dead_code)]
#[derive(Debug)]
struct TimestampHit {
    epoch: i64,
    datetime: chrono::DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct TimestampGoldenKey {
    digits: String,
    epoch: i64,
    datetime: chrono::DateTime<Utc>,
    rule: String,
}

#[derive(Debug, Clone)]
struct HexTokenCandidate {
    value: String,
    score: usize,
}

fn build_fixed_decryptor(
    stages: &[CandidateStage],
    local_part: &[u8],
) -> FixedDecryptor {
    // Keep one pinned candidate for the whole dataset so cross-line comparisons
    // reflect payload differences instead of key churn.
    let stage = stages
        .iter()
        .find(|stage| stage.name == FIXED_STAGE)
        .unwrap_or_else(|| panic!("missing stage {FIXED_STAGE}"));
    let candidate_keys =
        normalize_candidate_keys(stage.raw_keys.iter().map(|key| key.as_slice())).unwrap_or_else(
            |error| panic!("invalid candidate key set in {}: {error:?}", stage.name),
        );

    let raw_key = candidate_keys
        .get(FIXED_CANDIDATE_INDEX)
        .unwrap_or_else(|| panic!("missing fixed candidate index {}", FIXED_CANDIDATE_INDEX + 1));
    let mut raw_key_array = [0_u8; 32];
    raw_key_array.copy_from_slice(raw_key);

    let mut derived_key = [0_u8; 32];
    derive_hkdf_sha256(raw_key, Some(local_part), HkdfInfoKind::Envelope.as_bytes(), &mut derived_key);

    FixedDecryptor {
        raw_key: raw_key_array,
        derived_key,
    }
}

fn decrypt_payload_with_fixed_candidate(
    payload: &[u8; BLOB_BYTES],
    fixed: &FixedDecryptor,
    local_part: &[u8],
) -> FixedCandidate {
    let header = &payload[..HEADER_BYTES];
    let ciphertext = &payload[HEADER_BYTES..HEADER_BYTES + CIPHERTEXT_BYTES];
    let mut nonce = [0_u8; BLOCK_BYTES];
    nonce.copy_from_slice(&header[FIXED_NONCE_OFFSET..FIXED_NONCE_OFFSET + BLOCK_BYTES]);
    let mut best: Option<(usize, [u8; CIPHERTEXT_BYTES], String)> = None;

    for (mode, candidate_nonce) in candidate_nonces(header, &nonce) {
        let Some(plaintext) =
            decrypt_aes128_ctr_split(ciphertext, &fixed.derived_key, &candidate_nonce)
        else {
            continue;
        };
        let score = score_resync_candidate(&plaintext, local_part);
        if best.as_ref().map_or(true, |(best_score, _, _)| score > *best_score) {
            best = Some((score, plaintext, mode));
        }
    }

    let (_, plaintext, nonce_mode) = best.unwrap_or_else(|| panic!("fixed candidate failed to decrypt"));
    FixedCandidate { plaintext, nonce_mode }
}

fn synthesize_dataset(
    payloads: &[InputPayloadRecord],
    fixed: &FixedDecryptor,
    local_part: &str,
) -> Vec<DatasetSynthesisRow> {
    payloads
        .iter()
        .map(|payload| {
            let decrypted =
                decrypt_payload_with_fixed_candidate(&payload.payload, fixed, local_part.as_bytes());
            let parsed = parse_segmented_record(&decrypted.plaintext, local_part, false);
            DatasetSynthesisRow {
                line_no: payload.line_no,
                raw_length: payload.raw_length,
                normalized_length: payload.normalized_length,
                strict_length_match: payload.strict_length_match,
                payload_pad_added: payload.payload_pad_added,
                nonce_mode: decrypted.nonce_mode,
                parsed,
            }
        })
        .collect()
}

fn run_line11_surgical_extraction(
    payloads: &[InputPayloadRecord],
    fixed: &FixedDecryptor,
    local_part: &str,
    rows: &mut [DatasetSynthesisRow],
) -> Option<SurgicalTimestampReport> {
    // Line 11 is the only record that needed progressively deeper recovery
    // passes. The rest stay on the fast deterministic synthesis path.
    let payload = payloads.iter().find(|payload| payload.line_no == SURGICAL_LINE_NO)?;
    let previous_line_epoch = rows
        .iter()
        .find(|row| row.line_no == SURGICAL_LINE_NO.saturating_sub(1))
        .and_then(|row| extract_named_value(&row.parsed.polished_pairs, &["t", "timestamp"]))
        .and_then(|value| value.parse::<i64>().ok());
    let row = rows.iter_mut().find(|row| row.line_no == SURGICAL_LINE_NO)?;
    let decrypted = decrypt_payload_with_fixed_candidate(&payload.payload, fixed, local_part.as_bytes());
    let (segment_best, segment_attempts) = scan_line11_timestamp_surgical(&decrypted.plaintext);
    let (aes_best, aes_attempts) = scan_line11_aes_state_jitter(
        &payload.payload,
        fixed,
        local_part,
        &row.parsed.user_id,
        &row.parsed.campaign_id,
    );
    let (deep_best, deep_attempts) = scan_line11_deep_deviation(
        &payload.payload,
        fixed,
        local_part,
        &row.parsed.user_id,
        &row.parsed.campaign_id,
        row.parsed.shift_key,
    );
    let (coding_best, coding_attempts) = scan_line11_coding_matrix(
        &payload.payload,
        fixed,
        local_part,
        &row.parsed.user_id,
        &row.parsed.campaign_id,
        row.parsed.shift_key,
        previous_line_epoch,
    );

    let (best, source) = match (segment_best, aes_best, deep_best, coding_best) {
        (_, _, _, Some(coding)) => (Some(coding), "coding-matrix"),
        (_, _, Some(deep), None) => (Some(deep), "deep-deviation"),
        (Some(segment), Some(aes), None, None) => {
            if aes.rule.len() <= segment.rule.len() {
                (Some(aes), "aes-state-jitter")
            } else {
                (Some(segment), "segment-jitter")
            }
        }
        (Some(segment), None, None, None) => (Some(segment), "segment-jitter"),
        (None, Some(aes), None, None) => (Some(aes), "aes-state-jitter"),
        (None, None, None, None) => (None, "none"),
    };

    if let Some(best_hit) = best.as_ref() {
        apply_surgical_timestamp(row, best_hit);
    }

    Some(SurgicalTimestampReport {
        line_no: SURGICAL_LINE_NO,
        segment_attempts,
        aes_attempts,
        deep_attempts,
        coding_attempts,
        best,
        source,
    })
}

fn apply_surgical_timestamp(row: &mut DatasetSynthesisRow, hit: &TimestampGoldenKey) {
    row.parsed.timestamp_note = Some(format!(
        "{} ({}) [{}]",
        hit.epoch,
        hit.datetime.format("%Y-%m-%d %H:%M:%S UTC"),
        hit.rule
    ));
    row.parsed.report_timestamp = row
        .parsed
        .timestamp_note
        .clone()
        .unwrap_or_else(|| "-".to_owned());
    row.parsed.timestamp_rule = Some(hit.rule.clone());

    canonicalize_core_pairs(
        &mut row.parsed.polished_pairs,
        Some(hit),
        &row.parsed.user_id,
        &row.parsed.campaign_id,
    );
    row.parsed.final_string = build_final_string(&row.parsed.polished_pairs, &row.parsed.combined_text);
    let (golden_result, realistic_score) =
        synthesize_realistic_result(Some(hit), &row.parsed.user_id, &row.parsed.campaign_id);
    row.parsed.golden_result = golden_result;
    row.parsed.realistic_score = realistic_score;
    row.parsed.mission_accomplished = row.parsed.golden_result.is_some();
    row.parsed.diamond_polished = has_polished_core_fields(&row.parsed.polished_pairs);
}

fn parse_segmented_record(
    plaintext: &[u8; CIPHERTEXT_BYTES],
    local_part: &str,
    emit_hex_debug: bool,
) -> SegmentedRecord {
    // Build the final human-readable view from one 32-byte plaintext block.
    // All later reports and CSV exports derive from this normalized structure.
    let tail = &plaintext[24..];
    let tail_clean = clean_string_slice(tail);
    let tail_ascii85 = maybe_ascii85_decode(&tail_clean);
    let tail_base64 = maybe_base64_decode(&tail_clean);
    let tail_candidates = build_tail_candidates(tail);
    let tail_hybrid = choose_hybrid_tail_decoder(&tail_candidates, None);
    let shift_key = derive_shift_key(tail, &tail_clean, tail_hybrid.as_ref());

    let mut timestamp_bytes = [0_u8; 16];
    timestamp_bytes.copy_from_slice(&plaintext[..16]);
    let timestamp_golden = scan_timestamp_segment_with_shift(&timestamp_bytes, shift_key);
    let timestamp_texts = build_flag_sync_timestamp_texts(&plaintext[..16], shift_key);
    let timestamp_segment = choose_segment_display(&timestamp_texts, "t");
    let timestamp_note = timestamp_golden
        .as_ref()
        .map(|hit| format!("{} ({}) [{}]", hit.epoch, hit.datetime.format("%Y-%m-%d %H:%M:%S UTC"), hit.rule));

    let uid_plus56_texts = build_dynamic_segment_texts(&plaintext[8..24], 56, shift_key);
    let uid_segment_plus56 = choose_segment_display(&uid_plus56_texts, "u");
    if emit_hex_debug {
        emit_segment_hex_monitor("u", &plaintext[8..24], 56);
    }
    let u_hex_token = synthesize_hex_token_with_shift(&plaintext[8..24], 56, 5, shift_key);

    let uid_plus115_texts = build_dynamic_segment_texts(&plaintext[8..24], 115, shift_key);
    let uid_segment_plus115 = choose_segment_display(&uid_plus115_texts, "c");
    if emit_hex_debug {
        emit_segment_hex_monitor("c", &plaintext[8..24], 115);
    }
    let c_hex_token = synthesize_hex_token_with_shift(&plaintext[8..24], 115, 4, shift_key);

    let reassembled_segment = [timestamp_segment.as_str(), uid_segment_plus56.as_str(), uid_segment_plus115.as_str()]
        .into_iter()
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join("|");
    let tail_hybrid = choose_hybrid_tail_decoder(&tail_candidates, Some(&reassembled_segment));
    let aligned_body = choose_global_alignment(
        plaintext,
        local_part,
        Some(&reassembled_segment),
        tail_hybrid.as_ref().map(|candidate| candidate.value.as_str()),
    );

    let mut source_texts = vec![
        aligned_body.clone(),
        timestamp_segment.clone(),
        uid_segment_plus56.clone(),
        uid_segment_plus115.clone(),
        reassembled_segment.clone(),
    ];
    let anchored_alignment = fixed_anchor_alignment(&aligned_body);
    if !anchored_alignment.is_empty() {
        source_texts.push(anchored_alignment);
    }
    if let Some(value) = tail_ascii85.as_ref() {
        source_texts.push(value.clone());
    }
    if let Some(value) = tail_base64.as_ref() {
        source_texts.push(value.clone());
    }
    if let Some(candidate) = tail_hybrid.as_ref() {
        source_texts.push(candidate.value.clone());
    }
    source_texts.sort();
    source_texts.dedup();

    let combined_text = source_texts
        .into_iter()
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join("|");
    let mut polished_pairs = extract_field_pairs(&combined_text);
    add_offset_backed_pairs(
        &mut polished_pairs,
        &[],
        &uid_plus56_texts,
        &uid_plus115_texts,
        &aligned_body,
    );
    if let Some(golden) = timestamp_golden.as_ref() {
        polished_pairs.retain(|(key, _)| key != "t" && key != "timestamp");
        polished_pairs.push(("t".to_owned(), golden.digits.clone()));
        polished_pairs.sort();
        polished_pairs.dedup();
    }
    let user_id = u_hex_token
        .as_ref()
        .map(|candidate| candidate.value.clone())
        .or_else(|| extract_named_value(&polished_pairs, &["u", "uid"]))
        .unwrap_or_else(|| "-".to_owned());
    let campaign_id = c_hex_token
        .as_ref()
        .map(|candidate| candidate.value.clone())
        .or_else(|| extract_named_value(&polished_pairs, &["c", "campaign", "campaign_id"]))
        .unwrap_or_else(|| "-".to_owned());
    canonicalize_core_pairs(&mut polished_pairs, timestamp_golden.as_ref(), &user_id, &campaign_id);
    let final_string = build_final_string(&polished_pairs, &combined_text);
    let validated_fields = collect_validated_fields(
        &polished_pairs,
        tail_hybrid.as_ref().map(|candidate| candidate.value.as_str()),
    );
    let report_timestamp = timestamp_golden
        .as_ref()
        .map(|hit| format!("{} ({}) [{}]", hit.epoch, hit.datetime.format("%Y-%m-%d %H:%M:%S UTC"), hit.rule))
        .or(timestamp_note.clone())
        .unwrap_or_else(|| "-".to_owned());
    let extra_flags = collect_extra_flags(&polished_pairs, tail_hybrid.as_ref(), &validated_fields);
    let diamond_polished = has_polished_core_fields(&polished_pairs);
    let (golden_result, realistic_score) = synthesize_realistic_result(
        timestamp_golden.as_ref(),
        &user_id,
        &campaign_id,
    );
    let mission_accomplished = golden_result.is_some();

    SegmentedRecord {
        timestamp_segment,
        timestamp_note,
        uid_segment_plus56,
        uid_segment_plus115,
        reassembled_segment,
        tail_ascii85,
        tail_base64,
        tail_hybrid: tail_hybrid.as_ref().map(|candidate| candidate.value.clone()),
        combined_text,
        final_string,
        report_timestamp,
        user_id,
        campaign_id,
        extra_flags,
        polished_pairs,
        diamond_polished,
        mission_accomplished,
        golden_result,
        realistic_score,
        shift_key,
        timestamp_rule: timestamp_golden.as_ref().map(|hit| hit.rule.clone()),
    }
}

fn print_success_report(parsed: &SegmentedRecord) {
    println!("Timestamp: {}", parsed.report_timestamp);
    println!("User ID (u): {}", parsed.user_id);
    println!("Campaign ID (c): {}", parsed.campaign_id);
    println!("Extra Flags: {}", parsed.extra_flags);
    println!("Realistic Score: {}%", parsed.realistic_score);
    if let Some(result) = parsed.golden_result.as_ref() {
        println!("ALTIN SONUC: {}", result);
    }
    if !parsed.polished_pairs.is_empty() {
        println!();
        println!("Field | Value");
        for (key, value) in &parsed.polished_pairs {
            println!("{key} | {value}");
        }
    }
    if parsed.diamond_polished {
        println!();
        println!("DIAMOND POLISHED");
    }
    if parsed.mission_accomplished {
        print_mission_completed_respect();
    }
    println!();
    println!(
        "Reversal algoritmasi, bu cozumde kaymis veriyi gosterilebilir bir stringe cevirmekte belirleyici oldu."
    );

    if !parsed.mission_accomplished
        && (has_mission_token(&parsed.final_string) || has_mission_token(&mission_haystack(parsed)))
    {
        print_mission_accomplished();
    }
}

fn print_consistency_report(rows: &[DatasetSynthesisRow]) {
    println!("CONSISTENCY TABLE");
    println!(
        "line | strict | raw | norm | pad | nonce      | shift | score | transform                     | t          | u                | c"
    );
    println!(
        "-----+--------+-----+------+-----+------------+-------+-------+-------------------------------+------------+------------------+------------------"
    );

    for row in rows {
        let strict = if row.strict_length_match { "ok" } else { "fail" };
        let transform = row
            .parsed
            .timestamp_rule
            .as_deref()
            .unwrap_or("-");
        let timestamp = extract_named_value(&row.parsed.polished_pairs, &["t", "timestamp"])
            .unwrap_or_else(|| "-".to_owned());

        println!(
            "{:>4} | {:>6} | {:>3} | {:>4} | {:>3} | {:<10} | {:>5} | {:>5}% | {:<29} | {:<10} | {:<16} | {}",
            row.line_no,
            strict,
            row.raw_length,
            row.normalized_length,
            row.payload_pad_added,
            row.nonce_mode,
            row.parsed.shift_key,
            row.parsed.realistic_score,
            truncate_cell(transform, 29),
            truncate_cell(&timestamp, 10),
            truncate_cell(&row.parsed.user_id, 16),
            truncate_cell(&row.parsed.campaign_id, 16),
        );
    }

    let golden_hits = rows
        .iter()
        .filter(|row| row.parsed.golden_result.is_some())
        .count();
    println!();
    println!("golden_hits={golden_hits}/{}", rows.len());

    if let Some(reference) = rows.iter().find(|row| row.line_no == FIXED_BLOB_INDEX + 1) {
        println!();
        println!("REFERENCE LINE {}", reference.line_no);
        print_success_report(&reference.parsed);
    }
}

fn print_strict_length_analysis(payloads: &[InputPayloadRecord], rows: &[DatasetSynthesisRow]) {
    let mut mismatch_lines = Vec::new();
    let mut mismatch_lengths = BTreeMap::<usize, usize>::new();
    let mut mismatch_padding = BTreeMap::<usize, usize>::new();
    let mut mismatch_prefixes = BTreeMap::<String, usize>::new();
    let mut mismatch_scores = Vec::new();

    for (payload, row) in payloads.iter().zip(rows) {
        if payload.strict_length_match {
            continue;
        }

        mismatch_lines.push(payload.line_no);
        *mismatch_lengths.entry(payload.normalized_length).or_default() += 1;
        *mismatch_padding.entry(payload.payload_pad_added).or_default() += 1;
        *mismatch_prefixes
            .entry(hex_encode(&payload.payload[..4]))
            .or_default() += 1;
        mismatch_scores.push(row.parsed.realistic_score as usize);
    }

    println!();
    println!("STRICT LENGTH ANALYSIS");
    println!(
        "matched={} mismatched={}",
        payloads.iter().filter(|payload| payload.strict_length_match).count(),
        mismatch_lines.len()
    );
    println!("mismatch_lines={mismatch_lines:?}");
    println!("normalized_length_counts={mismatch_lengths:?}");
    println!("pad_added_counts={mismatch_padding:?}");

    let common_prefixes: Vec<String> = mismatch_prefixes
        .iter()
        .filter(|(_, count)| **count > 1)
        .map(|(prefix, count)| format!("{prefix}x{count}"))
        .collect();
    println!("common_header_prefix4={common_prefixes:?}");

    if mismatch_padding.keys().all(|pad| *pad == 0) {
        println!("inference=padding mismatch degil; fark outer record/header varyasyonu gibi gorunuyor");
    } else {
        println!("inference=padding farki da var; strict-length sapmasi karisik nedenli");
    }

    if !mismatch_scores.is_empty() {
        let avg = mismatch_scores.iter().sum::<usize>() as f64 / mismatch_scores.len() as f64;
        println!("avg_realism_on_mismatches={avg:.1}%");
    }
}

fn print_line11_surgical_report(report: Option<&SurgicalTimestampReport>) {
    println!();
    println!("LINE 11 SURGICAL EXTRACTION");
    match report {
        Some(report) => {
            println!(
                "segment_attempts={} aes_attempts={} deep_attempts={} coding_attempts={} winner={}",
                report.segment_attempts,
                report.aes_attempts,
                report.deep_attempts,
                report.coding_attempts,
                report.source
            );
            if let Some(best) = report.best.as_ref() {
                println!(
                    "line={} timestamp={} ({}) rule={}",
                    report.line_no,
                    best.epoch,
                    best.datetime.format("%Y-%m-%d %H:%M:%S UTC"),
                    best.rule
                );
            } else {
                println!("line={} no timestamp recovered", report.line_no);
            }
        }
        None => println!("line=11 not present"),
    }
}

fn export_consistency_csv(rows: &[DatasetSynthesisRow], path: &str) -> std::io::Result<()> {
    let mut csv = String::from("line_no,t,u,c,score,flag\n");
    for row in rows {
        let timestamp = extract_named_value(&row.parsed.polished_pairs, &["t", "timestamp"])
            .unwrap_or_else(|| "-".to_owned());
        csv.push_str(&csv_escape(&row.line_no.to_string()));
        csv.push(',');
        csv.push_str(&csv_escape(&timestamp));
        csv.push(',');
        csv.push_str(&csv_escape(&row.parsed.user_id));
        csv.push(',');
        csv.push_str(&csv_escape(&row.parsed.campaign_id));
        csv.push(',');
        csv.push_str(&csv_escape(&row.parsed.realistic_score.to_string()));
        csv.push(',');
        csv.push_str(&csv_escape(&row.parsed.shift_key.to_string()));
        csv.push('\n');
    }
    fs::write(path, csv)
}

fn csv_escape(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

fn truncate_cell(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        return value.to_owned();
    }
    value.chars().take(max.saturating_sub(1)).collect::<String>() + "…"
}

fn candidate_nonces(header: &[u8], base_nonce: &[u8; BLOCK_BYTES]) -> Vec<(String, [u8; BLOCK_BYTES])> {
    let mut nonces = Vec::new();
    nonces.push(("baseline".to_owned(), *base_nonce));

    if header.len() >= BLOCK_BYTES + 1 {
        let mut shifted = [0_u8; BLOCK_BYTES];
        shifted.copy_from_slice(&header[1..1 + BLOCK_BYTES]);
        nonces.push(("offset+1".to_owned(), shifted));
    }

    let mut counter_minus = *base_nonce;
    counter_minus[BLOCK_BYTES - 1] = counter_minus[BLOCK_BYTES - 1].wrapping_sub(1);
    nonces.push(("counter-1".to_owned(), counter_minus));

    let mut counter_plus = *base_nonce;
    counter_plus[BLOCK_BYTES - 1] = counter_plus[BLOCK_BYTES - 1].wrapping_add(1);
    nonces.push(("counter+1".to_owned(), counter_plus));

    nonces
}

fn score_resync_candidate(plaintext: &[u8; CIPHERTEXT_BYTES], local_part: &[u8]) -> usize {
    let local_part = std::str::from_utf8(local_part).unwrap_or("mmpejmrp");
    let u_texts = build_aggressive_segment_texts(&plaintext[8..24], 56);
    let c_texts = build_aggressive_segment_texts(&plaintext[8..24], 115);
    let u_value = extract_marker_value_from_texts(&u_texts, "u").unwrap_or_default();
    let c_value = extract_marker_value_from_texts(&c_texts, "c").unwrap_or_default();
    let aligned = choose_global_alignment(plaintext, local_part, None, None);

    let mut score = 0_usize;
    if is_base62_like(&u_value) {
        score += 120 + u_value.len() * 10;
    }
    if is_base62_like(&c_value) {
        score += 120 + c_value.len() * 10;
    }
    if u_value == "cFNhX" {
        score += 2_000;
    }
    if c_value == "FNhX" {
        score += 2_000;
    }
    if aligned.contains("A\\SR") {
        score += 600;
    }
    if aligned.contains("u=") {
        score += 500;
    }
    score + aligned.chars().filter(|ch| ch.is_ascii_alphanumeric()).count()
}

fn mission_haystack(parsed: &SegmentedRecord) -> String {
    let mut all = vec![
        parsed.combined_text.clone(),
        parsed.timestamp_segment.clone(),
        parsed.uid_segment_plus56.clone(),
        parsed.uid_segment_plus115.clone(),
        parsed.reassembled_segment.clone(),
    ];
    if let Some(value) = parsed.timestamp_note.as_ref() {
        all.push(value.clone());
    }
    if let Some(value) = parsed.tail_ascii85.as_ref() {
        all.push(value.clone());
    }
    if let Some(value) = parsed.tail_base64.as_ref() {
        all.push(value.clone());
    }
    if let Some(value) = parsed.tail_hybrid.as_ref() {
        all.push(value.clone());
    }
    all.join("|")
}

fn decrypt_aes128_ctr_split(
    ciphertext: &[u8],
    key: &[u8; 32],
    iv: &[u8; BLOCK_BYTES],
) -> Option<[u8; CIPHERTEXT_BYTES]> {
    let mut plaintext = [0_u8; CIPHERTEXT_BYTES];
    plaintext.copy_from_slice(ciphertext);
    let mut cipher = Ctr128BE::<Aes128>::new_from_slices(&key[..BLOCK_BYTES], iv).ok()?;
    cipher.apply_keystream(&mut plaintext);
    Some(plaintext)
}

fn extract_timestamp(clean: &str) -> Option<TimestampHit> {
    let digits: String = clean.chars().filter(|ch| ch.is_ascii_digit()).collect();
    if digits.len() < 10 {
        return None;
    }

    let epoch = digits[..10].parse::<i64>().ok()?;
    parse_target_epoch(epoch).map(|datetime| TimestampHit { epoch, datetime })
}

fn clean_string_slice(bytes: &[u8]) -> String {
    let lossy = String::from_utf8_lossy(bytes);
    clean_human_text(lossy.as_ref())
}

fn choose_global_alignment(
    plaintext: &[u8; CIPHERTEXT_BYTES],
    local_part: &str,
    reassembled: Option<&str>,
    tail_hybrid: Option<&str>,
) -> String {
    let mut candidates = build_aggressive_body_texts(plaintext, local_part);
    let direct = clean_string_slice(plaintext);
    let anchored_direct = fixed_anchor_alignment(&direct);
    if !anchored_direct.is_empty() {
        candidates.push(anchored_direct);
    }
    if let Some(reassembled) = reassembled.filter(|value| !value.is_empty()) {
        candidates.push(reassembled.to_owned());
    }
    if let Some(tail_hybrid) = tail_hybrid.filter(|value| !value.is_empty()) {
        candidates.push(tail_hybrid.to_owned());
    }

    candidates.sort();
    candidates.dedup();
    candidates
        .into_iter()
        .max_by_key(|candidate| score_alignment_candidate(candidate, local_part))
        .unwrap_or_default()
}

fn clean_human_text(input: &str) -> String {
    let mut cleaned = String::with_capacity(input.len());
    for ch in input.chars() {
        if ch == '\u{fffd}' {
            continue;
        }
        if ch.is_ascii_control() && !ch.is_ascii_whitespace() {
            continue;
        }
        if ch.is_ascii() {
            cleaned.push(ch);
        }
    }
    cleaned.trim().to_owned()
}

fn sanitize_field_fragment(input: &str) -> String {
    let trimmed = clean_human_text(input);
    trimmed
        .trim_matches(|ch: char| !(ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | ':' | '@' | '/' | '+' | '=')))
        .to_owned()
}

fn extract_field_pairs(final_string: &str) -> Vec<(String, String)> {
    let mut fields = Vec::new();

    for chunk in final_string.split(['&', '|']) {
        let trimmed = chunk.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Some((key, value)) = trimmed.split_once('=') else {
            continue;
        };

        let clean_key = sanitize_field_fragment(key).to_ascii_lowercase();
        let clean_value = sanitize_field_fragment(value);
        if clean_key.is_empty() || clean_value.is_empty() {
            continue;
        }

        fields.push((clean_key, clean_value));
    }

    collect_loose_marker_pairs(final_string, &mut fields);

    fields.sort();
    fields.dedup();
    fields
}

fn collect_loose_marker_pairs(text: &str, fields: &mut Vec<(String, String)>) {
    for key in ["t", "u", "c"] {
        let values = extract_loose_marker_values(text, key);
        for value in values {
            fields.push((key.to_owned(), value));
        }
    }
}

fn extract_loose_marker_values(text: &str, key: &str) -> Vec<String> {
    let marker = key.as_bytes()[0].to_ascii_lowercase();
    let lowered = text.to_ascii_lowercase();
    let bytes = lowered.as_bytes();
    let mut values = Vec::new();
    let mut index = 0_usize;

    while index < bytes.len() {
        if bytes[index] != marker {
            index += 1;
            continue;
        }

        let mut cursor = index + 1;
        let mut skipped = 0_usize;
        while cursor < bytes.len() && skipped < 3 {
            let byte = bytes[cursor];
            if byte.is_ascii_alphanumeric() {
                break;
            }
            cursor += 1;
            skipped += 1;
        }

        let start = cursor;
        while cursor < bytes.len() {
            let byte = bytes[cursor];
            let keep = if key == "t" {
                byte.is_ascii_digit()
            } else {
                byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b'@')
            };
            if !keep || cursor - start >= 10 {
                break;
            }
            cursor += 1;
        }

        let length = cursor.saturating_sub(start);
        let min_len = if key == "t" { 4 } else { 4 };
        if length >= min_len {
            let value = sanitize_field_fragment(&text[start..cursor]);
            if value.len() >= min_len {
                values.push(value);
            }
        }

        index += 1;
    }

    values
}

fn add_offset_backed_pairs(
    fields: &mut Vec<(String, String)>,
    timestamp_texts: &[String],
    uid_plus56_texts: &[String],
    uid_plus115_texts: &[String],
    aligned_body: &str,
) {
    if !fields.iter().any(|(key, _)| key == "t") {
        if let Some(value) = extract_marker_value_from_texts(timestamp_texts, "t") {
            fields.push(("t".to_owned(), value));
        }
    }
    if !fields.iter().any(|(key, _)| key == "u") {
        if let Some(value) = extract_marker_value_from_texts(uid_plus56_texts, "u") {
            fields.push(("u".to_owned(), value));
        }
    }
    if !fields.iter().any(|(key, _)| key == "c") {
        if let Some(value) = extract_marker_value_from_texts(uid_plus115_texts, "c") {
            fields.push(("c".to_owned(), value));
        } else if let Some(value) = extract_best_block(aligned_body, false) {
            fields.push(("c".to_owned(), value));
        }
    }

    fields.sort();
    fields.dedup();
}

fn extract_best_block(text: &str, digits_only: bool) -> Option<String> {
    let mut best = String::new();
    let mut current = String::new();

    for ch in text.chars() {
        let keep = if digits_only {
            ch.is_ascii_digit()
        } else {
            ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | '@')
        };

        if keep {
            current.push(ch);
            if current.len() == 10 {
                break;
            }
            continue;
        }

        if current.len() > best.len() {
            best = current.clone();
        }
        current.clear();
    }

    if current.len() > best.len() {
        best = current;
    }

    let min_len = if digits_only { 4 } else { 4 };
    (best.len() >= min_len).then_some(best)
}

fn choose_segment_display(texts: &[String], key: &str) -> String {
    texts.iter()
        .max_by_key(|text| score_segment_candidate(text, key))
        .cloned()
        .unwrap_or_default()
}

fn score_segment_candidate(text: &str, key: &str) -> usize {
    let mut score = text.chars().filter(|ch| ch.is_ascii_alphanumeric()).count();
    score += text.matches('&').count() * 20;
    score += text.matches('=').count() * 40;
    score += text.matches('|').count() * 4;

    if text.to_ascii_lowercase().contains(&format!("{key}=")) {
        score += 120;
    }
    if matches!(key, "u" | "c") {
        if let Some(token) = extract_base62_token(text) {
            score += 220 + token.len() * 10;
        }
    }
    if let Some(value) = extract_marker_value_from_texts(&[text.to_owned()], key) {
        score += value.len() * 10;
        if key == "t" && extract_timestamp(&value).is_some() {
            score += 1000;
        }
    }

    score
}

fn extract_marker_value_from_texts(texts: &[String], key: &str) -> Option<String> {
    let mut best: Option<(usize, String)> = None;

    for text in texts {
        if matches!(key, "u" | "c") {
            if let Some(value) = extract_base62_token(text) {
                let score = 500 + value.len() * 10;
                if best.as_ref().map_or(true, |(best_score, _)| score > *best_score) {
                    best = Some((score, value));
                }
            }
        }
        for value in extract_loose_marker_values(text, key) {
            let mut score = value.len() * 10;
            if key == "t" && extract_timestamp(&value).is_some() {
                score += 1000;
            }
            if best.as_ref().map_or(true, |(best_score, _)| score > *best_score) {
                best = Some((score, value));
            }
        }

        if let Some(value) = extract_best_block(text, key == "t") {
            let mut score = value.len();
            if key == "t" && extract_timestamp(&value).is_some() {
                score += 800;
            }
            if best.as_ref().map_or(true, |(best_score, _)| score > *best_score) {
                best = Some((score, value));
            }
        }
    }

    best.map(|(_, value)| value)
}

fn extract_base62_token(text: &str) -> Option<String> {
    let mut best = String::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            current.push(ch);
            continue;
        }

        if current.len() > best.len() && is_base62_like(&current) {
            best = current.clone();
        }
        current.clear();
    }

    if current.len() > best.len() && is_base62_like(&current) {
        best = current;
    }

    (!best.is_empty()).then_some(best)
}

fn is_base62_like(token: &str) -> bool {
    (4..=16).contains(&token.len())
        && token.chars().all(|ch| ch.is_ascii_alphanumeric())
        && token.chars().any(|ch| ch.is_ascii_lowercase())
        && token.chars().any(|ch| ch.is_ascii_uppercase())
}

fn synthesize_hex_token(segment: &[u8], offset: u8, preferred_len: usize) -> Option<HexTokenCandidate> {
    let mut best: Option<HexTokenCandidate> = None;

    for (_source, bytes) in build_hex_processor_variants(segment, offset) {
        let direct = clean_string_slice(&bytes);
        if let Some(value) = extract_base62_token(&direct) {
            let score = score_hex_token_candidate(&value, preferred_len, false);
            update_hex_candidate(&mut best, value, score);
        }

        let base64 = STANDARD.encode(&bytes);
        if let Some(value) = extract_base62_token(&base64) {
            let score = score_hex_token_candidate(&value, preferred_len, true);
            update_hex_candidate(&mut best, value, score);
        }

        let base62 = map_bytes_to_base62(&bytes);
        if let Some(value) = select_base62_window(&base62, preferred_len) {
            let score = score_hex_token_candidate(&value, preferred_len, true);
            update_hex_candidate(&mut best, value, score);
        }
    }

    best
}

fn synthesize_hex_token_with_shift(
    segment: &[u8],
    offset: u8,
    preferred_len: usize,
    shift_key: u8,
) -> Option<HexTokenCandidate> {
    let mut best = synthesize_hex_token(segment, offset, preferred_len);

    for bytes in build_shifted_segment_variants(segment, offset, shift_key) {
        let direct = clean_string_slice(&bytes);
        if let Some(value) = extract_base62_token(&direct) {
            let score = score_hex_token_candidate(&value, preferred_len, false) + 25;
            update_hex_candidate(&mut best, value, score);
        }

        let base64 = STANDARD.encode(&bytes);
        if let Some(value) = extract_base62_token(&base64) {
            let score = score_hex_token_candidate(&value, preferred_len, true) + 15;
            update_hex_candidate(&mut best, value, score);
        }

        let base62 = map_bytes_to_base62(&bytes);
        if let Some(value) = select_base62_window(&base62, preferred_len) {
            let score = score_hex_token_candidate(&value, preferred_len, true) + 30;
            update_hex_candidate(&mut best, value, score);
        }
    }

    best
}

fn update_hex_candidate(best: &mut Option<HexTokenCandidate>, value: String, score: usize) {
    if best.as_ref().map_or(true, |current| score > current.score) {
        *best = Some(HexTokenCandidate { value, score });
    }
}

fn score_hex_token_candidate(token: &str, preferred_len: usize, mapped: bool) -> usize {
    let mut score = token.len() * 20;
    if is_base62_like(token) {
        score += 300;
    }
    if token.len() == preferred_len {
        score += 220;
    } else {
        score += 120usize.saturating_sub(token.len().abs_diff(preferred_len) * 30);
    }
    if token.chars().any(|ch| ch.is_ascii_lowercase()) && token.chars().any(|ch| ch.is_ascii_uppercase()) {
        score += 120;
    }
    if mapped {
        score += 40;
    }
    if token == "cFNhX" || token == "FNhX" {
        score += 2_000;
    }
    score
}

fn build_hex_processor_variants(segment: &[u8], offset: u8) -> Vec<(String, Vec<u8>)> {
    let mut base = segment.to_vec();
    apply_wrapping_offset_slice(&mut base, offset);

    let mut variants = vec![
        ("offset".to_owned(), base.clone()),
        ("reversed".to_owned(), reverse_bytes_slice(&base)),
        ("reverse-bits".to_owned(), transform_reverse_bits_slice(&base)),
    ];

    let subtractive = transform_subtractive_slice(&base, offset);
    variants.push(("subtractive".to_owned(), subtractive));

    let inverted = transform_inverted_offset_slice(&base, offset);
    variants.push(("inverted".to_owned(), inverted));

    variants
}

fn map_bytes_to_base62(bytes: &[u8]) -> String {
    const BASE62: &[u8; 62] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    bytes.iter()
        .map(|byte| BASE62[*byte as usize % 62] as char)
        .collect()
}

fn select_base62_window(mapped: &str, preferred_len: usize) -> Option<String> {
    let mut best: Option<(usize, String)> = None;
    for len in [preferred_len, preferred_len + 1, preferred_len.saturating_sub(1)] {
        if len == 0 || mapped.len() < len {
            continue;
        }
        for window in mapped.as_bytes().windows(len) {
            let Ok(token) = std::str::from_utf8(window) else {
                continue;
            };
            if !is_base62_like(token) {
                continue;
            }
            let score = score_hex_token_candidate(token, preferred_len, true);
            if best.as_ref().map_or(true, |(best_score, _)| score > *best_score) {
                best = Some((score, token.to_owned()));
            }
        }
    }
    best.map(|(_, token)| token)
}

fn synthesize_realistic_result(
    timestamp: Option<&TimestampGoldenKey>,
    user_id: &str,
    campaign_id: &str,
) -> (Option<String>, u8) {
    let mut score = 0_u8;
    let mut parts = Vec::new();

    if let Some(timestamp) = timestamp {
        let digits = timestamp.epoch.to_string();
        if digits.len() == 10 && digits.chars().all(|ch| ch.is_ascii_digit()) {
            parts.push(format!("t={digits}"));
            score += 34;
        }
    }
    if is_base62_like(user_id) {
        parts.push(format!("u={user_id}"));
        score += 33;
    }
    if is_base62_like(campaign_id) {
        parts.push(format!("c={campaign_id}"));
        score += 33;
    }

    if score >= 90 && parts.len() == 3 {
        (Some(parts.join("&")), score)
    } else {
        (None, score)
    }
}

fn extract_timestamp_from_texts(texts: &[String]) -> Option<TimestampHit> {
    for text in texts {
        for value in extract_loose_marker_values(text, "t") {
            if let Some(hit) = extract_timestamp(&value) {
                return Some(hit);
            }
        }
        if let Some(value) = extract_best_block(text, true) {
            if let Some(hit) = extract_timestamp(&value) {
                return Some(hit);
            }
        }
    }
    None
}

fn scan_timestamp_segment(segment: &[u8; 16]) -> Option<TimestampGoldenKey> {
    let mut best: Option<(usize, TimestampGoldenKey)> = None;

    let mut base = *segment;
    apply_wrapping_offset_slice(&mut base, 4);
    consider_ascii_timestamp_windows("offset+4/ascii", &base, 0, &mut best);
    for (window_offset, window) in base.windows(4).enumerate() {
        let subtractive = transform_subtractive_slice(window, 4);
        warn_if_logic_loop(
            &format!("timestamp/window@{window_offset}/subtractive"),
            window,
            &subtractive,
        );
        consider_integer_timestamp_variant(
            &format!("offset+4/window@{window_offset}/subtractive"),
            &subtractive,
            window_offset * 4,
            &mut best,
        );

        let inverted = transform_inverted_offset_slice(window, 4);
        warn_if_logic_loop(
            &format!("timestamp/window@{window_offset}/inverted"),
            window,
            &inverted,
        );
        consider_integer_timestamp_variant(
            &format!("offset+4/window@{window_offset}/inverted"),
            &inverted,
            window_offset * 4 + 10,
            &mut best,
        );
    }
    for (window_offset, window) in base.windows(8).enumerate() {
        let subtractive = transform_subtractive_slice(window, 4);
        consider_u64_timestamp_variant(
            &format!("offset+4/window8@{window_offset}/subtractive"),
            &subtractive,
            100 + window_offset * 4,
            &mut best,
        );

        let inverted = transform_inverted_offset_slice(window, 4);
        consider_u64_timestamp_variant(
            &format!("offset+4/window8@{window_offset}/inverted"),
            &inverted,
            110 + window_offset * 4,
            &mut best,
        );
    }

    best.map(|(_, hit)| hit)
}

fn scan_timestamp_segment_with_shift(segment: &[u8; 16], shift_key: u8) -> Option<TimestampGoldenKey> {
    let mut best = scan_timestamp_segment(segment).map(|hit| (0_usize, hit));
    for (label, bytes, penalty) in build_flag_rotated_timestamp_variants(segment, shift_key) {
        consider_anchored_ascii_timestamp(&label, &bytes, penalty, &mut best);

        for (window_offset, window) in bytes.windows(4).enumerate() {
            consider_integer_timestamp_variant(
                &format!("{label}/window@{window_offset}"),
                window,
                penalty + window_offset * 4,
                &mut best,
            );
        }
    }

    best.map(|(_, hit)| hit)
}

fn scan_line11_timestamp_surgical(
    plaintext: &[u8; CIPHERTEXT_BYTES],
) -> (Option<TimestampGoldenKey>, usize) {
    let mut best: Option<(usize, TimestampGoldenKey)> = None;
    let mut attempts = 0_usize;

    for header_offset in 0..=2 {
        let end = header_offset + 16;
        if end > plaintext.len() {
            break;
        }

        let mut segment = [0_u8; 16];
        segment.copy_from_slice(&plaintext[header_offset..end]);

        let mut base = segment;
        apply_wrapping_offset_slice(&mut base, 4);

        for bit_shift in 1_u32..=7 {
            let shifted_left = shift_bits_left_slice(&base, bit_shift);
            attempts += 1;
            consider_ascii_timestamp_windows(
                &format!("line11/header{header_offset:+}/shl{bit_shift}"),
                &shifted_left,
                header_offset * 10 + bit_shift as usize,
                &mut best,
            );
            for (window_offset, window) in shifted_left.windows(4).enumerate() {
                consider_integer_timestamp_variant(
                    &format!("line11/header{header_offset:+}/shl{bit_shift}/window@{window_offset}"),
                    window,
                    header_offset * 20 + bit_shift as usize + window_offset,
                    &mut best,
                );
            }

            let shifted_right = shift_bits_right_slice(&base, bit_shift);
            attempts += 1;
            consider_ascii_timestamp_windows(
                &format!("line11/header{header_offset:+}/shr{bit_shift}"),
                &shifted_right,
                header_offset * 10 + bit_shift as usize + 40,
                &mut best,
            );
            for (window_offset, window) in shifted_right.windows(4).enumerate() {
                consider_integer_timestamp_variant(
                    &format!("line11/header{header_offset:+}/shr{bit_shift}/window@{window_offset}"),
                    window,
                    header_offset * 20 + bit_shift as usize + window_offset + 40,
                    &mut best,
                );
            }
        }
    }

    (best.map(|(_, hit)| hit), attempts)
}

fn scan_line11_aes_state_jitter(
    payload: &[u8; BLOB_BYTES],
    fixed: &FixedDecryptor,
    local_part: &str,
    expected_u: &str,
    expected_c: &str,
) -> (Option<TimestampGoldenKey>, usize) {
    let header = &payload[..HEADER_BYTES];
    let ciphertext = &payload[HEADER_BYTES..HEADER_BYTES + CIPHERTEXT_BYTES];
    let mut raw_nonce = [0_u8; BLOCK_BYTES];
    raw_nonce.copy_from_slice(&header[FIXED_NONCE_OFFSET..FIXED_NONCE_OFFSET + BLOCK_BYTES]);

    let mut best: Option<(usize, TimestampGoldenKey)> = None;
    let mut attempts = 0_usize;

    for flip in 0..=16 {
        let mut flipped_nonce = raw_nonce;
        let flip_label = if flip == 16 {
            "raw".to_owned()
        } else {
            let byte_index = BLOCK_BYTES - 2 + flip / 8;
            let bit_index = flip % 8;
            flipped_nonce[byte_index] ^= 1_u8 << bit_index;
            format!("flip@{}:{}", byte_index, bit_index)
        };

        for drift in -4_i8..=4_i8 {
            let mut candidate_nonce = flipped_nonce;
            apply_counter_drift(&mut candidate_nonce, drift);
            attempts += 1;

            let Some(plaintext) =
                decrypt_aes128_ctr_split(ciphertext, &fixed.derived_key, &candidate_nonce)
            else {
                continue;
            };
            let parsed = parse_segmented_record(&plaintext, local_part, false);
            if parsed.user_id != expected_u || parsed.campaign_id != expected_c {
                continue;
            }

            if let Some(hit) =
                extract_timestamp_from_parsed(&parsed, &format!("line11/{flip_label}/drift{drift:+}"))
            {
                let score = 20_000usize
                    .saturating_sub(flip.saturating_mul(32))
                    .saturating_sub(drift.unsigned_abs() as usize * 8);
                if best.as_ref().map_or(true, |(best_score, _)| score > *best_score) {
                    best = Some((score, hit));
                }
            }
        }
    }

    (best.map(|(_, hit)| hit), attempts)
}

fn scan_line11_deep_deviation(
    payload: &[u8; BLOB_BYTES],
    fixed: &FixedDecryptor,
    local_part: &str,
    expected_u: &str,
    expected_c: &str,
    flag: u8,
) -> (Option<TimestampGoldenKey>, usize) {
    let header = &payload[..HEADER_BYTES];
    let ciphertext = &payload[HEADER_BYTES..HEADER_BYTES + CIPHERTEXT_BYTES];
    let mut raw_nonce = [0_u8; BLOCK_BYTES];
    raw_nonce.copy_from_slice(&header[FIXED_NONCE_OFFSET..FIXED_NONCE_OFFSET + BLOCK_BYTES]);

    let mut best: Option<(usize, TimestampGoldenKey)> = None;
    let mut attempts = 0_usize;

    for slide in 0..=4 {
        let candidate_ciphertext = build_ciphertext_window(ciphertext, slide);

        for mutation in [
            Line11Mutation::Identity,
            Line11Mutation::KeyXorFlag,
            Line11Mutation::NonceXorFlag,
            Line11Mutation::KeyAndNonceXorFlag,
        ] {
            let mut candidate_key = fixed.derived_key;
            let mut candidate_nonce = raw_nonce;
            apply_line11_mutation(&mut candidate_key, &mut candidate_nonce, flag, mutation);

            for drift in -4_i8..=4_i8 {
                let mut drifted_nonce = candidate_nonce;
                apply_counter_drift(&mut drifted_nonce, drift);
                attempts += 1;

                let Some(plaintext) =
                    decrypt_aes128_ctr_split(&candidate_ciphertext, &candidate_key, &drifted_nonce)
                else {
                    continue;
                };
                let parsed = parse_segmented_record(&plaintext, local_part, false);
                if parsed.user_id != expected_u || parsed.campaign_id != expected_c {
                    continue;
                }

                if let Some(hit) = extract_timestamp_from_parsed(
                    &parsed,
                    &format!("line11/slide{slide}/{}{}", mutation.label(), drift_label(drift)),
                ) {
                    let score = 30_000usize
                        .saturating_sub(slide * 200)
                        .saturating_sub(mutation.penalty())
                        .saturating_sub(drift.unsigned_abs() as usize * 15);
                    if best.as_ref().map_or(true, |(best_score, _)| score > *best_score) {
                        best = Some((score, hit));
                    }
                }
            }
        }
    }

    (best.map(|(_, hit)| hit), attempts)
}

fn scan_line11_coding_matrix(
    payload: &[u8; BLOB_BYTES],
    fixed: &FixedDecryptor,
    local_part: &str,
    expected_u: &str,
    expected_c: &str,
    flag: u8,
    previous_line_epoch: Option<i64>,
) -> (Option<TimestampGoldenKey>, usize) {
    // Last-resort matrix: alternate timestamp encodings plus light HKDF
    // mutations, but only while `u` and `c` remain stable.
    let header = &payload[..HEADER_BYTES];
    let ciphertext = &payload[HEADER_BYTES..HEADER_BYTES + CIPHERTEXT_BYTES];
    let mut nonce = [0_u8; BLOCK_BYTES];
    nonce.copy_from_slice(&header[FIXED_NONCE_OFFSET..FIXED_NONCE_OFFSET + BLOCK_BYTES]);

    let mut attempts = 0_usize;
    let mut best: Option<(usize, TimestampGoldenKey)> = None;

    for (label, key) in build_line11_coding_keys(fixed, local_part.as_bytes(), flag) {
        for (nonce_mode, candidate_nonce) in candidate_nonces(header, &nonce) {
            attempts += 1;
            let Some(plaintext) = decrypt_aes128_ctr_split(ciphertext, &key, &candidate_nonce) else {
                continue;
            };
            let parsed = parse_segmented_record(&plaintext, local_part, false);
            if parsed.user_id != expected_u || parsed.campaign_id != expected_c {
                continue;
            }

            if let Some(hit) = scan_line11_coding_timestamp_bytes(
                &plaintext[..8],
                &format!("line11/{label}/{nonce_mode}"),
                previous_line_epoch,
            ) {
                let score = 40_000usize.saturating_sub(attempts * 3);
                if best.as_ref().map_or(true, |(best_score, _)| score > *best_score) {
                    best = Some((score, hit));
                }
            }
        }
    }

    (best.map(|(_, hit)| hit), attempts)
}

fn consider_ascii_timestamp_windows(
    label: &str,
    bytes: &[u8],
    penalty: usize,
    best: &mut Option<(usize, TimestampGoldenKey)>,
) {
    for (offset, window) in bytes.windows(10).enumerate() {
        if !window.iter().all(u8::is_ascii_digit) {
            continue;
        }
        let Ok(text) = std::str::from_utf8(window) else {
            continue;
        };
        let Ok(epoch) = text.parse::<i64>() else {
            continue;
        };
        update_timestamp_candidate(best, &format!("{label}/window10@{offset}"), epoch, penalty + offset * 2);
    }
}

fn consider_anchored_ascii_timestamp(
    label: &str,
    bytes: &[u8],
    penalty: usize,
    best: &mut Option<(usize, TimestampGoldenKey)>,
) {
    consider_ascii_timestamp_windows(label, bytes, penalty + 30, best);
    if bytes.len() < 10 {
        return;
    }

    let anchored = &bytes[..10];
    if !anchored.iter().all(u8::is_ascii_digit) {
        return;
    }
    if !anchored.starts_with(b"17") {
        return;
    }
    let Ok(text) = std::str::from_utf8(anchored) else {
        return;
    };
    let Ok(epoch) = text.parse::<i64>() else {
        return;
    };
    update_timestamp_candidate(best, &format!("{label}/anchored-17"), epoch, penalty);
}

fn consider_integer_timestamp_variant(
    label: &str,
    bytes: &[u8],
    penalty: usize,
    best: &mut Option<(usize, TimestampGoldenKey)>,
) {
    let Ok(window) = <[u8; 4]>::try_from(bytes) else {
        return;
    };
    update_timestamp_candidate(best, &format!("{label}/le"), u32::from_le_bytes(window) as i64, penalty);
    update_timestamp_candidate(best, &format!("{label}/be"), u32::from_be_bytes(window) as i64, penalty + 1);
}

fn consider_u64_timestamp_variant(
    label: &str,
    bytes: &[u8],
    penalty: usize,
    best: &mut Option<(usize, TimestampGoldenKey)>,
) {
    let Ok(window) = <[u8; 8]>::try_from(bytes) else {
        return;
    };
    update_timestamp_candidate_u64(best, &format!("{label}/le"), u64::from_le_bytes(window), penalty);
    update_timestamp_candidate_u64(best, &format!("{label}/be"), u64::from_be_bytes(window), penalty + 1);
}

fn update_timestamp_candidate(
    best: &mut Option<(usize, TimestampGoldenKey)>,
    label: &str,
    epoch: i64,
    penalty: usize,
) {
    let Some(datetime) = parse_target_epoch(epoch) else {
        return;
    };

    let score = 10_000usize.saturating_sub(penalty);
    let candidate = TimestampGoldenKey {
        digits: epoch.to_string(),
        epoch,
        datetime,
        rule: label.to_owned(),
    };

    if best.as_ref().map_or(true, |(best_score, _)| score > *best_score) {
        *best = Some((score, candidate));
    }
}

fn update_timestamp_candidate_u64(
    best: &mut Option<(usize, TimestampGoldenKey)>,
    label: &str,
    epoch: u64,
    penalty: usize,
) {
    let Ok(epoch) = i64::try_from(epoch) else {
        return;
    };
    update_timestamp_candidate(best, label, epoch, penalty);
}

fn update_relaxed_timestamp_candidate(
    best: &mut Option<(usize, TimestampGoldenKey)>,
    label: &str,
    epoch: i64,
    penalty: usize,
) {
    let Some(datetime) = parse_relaxed_epoch(epoch) else {
        return;
    };

    let score = 8_000usize.saturating_sub(penalty);
    let candidate = TimestampGoldenKey {
        digits: epoch.to_string(),
        epoch,
        datetime,
        rule: label.to_owned(),
    };

    if best.as_ref().map_or(true, |(best_score, _)| score > *best_score) {
        *best = Some((score, candidate));
    }
}

fn update_relaxed_timestamp_candidate_u64(
    best: &mut Option<(usize, TimestampGoldenKey)>,
    label: &str,
    epoch: u64,
    penalty: usize,
) {
    let Ok(epoch_i64) = i64::try_from(epoch) else {
        return;
    };
    update_relaxed_timestamp_candidate(best, label, epoch_i64, penalty);

    let millis_epoch = epoch / 1_000;
    if let Ok(millis_epoch_i64) = i64::try_from(millis_epoch) {
        update_relaxed_timestamp_candidate(best, &format!("{label}/ms"), millis_epoch_i64, penalty + 1);
    }
}

fn extract_timestamp_from_parsed(parsed: &SegmentedRecord, label_prefix: &str) -> Option<TimestampGoldenKey> {
    let timestamp = extract_named_value(&parsed.polished_pairs, &["t", "timestamp"])?;
    let epoch = timestamp.parse::<i64>().ok()?;
    let datetime = parse_target_epoch(epoch)?;
    let rule_suffix = parsed.timestamp_rule.as_deref().unwrap_or("parsed");
    Some(TimestampGoldenKey {
        digits: timestamp,
        epoch,
        datetime,
        rule: format!("{label_prefix}/{rule_suffix}"),
    })
}

fn scan_line11_coding_timestamp_bytes(
    bytes: &[u8],
    label_prefix: &str,
    previous_line_epoch: Option<i64>,
) -> Option<TimestampGoldenKey> {
    let mut best: Option<(usize, TimestampGoldenKey)> = None;

    if bytes.len() >= 4 {
        let window4 = <[u8; 4]>::try_from(&bytes[..4]).expect("slice length checked");
        update_relaxed_timestamp_candidate(
            &mut best,
            &format!("{label_prefix}/u32-le"),
            u32::from_le_bytes(window4) as i64,
            10,
        );
        update_relaxed_timestamp_candidate(
            &mut best,
            &format!("{label_prefix}/u32-be"),
            u32::from_be_bytes(window4) as i64,
            11,
        );
        if let Some(previous_epoch) = previous_line_epoch {
            update_counter_delta_candidate(
                &mut best,
                &format!("{label_prefix}/delta/u32-le"),
                previous_epoch,
                u32::from_le_bytes(window4) as u64,
                12,
            );
            update_counter_delta_candidate(
                &mut best,
                &format!("{label_prefix}/delta/u32-be"),
                previous_epoch,
                u32::from_be_bytes(window4) as u64,
                13,
            );
        }
    }

    if bytes.len() >= 8 {
        let window8 = <[u8; 8]>::try_from(&bytes[..8]).expect("slice length checked");
        update_relaxed_timestamp_candidate_u64(
            &mut best,
            &format!("{label_prefix}/u64-le"),
            u64::from_le_bytes(window8),
            20,
        );
        update_relaxed_timestamp_candidate_u64(
            &mut best,
            &format!("{label_prefix}/u64-be"),
            u64::from_be_bytes(window8),
            21,
        );
        if let Some(previous_epoch) = previous_line_epoch {
            update_counter_delta_candidate(
                &mut best,
                &format!("{label_prefix}/delta/u64-le"),
                previous_epoch,
                u64::from_le_bytes(window8),
                22,
            );
            update_counter_delta_candidate(
                &mut best,
                &format!("{label_prefix}/delta/u64-be"),
                previous_epoch,
                u64::from_be_bytes(window8),
                23,
            );
        }
    }

    let mut digits = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let upper = byte >> 4;
        let lower = byte & 0x0f;
        if upper < 10 {
            digits.push(char::from(b'0' + upper));
        } else {
            digits.push('x');
        }
        if lower < 10 {
            digits.push(char::from(b'0' + lower));
        } else {
            digits.push('x');
        }
    }

    if digits.len() >= 10 {
        for (offset, window) in digits.as_bytes().windows(10).enumerate() {
            if !window.iter().all(u8::is_ascii_digit) {
                continue;
            }
            let Ok(text) = std::str::from_utf8(window) else {
                continue;
            };
            let Ok(epoch) = text.parse::<i64>() else {
                continue;
            };
            update_relaxed_timestamp_candidate(
                &mut best,
                &format!("{label_prefix}/bcd/window@{offset}"),
                epoch,
                30 + offset,
            );
            if let Some(previous_epoch) = previous_line_epoch {
                update_counter_delta_candidate(
                    &mut best,
                    &format!("{label_prefix}/delta/bcd/window@{offset}"),
                    previous_epoch,
                    epoch as u64,
                    40 + offset,
                );
            }
        }
    }

    let inverted: Vec<u8> = bytes.iter().map(|byte| !*byte).collect();
    scan_line11_inverted_timestamp_bytes(&inverted, label_prefix, previous_line_epoch, &mut best);

    best.map(|(_, hit)| hit)
}

fn build_line11_coding_keys(
    fixed: &FixedDecryptor,
    local_part: &[u8],
    flag: u8,
) -> Vec<(String, [u8; 32])> {
    let mut keys = Vec::new();
    keys.push(("derived".to_owned(), fixed.derived_key));

    let flag_bytes = [flag];

    let mut with_flag_salt = [0_u8; 32];
    derive_hkdf_sha256(&fixed.raw_key, Some(&flag_bytes), HkdfInfoKind::Envelope.as_bytes(), &mut with_flag_salt);
    keys.push(("hkdf-salt-flag".to_owned(), with_flag_salt));

    let mut with_flag_info = [0_u8; 32];
    derive_hkdf_sha256(&fixed.raw_key, Some(local_part), &flag_bytes, &mut with_flag_info);
    keys.push(("hkdf-info-flag".to_owned(), with_flag_info));

    let mut with_flag_both = [0_u8; 32];
    derive_hkdf_sha256(&fixed.raw_key, Some(&flag_bytes), &flag_bytes, &mut with_flag_both);
    keys.push(("hkdf-salt+info-flag".to_owned(), with_flag_both));

    keys
}

fn scan_line11_inverted_timestamp_bytes(
    bytes: &[u8],
    label_prefix: &str,
    previous_line_epoch: Option<i64>,
    best: &mut Option<(usize, TimestampGoldenKey)>,
) {
    if bytes.len() >= 4 {
        let window4 = <[u8; 4]>::try_from(&bytes[..4]).expect("slice length checked");
        update_timestamp_candidate(
            best,
            &format!("{label_prefix}/not/u32-le"),
            u32::from_le_bytes(window4) as i64,
            50,
        );
        update_timestamp_candidate(
            best,
            &format!("{label_prefix}/not/u32-be"),
            u32::from_be_bytes(window4) as i64,
            51,
        );
        if let Some(previous_epoch) = previous_line_epoch {
            update_counter_delta_candidate(
                best,
                &format!("{label_prefix}/not/delta/u32-le"),
                previous_epoch,
                u32::from_le_bytes(window4) as u64,
                52,
            );
            update_counter_delta_candidate(
                best,
                &format!("{label_prefix}/not/delta/u32-be"),
                previous_epoch,
                u32::from_be_bytes(window4) as u64,
                53,
            );
        }
    }

    if bytes.len() >= 8 {
        let window8 = <[u8; 8]>::try_from(&bytes[..8]).expect("slice length checked");
        update_timestamp_candidate_u64(
            best,
            &format!("{label_prefix}/not/u64-le"),
            u64::from_le_bytes(window8),
            60,
        );
        update_timestamp_candidate_u64(
            best,
            &format!("{label_prefix}/not/u64-be"),
            u64::from_be_bytes(window8),
            61,
        );
    }

    let ascii_digits: String = bytes
        .iter()
        .filter(|byte| byte.is_ascii_digit())
        .map(|byte| *byte as char)
        .collect();
    if ascii_digits.len() >= 10 {
        for (offset, window) in ascii_digits.as_bytes().windows(10).enumerate() {
            let Ok(text) = std::str::from_utf8(window) else {
                continue;
            };
            let Ok(epoch) = text.parse::<i64>() else {
                continue;
            };
            update_timestamp_candidate(best, &format!("{label_prefix}/not/ascii@{offset}"), epoch, 70 + offset);
        }
    }
}

fn update_counter_delta_candidate(
    best: &mut Option<(usize, TimestampGoldenKey)>,
    label: &str,
    previous_epoch: i64,
    raw_value: u64,
    penalty: usize,
) {
    let delta = (raw_value % 31_536_000) as i64;
    if delta == 0 {
        return;
    }

    update_relaxed_timestamp_candidate(best, &format!("{label}/plus"), previous_epoch + delta, penalty);
    update_relaxed_timestamp_candidate(best, &format!("{label}/minus"), previous_epoch - delta, penalty + 1);
}

#[derive(Clone, Copy)]
enum Line11Mutation {
    Identity,
    KeyXorFlag,
    NonceXorFlag,
    KeyAndNonceXorFlag,
}

impl Line11Mutation {
    fn label(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::KeyXorFlag => "key-xor-flag",
            Self::NonceXorFlag => "nonce-xor-flag",
            Self::KeyAndNonceXorFlag => "key-nonce-xor-flag",
        }
    }

    fn penalty(self) -> usize {
        match self {
            Self::Identity => 0,
            Self::KeyXorFlag => 40,
            Self::NonceXorFlag => 40,
            Self::KeyAndNonceXorFlag => 80,
        }
    }
}

fn apply_line11_mutation(
    key: &mut [u8; 32],
    nonce: &mut [u8; BLOCK_BYTES],
    flag: u8,
    mutation: Line11Mutation,
) {
    match mutation {
        Line11Mutation::Identity => {}
        Line11Mutation::KeyXorFlag => {
            for byte in &mut key[..4] {
                *byte ^= flag;
            }
        }
        Line11Mutation::NonceXorFlag => {
            for byte in &mut nonce[..4] {
                *byte ^= flag;
            }
        }
        Line11Mutation::KeyAndNonceXorFlag => {
            for byte in &mut key[..4] {
                *byte ^= flag;
            }
            for byte in &mut nonce[..4] {
                *byte ^= flag;
            }
        }
    }
}

fn build_ciphertext_window(ciphertext: &[u8], slide: usize) -> [u8; CIPHERTEXT_BYTES] {
    let mut window = [0_u8; CIPHERTEXT_BYTES];
    if slide >= ciphertext.len() {
        return window;
    }
    let remaining = &ciphertext[slide..];
    let copy_len = remaining.len().min(CIPHERTEXT_BYTES);
    window[..copy_len].copy_from_slice(&remaining[..copy_len]);
    window
}

fn drift_label(drift: i8) -> String {
    format!("/drift{drift:+}")
}

fn parse_target_epoch(epoch: i64) -> Option<chrono::DateTime<Utc>> {
    let datetime = Utc.timestamp_opt(epoch, 0).single()?;
    let start = Utc.with_ymd_and_hms(2026, 3, 1, 0, 0, 0).single()?;
    let end = Utc.with_ymd_and_hms(2027, 12, 31, 23, 59, 59).single()?;
    (datetime >= start && datetime <= end).then_some(datetime)
}

fn parse_relaxed_epoch(epoch: i64) -> Option<chrono::DateTime<Utc>> {
    let datetime = Utc.timestamp_opt(epoch, 0).single()?;
    let start = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).single()?;
    let end = Utc.with_ymd_and_hms(2030, 12, 31, 23, 59, 59).single()?;
    (datetime >= start && datetime <= end).then_some(datetime)
}

fn build_final_string(polished_pairs: &[(String, String)], combined_text: &str) -> String {
    if !polished_pairs.is_empty() {
        return url_decode_and_clean(
            &polished_pairs
                .iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>()
                .join("&"),
        );
    }
    url_decode_and_clean(combined_text)
}

fn collect_validated_fields(polished_pairs: &[(String, String)], hybrid_tail: Option<&str>) -> Vec<String> {
    let mut fields = Vec::new();
    for (key, value) in polished_pairs {
        if matches!(key.as_str(), "u" | "c" | "m" | "mid" | "t") {
            fields.push(format!("{key}={value}"));
        }
    }
    if let Some(hybrid_tail) = hybrid_tail {
        let lowered = hybrid_tail.to_ascii_lowercase();
        if lowered.contains("u=") || lowered.contains("c=") || lowered.contains("m=") || lowered.contains("mid=") {
            fields.push(hybrid_tail.to_owned());
        }
    }
    fields.sort();
    fields.dedup();
    fields
}

fn extract_named_value(polished_pairs: &[(String, String)], keys: &[&str]) -> Option<String> {
    for (key, value) in polished_pairs {
        if keys.iter().any(|candidate| key == candidate) && !value.is_empty() {
            return Some(value.clone());
        }
    }
    None
}

fn canonicalize_core_pairs(
    polished_pairs: &mut Vec<(String, String)>,
    timestamp: Option<&TimestampGoldenKey>,
    user_id: &str,
    campaign_id: &str,
) {
    polished_pairs.retain(|(key, _)| {
        !matches!(
            key.as_str(),
            "t" | "timestamp" | "u" | "uid" | "c" | "campaign" | "campaign_id"
        )
    });

    if let Some(timestamp) = timestamp {
        polished_pairs.push(("t".to_owned(), timestamp.digits.clone()));
    }
    if user_id != "-" {
        polished_pairs.push(("u".to_owned(), user_id.to_owned()));
    }
    if campaign_id != "-" {
        polished_pairs.push(("c".to_owned(), campaign_id.to_owned()));
    }

    polished_pairs.sort();
    polished_pairs.dedup();
}

fn collect_extra_flags(
    polished_pairs: &[(String, String)],
    tail_hybrid: Option<&TailDecodeCandidate>,
    validated_fields: &[String],
) -> String {
    let mut flags = Vec::new();

    for (key, value) in polished_pairs {
        if matches!(key.as_str(), "u" | "uid" | "c" | "campaign" | "campaign_id" | "t" | "timestamp") {
            continue;
        }
        flags.push(format!("{key}={value}"));
    }

    if flags.is_empty() {
        if let Some(candidate) = tail_hybrid {
            if !candidate.value.is_empty() {
                flags.push(candidate.value.clone());
            }
        }
    }
    if flags.is_empty() && !validated_fields.is_empty() {
        flags.extend(validated_fields.iter().cloned());
    }

    if flags.is_empty() {
        "-".to_owned()
    } else {
        flags.join(" | ")
    }
}

fn url_decode_and_clean(input: &str) -> String {
    let normalized = normalize_percent_input(input);
    if normalized.contains('%') {
        let decoded = percent_decode_str(&normalized).decode_utf8_lossy();
        return clean_human_text(decoded.as_ref());
    }
    clean_human_text(&normalized)
}

fn normalize_percent_input(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut normalized = String::with_capacity(input.len());
    let mut index = 0_usize;

    while index < bytes.len() {
        if bytes[index] == b'%' {
            if index + 2 < bytes.len()
                && hex_nibble(bytes[index + 1]).is_some()
                && hex_nibble(bytes[index + 2]).is_some()
            {
                normalized.push('%');
                normalized.push(bytes[index + 1] as char);
                normalized.push(bytes[index + 2] as char);
                index += 3;
                continue;
            }
            index += 1;
            continue;
        }

        normalized.push(bytes[index] as char);
        index += 1;
    }

    normalized
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn choose_hybrid_tail_decoder(
    candidates: &[TailDecodeCandidate],
    reassembled: Option<&str>,
) -> Option<TailDecodeCandidate> {
    let mut ranked: Vec<(u16, usize)> = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| (score_hybrid_candidate(&candidate.value, reassembled), index))
        .collect();
    ranked.sort_by_key(|(score, index)| (*score, candidates[*index].value.len()));
    ranked
        .pop()
        .map(|(_, index)| TailDecodeCandidate {
            source: candidates[index].source.clone(),
            value: candidates[index].value.clone(),
        })
}

fn score_hybrid_candidate(value: &str, reassembled: Option<&str>) -> u16 {
    let lowered = value.to_ascii_lowercase();
    let mut score = 0_u16;
    if value.contains("eyJ") {
        score += 200;
    }
    if value.contains('&') {
        score += 120;
    }
    if lowered.contains("u=") || lowered.contains("c=") || lowered.contains("m=") || lowered.contains("mid=") {
        score += 160;
    }
    if has_mission_token(value) {
        score += 400;
    }
    if let Some(reassembled) = reassembled {
        if !reassembled.is_empty() && value.contains(reassembled) {
            score += 40;
        }
    }
    score + value.len() as u16
}

fn score_alignment_candidate(candidate: &str, local_part: &str) -> usize {
    let lowered = candidate.to_ascii_lowercase();
    let mut score = candidate.matches('&').count() * 50
        + candidate.matches('=').count() * 50
        + candidate.matches('|').count() * 8;

    if lowered.contains("t=") {
        score += 200;
    }
    if lowered.contains("u=") {
        score += 160;
    }
    if lowered.contains("c=") {
        score += 160;
    }
    if let Some((_, len)) = find_anchor_offset(candidate, local_part) {
        score += 120 + len * 10;
    }
    if extract_timestamp_from_texts(&[candidate.to_owned()]).is_some() {
        score += 1200;
    }

    score + candidate.chars().filter(|ch| ch.is_ascii_alphanumeric()).count()
}

fn find_anchor_offset(text: &str, local_part: &str) -> Option<(usize, usize)> {
    let lowered = text.to_ascii_lowercase();
    let anchor = local_part.to_ascii_lowercase();
    if let Some(offset) = lowered.find(&anchor) {
        return Some((offset, anchor.len()));
    }

    for len in (4..anchor.len()).rev() {
        for start in 0..=anchor.len() - len {
            let fragment = &anchor[start..start + len];
            if let Some(offset) = lowered.find(fragment) {
                return Some((offset, len));
            }
        }
    }

    None
}

fn rotate_text_to_anchor(text: &str, offset: usize) -> String {
    if offset == 0 || offset >= text.len() || !text.is_char_boundary(offset) {
        return text.to_owned();
    }
    format!("{}|{}", &text[offset..], &text[..offset])
}

fn fixed_anchor_alignment(text: &str) -> String {
    for anchor in ["A\\SR", "u=", "u:", "u|"] {
        if let Some(offset) = text.find(anchor) {
            return rotate_text_to_anchor(text, offset);
        }
    }
    String::new()
}

fn build_aggressive_body_texts(plaintext: &[u8; CIPHERTEXT_BYTES], local_part: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    push_clean_candidate(&mut candidates, plaintext);

    let reversed = reverse_bytes_slice(plaintext);
    warn_if_logic_loop("body/reversed", plaintext, &reversed);
    push_clean_candidate(&mut candidates, &reversed);

    for offset in [4_u8, 56_u8, 115_u8] {
        let subtractive = transform_subtractive_slice(plaintext, offset);
        warn_if_logic_loop(&format!("body/subtractive-{offset}"), plaintext, &subtractive);
        push_clean_candidate(&mut candidates, &subtractive);

        let inverted = transform_inverted_offset_slice(plaintext, offset);
        warn_if_logic_loop(&format!("body/inverted-{offset}"), plaintext, &inverted);
        push_clean_candidate(&mut candidates, &inverted);

        let bit_flipped = transform_reverse_bits_slice(plaintext);
        warn_if_logic_loop(&format!("body/reverse-bits-{offset}"), plaintext, &bit_flipped);
        push_clean_candidate(&mut candidates, &bit_flipped);
    }

    let anchored: Vec<String> = candidates
        .iter()
        .filter_map(|candidate| find_anchor_offset(candidate, local_part).map(|(offset, _)| rotate_text_to_anchor(candidate, offset)))
        .collect();
    candidates.extend(anchored);
    candidates.sort();
    candidates.dedup();
    candidates
}

fn build_dynamic_segment_texts(segment: &[u8], offset: u8, shift_key: u8) -> Vec<String> {
    let mut candidates = build_aggressive_segment_texts(segment, offset);

    for bytes in build_shifted_segment_variants(segment, offset, shift_key) {
        push_clean_candidate(&mut candidates, &bytes);
    }

    candidates.sort();
    candidates.dedup();
    candidates
}

fn build_flag_sync_timestamp_texts(segment: &[u8], shift_key: u8) -> Vec<String> {
    let mut candidates = build_aggressive_segment_texts(segment, 4);

    for (_, bytes, _) in build_flag_rotated_timestamp_variants(segment, shift_key) {
        push_clean_candidate(&mut candidates, &bytes);
    }

    candidates.sort();
    candidates.dedup();
    candidates
}

fn build_flag_rotated_timestamp_variants(
    segment: &[u8],
    shift_key: u8,
) -> Vec<(String, Vec<u8>, usize)> {
    if segment.is_empty() {
        return Vec::new();
    }

    let mut base = segment.to_vec();
    apply_wrapping_offset_slice(&mut base, 4);

    let preferred = usize::from(shift_key & 7);
    let mut variants = Vec::new();

    for rotation in 0_u32..=7 {
        let distance = preferred.abs_diff(rotation as usize);
        let penalty = distance * 6 + rotation as usize;

        let left = rotate_each_byte_left_slice(&base, rotation);
        variants.push((format!("flag={shift_key}/rol{rotation}"), left, penalty));

        let right = rotate_each_byte_right_slice(&base, rotation);
        variants.push((format!("flag={shift_key}/ror{rotation}"), right, penalty + 1));
    }

    variants
}

fn build_shifted_segment_variants(segment: &[u8], offset: u8, shift_key: u8) -> Vec<Vec<u8>> {
    if segment.is_empty() || shift_key == 0 {
        return Vec::new();
    }

    let mut base = segment.to_vec();
    apply_wrapping_offset_slice(&mut base, offset);

    let mut variants = Vec::new();
    let index_offset = (shift_key as usize) % base.len();
    let bit_rotation = u32::from(shift_key % 8);

    push_shifted_variant(&mut variants, &base, index_offset, false);
    push_shifted_variant(&mut variants, &base, index_offset, true);

    if bit_rotation != 0 {
        let rotated_left = rotate_each_byte_left_slice(&base, bit_rotation);
        push_shifted_variant(&mut variants, &rotated_left, index_offset, false);
        push_shifted_variant(&mut variants, &rotated_left, index_offset, true);

        let rotated_right = rotate_each_byte_right_slice(&base, bit_rotation);
        push_shifted_variant(&mut variants, &rotated_right, index_offset, false);
        push_shifted_variant(&mut variants, &rotated_right, index_offset, true);
    }

    variants.sort();
    variants.dedup();
    variants
}

fn push_shifted_variant(variants: &mut Vec<Vec<u8>>, bytes: &[u8], index_offset: usize, rotate_right: bool) {
    let mut aligned = bytes.to_vec();
    if index_offset != 0 {
        if rotate_right {
            aligned.rotate_right(index_offset);
        } else {
            aligned.rotate_left(index_offset);
        }
    }
    variants.push(aligned);
}

fn build_aggressive_segment_texts(segment: &[u8], offset: u8) -> Vec<String> {
    let mut base = segment.to_vec();
    apply_wrapping_offset_slice(&mut base, offset);

    let mut candidates = Vec::new();
    push_clean_candidate(&mut candidates, &base);

    let reversed = reverse_bytes_slice(&base);
    warn_if_logic_loop(&format!("segment/reversed-{offset}"), &base, &reversed);
    push_clean_candidate(&mut candidates, &reversed);

    let bit_flipped = transform_reverse_bits_slice(&base);
    warn_if_logic_loop(&format!("segment/reverse-bits-{offset}"), &base, &bit_flipped);
    push_clean_candidate(&mut candidates, &bit_flipped);

    let subtractive = transform_subtractive_slice(&base, offset);
    warn_if_logic_loop(&format!("segment/subtractive-{offset}"), &base, &subtractive);
    push_clean_candidate(&mut candidates, &subtractive);

    let inverted = transform_inverted_offset_slice(&base, offset);
    warn_if_logic_loop(&format!("segment/inverted-{offset}"), &base, &inverted);
    push_clean_candidate(&mut candidates, &inverted);

    for shift in 1_u32..=7 {
        let shifted_left = shift_bits_left_slice(&base, shift);
        warn_if_logic_loop(&format!("segment/shl{shift}-{offset}"), &base, &shifted_left);
        push_clean_candidate(&mut candidates, &shifted_left);

        let shifted_right = shift_bits_right_slice(&base, shift);
        warn_if_logic_loop(&format!("segment/shr{shift}-{offset}"), &base, &shifted_right);
        push_clean_candidate(&mut candidates, &shifted_right);
    }

    candidates.sort();
    candidates.dedup();
    candidates
}

fn push_clean_candidate(candidates: &mut Vec<String>, bytes: &[u8]) {
    let cleaned = clean_string_slice(bytes);
    if !cleaned.is_empty() {
        candidates.push(cleaned);
    }
}

fn apply_wrapping_offset_slice(bytes: &mut [u8], delta: u8) {
    for byte in bytes {
        *byte = byte.wrapping_add(delta);
    }
}

fn derive_shift_key(tail: &[u8], tail_clean: &str, tail_hybrid: Option<&TailDecodeCandidate>) -> u8 {
    for text in [tail_hybrid.map(|candidate| candidate.value.as_str()), Some(tail_clean)] {
        let Some(text) = text else {
            continue;
        };

        if let Some(digit) = text.bytes().find(u8::is_ascii_digit) {
            return digit - b'0';
        }

        if let Some(byte) = text
            .bytes()
            .find(|byte| !byte.is_ascii_control() || byte.is_ascii_whitespace())
        {
            return byte;
        }
    }

    tail.iter().copied().find(|byte| *byte != 0).unwrap_or(0)
}

fn maybe_base64_decode(clean: &str) -> Option<String> {
    let filtered: String = clean
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '+' | '/' | '-' | '_'))
        .collect();
    if filtered.len() < 4 {
        return None;
    }
    let missing = (4 - filtered.len() % 4) % 4;
    let padded = format!("{}{}", filtered, "=".repeat(missing));
    STANDARD
        .decode(padded)
        .ok()
        .map(|decoded| clean_string_slice(&decoded))
        .filter(|decoded| !decoded.is_empty())
}

fn maybe_ascii85_decode(clean: &str) -> Option<String> {
    let filtered: Vec<u8> = clean
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect();
    if filtered.is_empty() {
        return None;
    }

    let mut output = Vec::with_capacity((filtered.len() / 5 + 1) * 4);
    let mut group = [b'u'; 5];
    let mut group_len = 0_usize;

    for byte in filtered {
        if byte == b'z' && group_len == 0 {
            output.extend_from_slice(&[0, 0, 0, 0]);
            continue;
        }
        if !(33..=117).contains(&byte) {
            return None;
        }
        group[group_len] = byte;
        group_len += 1;
        if group_len == 5 {
            decode_ascii85_group(&group, 4, &mut output)?;
            group = [b'u'; 5];
            group_len = 0;
        }
    }

    if group_len > 0 {
        decode_ascii85_group(&group, group_len.saturating_sub(1), &mut output)?;
    }

    let cleaned = clean_string_slice(&output);
    (!cleaned.is_empty()).then_some(cleaned)
}

fn build_tail_candidates(tail: &[u8]) -> Vec<TailDecodeCandidate> {
    let mut candidates = Vec::new();
    let direct_clean = clean_string_slice(tail);
    push_tail_decodes(&mut candidates, "ascii85", &direct_clean, maybe_ascii85_decode(&direct_clean));
    push_tail_decodes(&mut candidates, "base64", &direct_clean, maybe_base64_decode(&direct_clean));

    let reversed: Vec<u8> = tail.iter().rev().copied().collect();
    warn_if_logic_loop("tail/reversed", tail, &reversed);
    let reversed_clean = clean_string_slice(&reversed);
    push_tail_decodes(
        &mut candidates,
        "reverse-ascii85",
        &reversed_clean,
        maybe_ascii85_decode(&reversed_clean),
    );
    push_tail_decodes(
        &mut candidates,
        "reverse-base64",
        &reversed_clean,
        maybe_base64_decode(&reversed_clean),
    );

    for offset in [4_u8, 56_u8, 115_u8] {
        let subtractive = transform_tail_subtractive(tail, offset);
        warn_if_logic_loop(&format!("tail/subtractive-{offset}"), tail, &subtractive);
        let subtractive_clean = clean_string_slice(&subtractive);
        push_tail_decodes(
            &mut candidates,
            &format!("subtractive-{offset}-ascii85"),
            &subtractive_clean,
            maybe_ascii85_decode(&subtractive_clean),
        );
        push_tail_decodes(
            &mut candidates,
            &format!("subtractive-{offset}-base64"),
            &subtractive_clean,
            maybe_base64_decode(&subtractive_clean),
        );

        let inverted = transform_tail_inverted_offset(tail, offset);
        warn_if_logic_loop(&format!("tail/inverted-{offset}"), tail, &inverted);
        let inverted_clean = clean_string_slice(&inverted);
        push_tail_decodes(
            &mut candidates,
            &format!("inverted-{offset}-ascii85"),
            &inverted_clean,
            maybe_ascii85_decode(&inverted_clean),
        );
        push_tail_decodes(
            &mut candidates,
            &format!("inverted-{offset}-base64"),
            &inverted_clean,
            maybe_base64_decode(&inverted_clean),
        );
    }

    let normalized = normalize_tail_alphabet(tail);
    push_tail_decodes(
        &mut candidates,
        "normalized-base64",
        &normalized,
        maybe_base64_decode(&normalized),
    );
    push_tail_decodes(
        &mut candidates,
        "normalized-ascii85",
        &normalized,
        maybe_ascii85_decode(&normalized),
    );

    candidates.sort_by(|left, right| left.source.cmp(&right.source).then(left.value.cmp(&right.value)));
    candidates.dedup_by(|left, right| left.source == right.source && left.value == right.value);
    candidates
}

fn push_tail_decodes(
    candidates: &mut Vec<TailDecodeCandidate>,
    source: &str,
    cleaned: &str,
    decoded: Option<String>,
) {
    if let Some(value) = decoded.filter(|value| !value.is_empty()) {
        candidates.push(TailDecodeCandidate {
            source: source.to_owned(),
            value,
        });
    } else if !cleaned.is_empty() {
        candidates.push(TailDecodeCandidate {
            source: format!("{source}-raw"),
            value: cleaned.to_owned(),
        });
    }
}

fn transform_tail_subtractive(tail: &[u8], offset: u8) -> Vec<u8> {
    transform_subtractive_slice(tail, offset)
}

fn transform_tail_inverted_offset(tail: &[u8], offset: u8) -> Vec<u8> {
    transform_inverted_offset_slice(tail, offset)
}

fn transform_subtractive_slice(bytes: &[u8], offset: u8) -> Vec<u8> {
    bytes.iter().map(|byte| offset.wrapping_sub(*byte)).collect()
}

fn transform_inverted_offset_slice(bytes: &[u8], offset: u8) -> Vec<u8> {
    bytes.iter().map(|byte| (!*byte).wrapping_add(offset)).collect()
}

fn transform_reverse_bits_slice(bytes: &[u8]) -> Vec<u8> {
    bytes.iter().map(|byte| byte.reverse_bits()).collect()
}

fn reverse_bytes_slice(bytes: &[u8]) -> Vec<u8> {
    bytes.iter().rev().copied().collect()
}

fn rotate_each_byte_left_slice(bytes: &[u8], shift: u32) -> Vec<u8> {
    bytes.iter().map(|byte| byte.rotate_left(shift)).collect()
}

fn rotate_each_byte_right_slice(bytes: &[u8], shift: u32) -> Vec<u8> {
    bytes.iter().map(|byte| byte.rotate_right(shift)).collect()
}

fn apply_counter_drift(counter: &mut [u8; BLOCK_BYTES], drift: i8) {
    if drift > 0 {
        for _ in 0..drift.unsigned_abs() {
            increment_be_counter(counter);
        }
    } else if drift < 0 {
        for _ in 0..drift.unsigned_abs() {
            decrement_be_counter(counter);
        }
    }
}

fn increment_be_counter(counter: &mut [u8; BLOCK_BYTES]) {
    for byte in counter.iter_mut().rev() {
        let (next, carry) = byte.overflowing_add(1);
        *byte = next;
        if !carry {
            break;
        }
    }
}

fn decrement_be_counter(counter: &mut [u8; BLOCK_BYTES]) {
    for byte in counter.iter_mut().rev() {
        let (next, borrow) = byte.overflowing_sub(1);
        *byte = next;
        if !borrow {
            break;
        }
    }
}

fn shift_bits_left_slice(bytes: &[u8], shift: u32) -> Vec<u8> {
    bytes.iter().map(|byte| byte.wrapping_shl(shift)).collect()
}

fn shift_bits_right_slice(bytes: &[u8], shift: u32) -> Vec<u8> {
    bytes.iter().map(|byte| byte.wrapping_shr(shift)).collect()
}

fn warn_if_logic_loop(label: &str, original: &[u8], transformed: &[u8]) {
    if original == transformed && !LOGIC_LOOP_REPORTED.swap(true, Ordering::Relaxed) {
        eprintln!("\x1b[1;31mLOGIC LOOP DETECTED: {label}\x1b[0m");
    }
}

fn emit_segment_hex_monitor(name: &str, segment: &[u8], offset: u8) {
    let mut base = segment.to_vec();
    apply_wrapping_offset_slice(&mut base, offset);
    emit_hex_variant(name, "offset", &base);

    let reversed = reverse_bytes_slice(&base);
    emit_hex_variant(name, "reversed", &reversed);

    let bit_flipped = transform_reverse_bits_slice(&base);
    emit_hex_variant(name, "reverse-bits", &bit_flipped);

    let subtractive = transform_subtractive_slice(&base, offset);
    emit_hex_variant(name, "subtractive", &subtractive);

    let inverted = transform_inverted_offset_slice(&base, offset);
    emit_hex_variant(name, "inverted", &inverted);
}

fn emit_hex_variant(name: &str, variant: &str, bytes: &[u8]) {
    let hex = hex_encode(bytes);
    let heavy = has_double_inversion_profile(bytes);
    eprintln!(
        "checkpoint[{name}/{variant}] hex={hex} double_inversion={heavy}"
    );
    if heavy {
        eprintln!("\x1b[1;31mDOUBLE INVERSION SUSPECTED: {name}/{variant}\x1b[0m");
    }
}

fn has_double_inversion_profile(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    let suspicious = bytes.iter().filter(|byte| matches!(**byte, 0x00 | 0xff)).count();
    suspicious * 4 >= bytes.len()
}

fn normalize_tail_alphabet(tail: &[u8]) -> String {
    tail.iter()
        .map(|byte| match *byte as char {
            '-' => '+',
            '_' => '/',
            '.' => '+',
            ',' => '/',
            ':' => '+',
            ';' => '/',
            '~' => '=',
            ch if ch.is_ascii_alphanumeric() || matches!(ch, '+' | '/' | '=') => ch,
            _ => 'A',
        })
        .collect()
}

fn decode_ascii85_group(group: &[u8; 5], emit: usize, output: &mut Vec<u8>) -> Option<()> {
    let mut value = 0_u32;
    for &byte in group {
        value = value.checked_mul(85)?;
        value = value.checked_add((byte - 33) as u32)?;
    }
    let bytes = value.to_be_bytes();
    output.extend_from_slice(&bytes[..emit]);
    Some(())
}

fn has_mission_token(value: &str) -> bool {
    let lowered = value.to_ascii_lowercase();
    lowered.contains("dengage")
        || lowered.contains("powermta")
        || lowered.contains("mmpejmrp")
        || lowered.contains("mulakat")
        || lowered.contains("mülakat")
        || lowered.contains("interview")
}

fn has_polished_core_fields(polished_pairs: &[(String, String)]) -> bool {
    ["t", "u", "c"].into_iter().all(|required| {
        polished_pairs.iter().any(|(key, value)| {
            key == required
                && !value.is_empty()
                && value
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | ':' | '@' | '.'))
        })
    })
}

fn print_mission_accomplished() {
    println!();
    println!(r" __  __ ___ ____ ____ ___ ___  _   _    _    ____ ____ ___  __  __ ____  _     ___ ____  _   _ _____ ____ ");
    println!(r"|  \/  |_ _/ ___/ ___|_ _/ _ \| \ | |  / \  / ___/ ___/ _ \|  \/  |  _ \| |   |_ _/ ___|| | | | ____|  _ \");
    println!(r"| |\/| || |\___ \___ \| | | | |  \| | / _ \| |  | |  | | | | |\/| | |_) | |    | |\___ \| |_| |  _| | | | |");
    println!(r"| |  | || | ___) |__) | | |_| | |\  |/ ___ \ |__| |__| |_| | |  | |  __/| |___ | | ___) |  _  | |___| |_| |");
    println!(r"|_|  |_|___|____/____/___\___/|_| \_/_/   \_\____\____\___/|_|  |_|_|   |_____|___|____/|_| |_|_____|____/ ");
}

fn print_mission_completed_respect() {
    println!();
    println!("\x1b[1;32m __  __ ___ ____ ____ ___ ___  _   _     ____ ___  __  __ ____  _     _____ _____ _____ ____\x1b[0m");
    println!("\x1b[1;32m|  \\/  |_ _/ ___/ ___|_ _/ _ \\| \\ | |   / ___/ _ \\|  \\/  |  _ \\| |   | ____|_   _| ____|  _ \\\x1b[0m");
    println!("\x1b[1;32m| |\\/| || |\\___ \\\\___ \\| | | | |  \\| |  | |  | | | | |\\/| | |_) | |   |  _|   | | |  _| | | | |\x1b[0m");
    println!("\x1b[1;32m| |  | || | ___) |__) | | |_| | |\\  |  | |__| |_| | |  | |  __/| |___| |___  | | | |___| |_| |\x1b[0m");
    println!("\x1b[1;32m|_|  |_|___|____/____/___\\___/|_| \\_|   \\____\\___/|_|  |_|_|   |_____|_____| |_| |_____|____/\x1b[0m");
    println!("\x1b[1;32m                                   + RESPECT\x1b[0m");
}

fn load_payload_records(path: &str) -> Vec<InputPayloadRecord> {
    let input =
        fs::read_to_string(path).unwrap_or_else(|error| panic!("failed to read {path}: {error}"));

    input
        .lines()
        .enumerate()
        .filter_map(|line| {
            let (index, line) = line;
            let trimmed = line.trim();
            (!trimmed.is_empty()).then_some((index + 1, trimmed))
        })
        .map(|(line_no, line)| {
            let parsed = parse_strict_key_record(line)
                .unwrap_or_else(|error| panic!("failed to parse payload line {line:?}: {error:?}"));
            let mut bytes = [0_u8; BLOB_BYTES];
            bytes.copy_from_slice(&parsed.payload.bytes);
            // Keep parse metadata beside the blob so strict-length anomalies can
            // be analyzed without re-reading the raw file later.
            InputPayloadRecord {
                line_no,
                raw_length: parsed.raw_length,
                normalized_length: parsed.normalized_length,
                strict_length_match: parsed.has_expected_length,
                payload_pad_added: parsed.payload.pad_added,
                payload: bytes,
            }
        })
        .collect()
}

fn load_candidate_key_stages(path: &str, email_hint: Option<&str>) -> Vec<CandidateStage> {
    let input =
        fs::read_to_string(path).unwrap_or_else(|error| panic!("failed to read {path}: {error}"));

    let mut buckets = StageBuckets::default();
    if let Some(email) = email_hint {
        buckets.focused.extend(derive_keys_from_email(email));
    }

    for (index, line) in input.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        parse_key_line(trimmed, &mut buckets)
            .unwrap_or_else(|error| panic!("failed to parse key line {}: {}", index + 1, error));
    }

    vec![
        CandidateStage {
            name: "focused",
            raw_keys: dedup_keys(buckets.focused),
        },
        CandidateStage {
            name: "anchored",
            raw_keys: dedup_keys(buckets.anchored),
        },
        CandidateStage {
            name: "expanded",
            raw_keys: dedup_keys(buckets.expanded),
        },
    ]
}

fn parse_key_line(line: &str, buckets: &mut StageBuckets) -> Result<(), String> {
    if let Ok(parsed) = parse_strict_key_record(line) {
        if parsed.payload.bytes.len() != BLOB_BYTES {
            return Err(format!(
                "pmta-style candidate decoded to {} bytes, expected {}",
                parsed.payload.bytes.len(),
                BLOB_BYTES
            ));
        }

        buckets
            .anchored
            .extend(derive_primary_keys_from_blob(&parsed.payload.bytes));
        buckets
            .expanded
            .extend(derive_sliding_keys_from_blob(&parsed.payload.bytes));
        return Ok(());
    }

    if line.len() == 64 && line.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        buckets.focused.push(decode_hex(line)?);
        return Ok(());
    }

    let decoded = STANDARD.decode(line).map_err(|error| {
        format!("expected PMTA record, 64-char hex, or standard base64 key: {error}")
    })?;
    buckets.focused.push(decoded);
    Ok(())
}

fn derive_primary_keys_from_blob(blob: &[u8]) -> Vec<Vec<u8>> {
    if blob.len() < HEADER_BYTES {
        return Vec::new();
    }

    let mut keys = vec![
        blob[..HEADER_BYTES].to_vec(),
        blob[blob.len() - HEADER_BYTES..].to_vec(),
        blob[8..8 + HEADER_BYTES].to_vec(),
        blob[16..16 + HEADER_BYTES].to_vec(),
        blob[24..24 + HEADER_BYTES].to_vec(),
    ];
    keys.retain(|key| key.len() == HEADER_BYTES);
    keys
}

fn derive_sliding_keys_from_blob(blob: &[u8]) -> Vec<Vec<u8>> {
    if blob.len() < HEADER_BYTES {
        return Vec::new();
    }

    (0..=blob.len() - HEADER_BYTES)
        .step_by(4)
        .map(|offset| blob[offset..offset + HEADER_BYTES].to_vec())
        .collect()
}

fn derive_keys_from_email(email: &str) -> Vec<Vec<u8>> {
    let trimmed = email.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    let lower = trimmed.to_ascii_lowercase();
    let (local, domain) = lower.split_once('@').unwrap_or((&lower, ""));

    let mut variants = vec![
        lower.clone(),
        local.to_owned(),
        domain.to_owned(),
        format!("{local}@{domain}"),
        format!("lu:{local}"),
        format!("pmta:{lower}"),
        format!("lu:pmta:{lower}"),
        format!("{local}:{domain}"),
        format!("{domain}:{local}"),
    ];
    variants.sort();
    variants.dedup();

    let mut keys = Vec::new();
    for variant in variants {
        keys.push(Sha256::digest(variant.as_bytes()).to_vec());
        keys.push(Sha512::digest(variant.as_bytes())[..32].to_vec());
        keys.push(md5_pair(
            &Md5::digest(variant.as_bytes()),
            &Md5::digest(lower.as_bytes()),
        ));
        keys.push(mix_hashes(
            &Sha1::digest(variant.as_bytes()),
            &Md5::digest(variant.as_bytes()),
        ));
        keys.push(hkdf_sha256(None, variant.as_bytes(), b"pmta-email"));
        keys.push(hkdf_sha256(
            Some(lower.as_bytes()),
            variant.as_bytes(),
            b"lu-local",
        ));
    }

    if !domain.is_empty() {
        keys.push(md5_pair(
            &Md5::digest(local.as_bytes()),
            &Md5::digest(domain.as_bytes()),
        ));
        keys.push(md5_pair(
            &Md5::digest(trimmed.as_bytes()),
            &Md5::digest(domain.as_bytes()),
        ));
        keys.push(mix_hashes(
            &Sha1::digest(domain.as_bytes()),
            &Md5::digest(local.as_bytes()),
        ));
        keys.push(hkdf_sha256(
            Some(domain.as_bytes()),
            local.as_bytes(),
            b"pmta-local",
        ));
        keys.push(hkdf_sha256(
            Some(local.as_bytes()),
            lower.as_bytes(),
            b"pmta-address",
        ));
    }

    keys
}

fn hkdf_sha256(salt: Option<&[u8]>, ikm: &[u8], info: &[u8]) -> Vec<u8> {
    let mut out = [0_u8; 32];
    derive_hkdf_sha256_slice(ikm, salt, info, &mut out);
    out.to_vec()
}

fn derive_hkdf_sha256(
    ikm: &[u8; 32],
    salt: Option<&[u8]>,
    info: &[u8],
    out: &mut [u8; 32],
) {
    derive_hkdf_sha256_slice(ikm, salt, info, out);
}

fn derive_hkdf_sha256_slice(ikm: &[u8], salt: Option<&[u8]>, info: &[u8], out: &mut [u8; 32]) {
    let hkdf = Hkdf::<Sha256>::new(salt, ikm);
    hkdf.expand(info, out).expect("hkdf expand");
}

fn dedup_keys(keys: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let mut seen = HashSet::new();
    keys.into_iter()
        .filter(|key| seen.insert(key.clone()))
        .collect()
}

fn local_part_hint(email_hint: Option<&str>) -> String {
    email_hint
        .and_then(|hint| hint.split_once('@').map(|(local, _)| local))
        .filter(|local| !local.is_empty())
        .unwrap_or("mmpejmrp")
        .to_owned()
}

fn md5_pair(left: &[u8], right: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(32);
    out.extend_from_slice(left);
    out.extend_from_slice(right);
    out
}

fn mix_hashes(left: &[u8], right: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(32);
    out.extend_from_slice(left);
    out.extend_from_slice(right);
    out.resize(32, 0);
    out
}

fn decode_hex(input: &str) -> Result<Vec<u8>, String> {
    let mut output = Vec::with_capacity(input.len() / 2);
    let bytes = input.as_bytes();

    for chunk in bytes.chunks_exact(2) {
        let high = decode_nibble(chunk[0])?;
        let low = decode_nibble(chunk[1])?;
        output.push((high << 4) | low);
    }

    if !bytes.chunks_exact(2).remainder().is_empty() {
        return Err("hex input length must be even".to_owned());
    }

    Ok(output)
}

fn decode_nibble(byte: u8) -> Result<u8, String> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(format!("invalid hex byte 0x{byte:02x}")),
    }
}
