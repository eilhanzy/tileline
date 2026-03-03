use crate::scheduler::TaskId;
use crate::{CorePreference, MpsScheduler, NativeTask, TaskPriority};
use aes::cipher::{KeyIvInit, StreamCipher};
use aes::{Aes128, Aes256};
use crossbeam::channel::{self, Receiver, TryRecvError};
use ctr::Ctr128BE;
use hmac::{Hmac, Mac};
use md5::Md5;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub const HEADER_BYTES: usize = 32;
pub const CIPHERTEXT_BYTES: usize = 32;
pub const TAG_BYTES: usize = 16;
pub const BLOB_BYTES: usize = HEADER_BYTES + CIPHERTEXT_BYTES + TAG_BYTES;
pub const CTR_NONCE_BYTES: usize = 16;
pub const HEURISTIC_SCORE_THRESHOLD: u16 = 30;
pub const MAC_VALIDATION_SCORE_THRESHOLD: u16 = 40;
pub const BINARY_ENTROPY_THRESHOLD_MILLI: u16 = 4000;

const PREFIX_CONTROL_WINDOW: usize = 8;
const PREFIX_CONTROL_REJECT_THRESHOLD: u8 = 3;
const PREVIEW_BYTES: usize = CIPHERTEXT_BYTES * 4;
const PINNED_MATRIX_NONCE_MAX_OFFSET: usize = 64;
const PINNED_MATRIX_NONCE_LENGTHS: [usize; 3] = [8, 12, 16];
const PINNED_MATRIX_LOCAL_PART: &[u8] = b"mmpejmrp";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PinnedMatrixTarget {
    blob_index: usize,
    key_index: usize,
    decrypt_nonce_offset: usize,
    blob_transform: BlobTransform,
    key_transform: KeyTransform,
}

const PINNED_MATRIX_TARGETS: [PinnedMatrixTarget; 2] = [
    PinnedMatrixTarget {
        blob_index: 0,
        key_index: 42,
        decrypt_nonce_offset: 10,
        blob_transform: BlobTransform::Identity,
        key_transform: KeyTransform::Identity,
    },
    PinnedMatrixTarget {
        blob_index: 4,
        key_index: 157,
        decrypt_nonce_offset: 5,
        blob_transform: BlobTransform::Identity,
        key_transform: KeyTransform::Identity,
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CipherAlgorithm {
    Aes256Ctr,
    Aes128CtrSplit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FindingKind {
    ExactUtf8,
    Heuristic,
    Binary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FindingPriority {
    High,
    Normal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlobTransform {
    Identity,
    ReverseCiphertext,
    SwapU32,
    SwapU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyTransform {
    Identity,
    Reversed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacAlgorithm {
    HmacMd5,
    HmacSha256Truncated16,
    RawMd5,
    RawSha256Truncated16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacKeyStrategy {
    DirectCandidateKey,
    CandidateTail16,
    DerivedHmacSha256,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacScope {
    Ciphertext,
    HeaderCiphertext,
    Plaintext,
    PlaintextNonce,
    HeaderPlaintext,
    HeaderPlaintextCiphertext,
    HeaderPlaintextNonce,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacPadding {
    None,
    ZeroTrimmed,
    Pkcs7Trimmed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacKeyTransform {
    Identity,
    Reversed,
    SwapU32,
    SwapU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacContextInjection {
    None,
    LocalPart,
    LocalPartMd5,
    LocalPartSha256,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MacValidation {
    pub algorithm: MacAlgorithm,
    pub key_strategy: MacKeyStrategy,
    pub scope: MacScope,
    pub nonce_offset: Option<u8>,
    pub nonce_length: Option<u8>,
    pub padding: MacPadding,
    pub mac_key_transform: MacKeyTransform,
    pub context_injection: MacContextInjection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeuristicSignal {
    pub score: u16,
    pub printable_count: u8,
    pub equals_count: u8,
    pub ampersand_count: u8,
    pub t_eq_hits: u8,
    pub u_eq_hits: u8,
    pub c_eq_hits: u8,
    pub digit_run_count: u8,
    pub longest_digit_run: u8,
    pub prefix_control_count: u8,
    pub entropy_milli_bits: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BruteForceFinding {
    pub blob_index: usize,
    pub key_index: usize,
    pub algorithm: CipherAlgorithm,
    pub blob_transform: BlobTransform,
    pub key_transform: KeyTransform,
    pub nonce_offset: usize,
    pub nonce_length: usize,
    pub use_header_aad: bool,
    pub kind: FindingKind,
    pub priority: FindingPriority,
    pub heuristic: HeuristicSignal,
    pub mac_validation: Option<MacValidation>,
    pub decoded_len: u8,
    pub decoded_bytes: [u8; CIPHERTEXT_BYTES],
}

pub type BruteForceMatch = BruteForceFinding;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntropyCheck {
    pub shannon_bits_per_symbol: f32,
    pub unique_bytes: u8,
    pub looks_random: bool,
}

#[derive(Debug)]
pub enum BruteForceInputError {
    EmptyKeys,
    InvalidBlobLength { index: usize, length: usize },
    InvalidKeyLength { index: usize, length: usize },
    MissingPinnedTarget { blob_index: usize, key_index: usize },
}

#[derive(Debug)]
pub struct BruteForceScanHandle {
    pub submitted_task_ids: Vec<TaskId>,
    receiver: Receiver<BruteForceFinding>,
    found: Arc<AtomicBool>,
}

impl BruteForceScanHandle {
    pub fn try_recv(&self) -> Result<BruteForceFinding, TryRecvError> {
        self.receiver.try_recv()
    }

    pub fn found(&self) -> bool {
        self.found.load(Ordering::Acquire)
    }
}

pub fn normalize_candidate_keys<T>(
    keys: impl IntoIterator<Item = T>,
) -> Result<Vec<[u8; 32]>, BruteForceInputError>
where
    T: AsRef<[u8]>,
{
    let mut normalized = Vec::new();
    for (index, key) in keys.into_iter().enumerate() {
        let bytes = key.as_ref();
        if bytes.len() != 32 {
            return Err(BruteForceInputError::InvalidKeyLength {
                index,
                length: bytes.len(),
            });
        }

        let mut normalized_key = [0_u8; 32];
        normalized_key.copy_from_slice(bytes);
        normalized.push(normalized_key);
    }

    if normalized.is_empty() {
        return Err(BruteForceInputError::EmptyKeys);
    }

    Ok(normalized)
}

pub fn submit_payload_bruteforce_scan(
    scheduler: &MpsScheduler,
    payloads: Vec<Arc<[u8]>>,
    candidate_keys: Vec<[u8; 32]>,
) -> Result<BruteForceScanHandle, BruteForceInputError> {
    if candidate_keys.is_empty() {
        return Err(BruteForceInputError::EmptyKeys);
    }

    for (index, payload) in payloads.iter().enumerate() {
        if payload.len() != BLOB_BYTES {
            return Err(BruteForceInputError::InvalidBlobLength {
                index,
                length: payload.len(),
            });
        }
    }

    let found = Arc::new(AtomicBool::new(false));
    let shared_keys = Arc::new(candidate_keys);
    let (tx, rx) = channel::unbounded();

    let tasks: Vec<NativeTask> = payloads
        .into_iter()
        .enumerate()
        .map(|(blob_index, payload)| {
            let tx = tx.clone();
            let found = Arc::clone(&found);
            let keys = Arc::clone(&shared_keys);

            Box::new(move || {
                if let Some(finding) = scan_blob(blob_index, payload.as_ref(), &keys, &found) {
                    let _ = tx.send(finding);
                }
            }) as NativeTask
        })
        .collect();
    drop(tx);

    let submitted_task_ids =
        scheduler.submit_batch_native(TaskPriority::Critical, CorePreference::Performance, tasks);

    Ok(BruteForceScanHandle {
        submitted_task_ids,
        receiver: rx,
        found,
    })
}

pub fn submit_pinned_nonce_matrix_scan(
    scheduler: &MpsScheduler,
    payloads: Vec<Arc<[u8]>>,
    candidate_keys: Vec<[u8; 32]>,
) -> Result<BruteForceScanHandle, BruteForceInputError> {
    if candidate_keys.is_empty() {
        return Err(BruteForceInputError::EmptyKeys);
    }

    for (index, payload) in payloads.iter().enumerate() {
        if payload.len() != BLOB_BYTES {
            return Err(BruteForceInputError::InvalidBlobLength {
                index,
                length: payload.len(),
            });
        }
    }

    for target in PINNED_MATRIX_TARGETS {
        if payloads.get(target.blob_index).is_none()
            || candidate_keys.get(target.key_index).is_none()
        {
            return Err(BruteForceInputError::MissingPinnedTarget {
                blob_index: target.blob_index,
                key_index: target.key_index,
            });
        }
    }

    let found = Arc::new(AtomicBool::new(false));
    let shared_keys = Arc::new(candidate_keys);
    let shared_payloads = Arc::new(payloads);
    let (tx, rx) = channel::unbounded();

    let tasks: Vec<NativeTask> = PINNED_MATRIX_TARGETS
        .iter()
        .copied()
        .map(|target| {
            let tx = tx.clone();
            let found = Arc::clone(&found);
            let keys = Arc::clone(&shared_keys);
            let payloads = Arc::clone(&shared_payloads);

            Box::new(move || {
                let payload = payloads[target.blob_index].as_ref();
                let key = &keys[target.key_index];
                if let Some(finding) = scan_pinned_matrix_target(target, payload, key, &found) {
                    let _ = tx.send(finding);
                }
            }) as NativeTask
        })
        .collect();
    drop(tx);

    let submitted_task_ids =
        scheduler.submit_batch_native(TaskPriority::Critical, CorePreference::Performance, tasks);

    Ok(BruteForceScanHandle {
        submitted_task_ids,
        receiver: rx,
        found,
    })
}

pub fn entropy_check(bytes: &[u8]) -> EntropyCheck {
    if bytes.is_empty() {
        return EntropyCheck {
            shannon_bits_per_symbol: 0.0,
            unique_bytes: 0,
            looks_random: false,
        };
    }

    let mut counts = [0_u8; 256];
    let mut unique = 0_u8;
    for &byte in bytes {
        let count = &mut counts[byte as usize];
        if *count == 0 {
            unique = unique.saturating_add(1);
        }
        *count = count.saturating_add(1);
    }

    let len = bytes.len() as f32;
    let mut entropy = 0.0_f32;
    for count in counts {
        if count == 0 {
            continue;
        }

        let p = count as f32 / len;
        entropy -= p * p.log2();
    }

    EntropyCheck {
        shannon_bits_per_symbol: entropy,
        unique_bytes: unique,
        looks_random: unique >= 12 && entropy >= 3.4,
    }
}

fn scan_blob(
    blob_index: usize,
    blob: &[u8],
    candidate_keys: &[[u8; 32]],
    found: &AtomicBool,
) -> Option<BruteForceFinding> {
    let mut best: Option<BruteForceFinding> = None;
    let mut transformed_blob = [0_u8; BLOB_BYTES];
    let mut reversed_key = [0_u8; 32];

    for blob_transform in [
        BlobTransform::Identity,
        BlobTransform::ReverseCiphertext,
        BlobTransform::SwapU32,
        BlobTransform::SwapU64,
    ] {
        let working_blob: &[u8] = if blob_transform == BlobTransform::Identity {
            blob
        } else {
            transformed_blob.copy_from_slice(blob);
            apply_blob_transform(blob_transform, &mut transformed_blob);
            &transformed_blob
        };

        let header = &working_blob[..HEADER_BYTES];
        let ciphertext = &working_blob[HEADER_BYTES..HEADER_BYTES + CIPHERTEXT_BYTES];

        for (key_index, key) in candidate_keys.iter().enumerate() {
            if found.load(Ordering::Acquire) {
                break;
            }

            for key_transform in [KeyTransform::Identity, KeyTransform::Reversed] {
                let key_bytes: &[u8; 32] = match key_transform {
                    KeyTransform::Identity => key,
                    KeyTransform::Reversed => {
                        reversed_key.copy_from_slice(key);
                        reversed_key.reverse();
                        &reversed_key
                    }
                };

                for nonce_offset in 0..=HEADER_BYTES - CTR_NONCE_BYTES {
                    if found.load(Ordering::Acquire) {
                        break;
                    }

                    let nonce = &header[nonce_offset..nonce_offset + CTR_NONCE_BYTES];
                    if let Some(mut candidate) = take_candidate(
                        blob_index,
                        key_index,
                        blob_transform,
                        key_transform,
                        nonce_offset,
                        CipherAlgorithm::Aes256Ctr,
                        key_bytes,
                        ciphertext,
                        nonce,
                    ) {
                        if candidate.heuristic.score >= MAC_VALIDATION_SCORE_THRESHOLD {
                            candidate.mac_validation = validate_candidate_mac(
                                CipherAlgorithm::Aes256Ctr,
                                key_bytes,
                                blob,
                                &candidate.decoded_bytes,
                            );
                            print_candidate_hit(&candidate);
                        }
                        if should_short_circuit(found, &candidate) {
                            print_validated_hit(&candidate);
                            return Some(candidate);
                        }
                        update_best(&mut best, candidate);
                    }

                    if let Some(mut candidate) = take_candidate(
                        blob_index,
                        key_index,
                        blob_transform,
                        key_transform,
                        nonce_offset,
                        CipherAlgorithm::Aes128CtrSplit,
                        key_bytes,
                        ciphertext,
                        nonce,
                    ) {
                        if candidate.heuristic.score >= MAC_VALIDATION_SCORE_THRESHOLD {
                            candidate.mac_validation = validate_candidate_mac(
                                CipherAlgorithm::Aes128CtrSplit,
                                key_bytes,
                                blob,
                                &candidate.decoded_bytes,
                            );
                            print_candidate_hit(&candidate);
                        }
                        if should_short_circuit(found, &candidate) {
                            print_validated_hit(&candidate);
                            return Some(candidate);
                        }
                        update_best(&mut best, candidate);
                    }
                }
            }
        }
    }

    best
}

fn scan_pinned_matrix_target(
    target: PinnedMatrixTarget,
    blob: &[u8],
    key: &[u8; 32],
    found: &AtomicBool,
) -> Option<BruteForceFinding> {
    if found.load(Ordering::Acquire) {
        return None;
    }

    let mut transformed_blob = [0_u8; BLOB_BYTES];
    let working_blob: &[u8] = if target.blob_transform == BlobTransform::Identity {
        blob
    } else {
        transformed_blob.copy_from_slice(blob);
        apply_blob_transform(target.blob_transform, &mut transformed_blob);
        &transformed_blob
    };

    let mut reversed_key = [0_u8; 32];
    let key_bytes: &[u8; 32] = match target.key_transform {
        KeyTransform::Identity => key,
        KeyTransform::Reversed => {
            reversed_key.copy_from_slice(key);
            reversed_key.reverse();
            &reversed_key
        }
    };

    let header = &working_blob[..HEADER_BYTES];
    let ciphertext = &working_blob[HEADER_BYTES..HEADER_BYTES + CIPHERTEXT_BYTES];
    let decrypt_nonce =
        &header[target.decrypt_nonce_offset..target.decrypt_nonce_offset + CTR_NONCE_BYTES];
    let plaintext = try_aes128_ctr_split(key_bytes, decrypt_nonce, ciphertext)?;
    let mut candidate = evaluate_candidate(
        target.blob_index,
        target.key_index,
        CipherAlgorithm::Aes128CtrSplit,
        target.blob_transform,
        target.key_transform,
        target.decrypt_nonce_offset,
        &plaintext,
    )?;

    candidate.mac_validation = validate_pinned_nonce_matrix_mac(key_bytes, blob, &plaintext, found);
    if candidate.mac_validation.is_some() && should_short_circuit(found, &candidate) {
        print_validated_hit(&candidate);
        return Some(candidate);
    }

    None
}

fn take_candidate(
    blob_index: usize,
    key_index: usize,
    blob_transform: BlobTransform,
    key_transform: KeyTransform,
    nonce_offset: usize,
    algorithm: CipherAlgorithm,
    key_bytes: &[u8; 32],
    ciphertext: &[u8],
    nonce: &[u8],
) -> Option<BruteForceFinding> {
    let plaintext = match algorithm {
        CipherAlgorithm::Aes256Ctr => try_aes256_ctr(key_bytes, nonce, ciphertext),
        CipherAlgorithm::Aes128CtrSplit => try_aes128_ctr_split(key_bytes, nonce, ciphertext),
    }?;

    evaluate_candidate(
        blob_index,
        key_index,
        algorithm,
        blob_transform,
        key_transform,
        nonce_offset,
        &plaintext,
    )
}

fn apply_blob_transform(transform: BlobTransform, blob: &mut [u8; BLOB_BYTES]) {
    match transform {
        BlobTransform::Identity => {}
        BlobTransform::ReverseCiphertext => {
            blob[HEADER_BYTES..HEADER_BYTES + CIPHERTEXT_BYTES].reverse();
        }
        BlobTransform::SwapU32 => swap_chunk_endianness(blob, 4),
        BlobTransform::SwapU64 => swap_chunk_endianness(blob, 8),
    }
}

fn swap_chunk_endianness(blob: &mut [u8; BLOB_BYTES], chunk_len: usize) {
    for chunk in blob.chunks_exact_mut(chunk_len) {
        chunk.reverse();
    }
}

fn evaluate_candidate(
    blob_index: usize,
    key_index: usize,
    algorithm: CipherAlgorithm,
    blob_transform: BlobTransform,
    key_transform: KeyTransform,
    nonce_offset: usize,
    plaintext_bytes: &[u8; CIPHERTEXT_BYTES],
) -> Option<BruteForceFinding> {
    let prefix_control_count = prefix_control_count(plaintext_bytes);
    if prefix_control_count >= PREFIX_CONTROL_REJECT_THRESHOLD {
        return None;
    }

    let mut heuristic = analyze_query_string_bytes(plaintext_bytes);
    heuristic.prefix_control_count = prefix_control_count;
    heuristic.entropy_milli_bits =
        (entropy_check(plaintext_bytes).shannon_bits_per_symbol * 1000.0).round() as u16;

    let kind = if std::str::from_utf8(plaintext_bytes).is_ok()
        && heuristic.score >= HEURISTIC_SCORE_THRESHOLD
    {
        FindingKind::ExactUtf8
    } else if heuristic.score >= HEURISTIC_SCORE_THRESHOLD {
        FindingKind::Heuristic
    } else if heuristic.entropy_milli_bits < BINARY_ENTROPY_THRESHOLD_MILLI {
        FindingKind::Binary
    } else {
        return None;
    };

    Some(BruteForceFinding {
        blob_index,
        key_index,
        algorithm,
        blob_transform,
        key_transform,
        nonce_offset,
        nonce_length: CTR_NONCE_BYTES,
        use_header_aad: false,
        kind,
        priority: if heuristic.score >= HEURISTIC_SCORE_THRESHOLD {
            FindingPriority::High
        } else {
            FindingPriority::Normal
        },
        heuristic,
        mac_validation: None,
        decoded_len: CIPHERTEXT_BYTES as u8,
        decoded_bytes: *plaintext_bytes,
    })
}

fn analyze_query_string_bytes(bytes: &[u8; CIPHERTEXT_BYTES]) -> HeuristicSignal {
    let mut signal = HeuristicSignal {
        score: 0,
        printable_count: 0,
        equals_count: 0,
        ampersand_count: 0,
        t_eq_hits: 0,
        u_eq_hits: 0,
        c_eq_hits: 0,
        digit_run_count: 0,
        longest_digit_run: 0,
        prefix_control_count: 0,
        entropy_milli_bits: 0,
    };

    let mut digit_run_len = 0_u8;

    for index in 0..bytes.len() {
        let byte = bytes[index];
        if is_printable_ascii(byte) {
            signal.printable_count = signal.printable_count.saturating_add(1);
            signal.score = signal.score.saturating_add(1);
        }

        match byte {
            b'=' => {
                signal.equals_count = signal.equals_count.saturating_add(1);
                signal.score = signal.score.saturating_add(5);
            }
            b'&' => {
                signal.ampersand_count = signal.ampersand_count.saturating_add(1);
                signal.score = signal.score.saturating_add(5);
            }
            b'0'..=b'9' => {
                digit_run_len = digit_run_len.saturating_add(1);
            }
            _ => {
                score_digit_run(&mut signal, digit_run_len);
                digit_run_len = 0;
            }
        }

        if index + 1 >= bytes.len() {
            continue;
        }

        match (byte, bytes[index + 1]) {
            (b't', b'=') => {
                signal.t_eq_hits = signal.t_eq_hits.saturating_add(1);
                signal.score = signal.score.saturating_add(10);
            }
            (b'u', b'=') => {
                signal.u_eq_hits = signal.u_eq_hits.saturating_add(1);
                signal.score = signal.score.saturating_add(6);
            }
            (b'c', b'=') => {
                signal.c_eq_hits = signal.c_eq_hits.saturating_add(1);
                signal.score = signal.score.saturating_add(6);
            }
            _ => {}
        }
    }

    score_digit_run(&mut signal, digit_run_len);
    signal
}

fn score_digit_run(signal: &mut HeuristicSignal, run_len: u8) {
    if run_len < 2 {
        return;
    }

    signal.digit_run_count = signal.digit_run_count.saturating_add(1);
    signal.longest_digit_run = signal.longest_digit_run.max(run_len);
    signal.score = signal
        .score
        .saturating_add(run_len.saturating_sub(1) as u16);

    if run_len >= 6 {
        signal.score = signal.score.saturating_add(4);
    }
    if run_len >= 9 {
        signal.score = signal.score.saturating_add(6);
    }
}

fn prefix_control_count(bytes: &[u8; CIPHERTEXT_BYTES]) -> u8 {
    let mut count = 0_u8;
    for &byte in bytes.iter().take(PREFIX_CONTROL_WINDOW) {
        if byte < 0x20 || byte == 0x7f {
            count = count.saturating_add(1);
        }
    }
    count
}

fn should_short_circuit(found: &AtomicBool, finding: &BruteForceFinding) -> bool {
    if finding.mac_validation.is_some() {
        return found
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok();
    }

    false
}

fn update_best(best: &mut Option<BruteForceFinding>, candidate: BruteForceFinding) {
    let candidate_rank = finding_rank(&candidate);
    let current_rank = best
        .as_ref()
        .map(finding_rank)
        .unwrap_or((0_u8, 0_u16, 0_u8, 0_u16, 0_u8));
    if candidate_rank > current_rank {
        *best = Some(candidate);
    }
}

fn finding_rank(finding: &BruteForceFinding) -> (u8, u16, u8, u16, u8) {
    let class = match (finding.kind, finding.priority) {
        (_, _) if finding.mac_validation.is_some() => 5,
        (FindingKind::ExactUtf8, FindingPriority::High) => 4,
        (FindingKind::Heuristic, FindingPriority::High) => 3,
        (FindingKind::Binary, FindingPriority::High) => 2,
        (FindingKind::ExactUtf8, FindingPriority::Normal) => 1,
        (FindingKind::Heuristic, FindingPriority::Normal)
        | (FindingKind::Binary, FindingPriority::Normal) => 0,
    };
    (
        class,
        finding.heuristic.score,
        finding.heuristic.longest_digit_run,
        u16::MAX.saturating_sub(finding.heuristic.entropy_milli_bits),
        finding.mac_validation.is_some() as u8,
    )
}

fn print_candidate_hit(finding: &BruteForceFinding) {
    let bytes = &finding.decoded_bytes[..finding.decoded_len as usize];
    let mut preview = [0_u8; PREVIEW_BYTES];
    let preview_len = write_escaped_preview(bytes, &mut preview);
    let preview = std::str::from_utf8(&preview[..preview_len]).unwrap_or("<preview>");
    let signal = &finding.heuristic;

    println!(
        "score-hit score={} blob={} key={} algorithm={:?} blob_transform={:?} key_transform={:?} nonce@{}+{} printable={} t={} u={} c={} eq={} amp={} digit_runs={} longest_digits={} prefix_ctrl={} entropy={:.2} mac={:?} preview={}",
        signal.score,
        finding.blob_index + 1,
        finding.key_index + 1,
        finding.algorithm,
        finding.blob_transform,
        finding.key_transform,
        finding.nonce_offset,
        finding.nonce_length,
        signal.printable_count,
        signal.t_eq_hits,
        signal.u_eq_hits,
        signal.c_eq_hits,
        signal.equals_count,
        signal.ampersand_count,
        signal.digit_run_count,
        signal.longest_digit_run,
        signal.prefix_control_count,
        signal.entropy_milli_bits as f32 / 1000.0,
        finding.mac_validation,
        preview
    );
}

fn validate_candidate_mac(
    algorithm: CipherAlgorithm,
    key: &[u8; 32],
    original_blob: &[u8],
    plaintext: &[u8; CIPHERTEXT_BYTES],
) -> Option<MacValidation> {
    let expected_mac = &original_blob[HEADER_BYTES + CIPHERTEXT_BYTES..BLOB_BYTES];
    let header = &original_blob[..HEADER_BYTES];
    let ciphertext = &original_blob[HEADER_BYTES..HEADER_BYTES + CIPHERTEXT_BYTES];
    let mut derived_mac_key = [0_u8; 32];
    derive_mac_key_hmac_sha256(key, &mut derived_mac_key);

    match algorithm {
        CipherAlgorithm::Aes256Ctr => {
            if let Some(validation) = try_mac_key_strategy(
                MacKeyStrategy::DirectCandidateKey,
                key.as_slice(),
                header,
                ciphertext,
                plaintext,
                expected_mac,
                original_blob,
            ) {
                return Some(validation);
            }
            if let Some(validation) = try_mac_key_strategy(
                MacKeyStrategy::CandidateTail16,
                &key[16..],
                header,
                ciphertext,
                plaintext,
                expected_mac,
                original_blob,
            ) {
                return Some(validation);
            }
            try_mac_key_strategy(
                MacKeyStrategy::DerivedHmacSha256,
                derived_mac_key.as_slice(),
                header,
                ciphertext,
                plaintext,
                expected_mac,
                original_blob,
            )
        }
        CipherAlgorithm::Aes128CtrSplit => {
            if let Some(validation) = try_mac_key_strategy(
                MacKeyStrategy::CandidateTail16,
                &key[16..],
                header,
                ciphertext,
                plaintext,
                expected_mac,
                original_blob,
            ) {
                return Some(validation);
            }
            try_mac_key_strategy(
                MacKeyStrategy::DerivedHmacSha256,
                derived_mac_key.as_slice(),
                header,
                ciphertext,
                plaintext,
                expected_mac,
                original_blob,
            )
        }
    }
}

fn validate_pinned_nonce_matrix_mac(
    key: &[u8; 32],
    original_blob: &[u8],
    plaintext: &[u8; CIPHERTEXT_BYTES],
    found: &AtomicBool,
) -> Option<MacValidation> {
    let expected_mac = &original_blob[HEADER_BYTES + CIPHERTEXT_BYTES..BLOB_BYTES];
    let header = &original_blob[..HEADER_BYTES];
    let ciphertext = &original_blob[HEADER_BYTES..HEADER_BYTES + CIPHERTEXT_BYTES];
    let local_part_md5 = Md5::digest(PINNED_MATRIX_LOCAL_PART);
    let local_part_sha256 = Sha256::digest(PINNED_MATRIX_LOCAL_PART);
    let mut transformed_mac_key = [0_u8; TAG_BYTES];

    for mac_key_transform in [
        MacKeyTransform::Identity,
        MacKeyTransform::Reversed,
        MacKeyTransform::SwapU32,
        MacKeyTransform::SwapU64,
    ] {
        if found.load(Ordering::Acquire) {
            return None;
        }

        transformed_mac_key.copy_from_slice(&key[16..]);
        apply_mac_key_transform(mac_key_transform, &mut transformed_mac_key);

        for &(context_injection, context) in &[
            (MacContextInjection::None, None),
            (
                MacContextInjection::LocalPart,
                Some(PINNED_MATRIX_LOCAL_PART),
            ),
            (
                MacContextInjection::LocalPartMd5,
                Some(local_part_md5.as_slice()),
            ),
            (
                MacContextInjection::LocalPartSha256,
                Some(local_part_sha256.as_slice()),
            ),
        ] {
            if let Some(validation) = try_pinned_matrix_padding_variant(
                &transformed_mac_key,
                mac_key_transform,
                context_injection,
                context,
                header,
                ciphertext,
                plaintext,
                expected_mac,
                original_blob,
                CIPHERTEXT_BYTES,
                MacPadding::None,
                found,
            ) {
                return Some(validation);
            }
        }
    }

    if !needs_padding_scan(plaintext) {
        return None;
    }

    if let Some(trimmed_len) = zero_trimmed_len(plaintext) {
        for mac_key_transform in [
            MacKeyTransform::Identity,
            MacKeyTransform::Reversed,
            MacKeyTransform::SwapU32,
            MacKeyTransform::SwapU64,
        ] {
            if found.load(Ordering::Acquire) {
                return None;
            }
            transformed_mac_key.copy_from_slice(&key[16..]);
            apply_mac_key_transform(mac_key_transform, &mut transformed_mac_key);
            for &(context_injection, context) in &[
                (MacContextInjection::None, None),
                (
                    MacContextInjection::LocalPart,
                    Some(PINNED_MATRIX_LOCAL_PART),
                ),
                (
                    MacContextInjection::LocalPartMd5,
                    Some(local_part_md5.as_slice()),
                ),
                (
                    MacContextInjection::LocalPartSha256,
                    Some(local_part_sha256.as_slice()),
                ),
            ] {
                if let Some(validation) = try_pinned_matrix_padding_variant(
                    &transformed_mac_key,
                    mac_key_transform,
                    context_injection,
                    context,
                    header,
                    ciphertext,
                    plaintext,
                    expected_mac,
                    original_blob,
                    trimmed_len,
                    MacPadding::ZeroTrimmed,
                    found,
                ) {
                    return Some(validation);
                }
            }
        }
    }

    if let Some(trimmed_len) = pkcs7_trimmed_len(plaintext) {
        for mac_key_transform in [
            MacKeyTransform::Identity,
            MacKeyTransform::Reversed,
            MacKeyTransform::SwapU32,
            MacKeyTransform::SwapU64,
        ] {
            if found.load(Ordering::Acquire) {
                return None;
            }
            transformed_mac_key.copy_from_slice(&key[16..]);
            apply_mac_key_transform(mac_key_transform, &mut transformed_mac_key);
            for &(context_injection, context) in &[
                (MacContextInjection::None, None),
                (
                    MacContextInjection::LocalPart,
                    Some(PINNED_MATRIX_LOCAL_PART),
                ),
                (
                    MacContextInjection::LocalPartMd5,
                    Some(local_part_md5.as_slice()),
                ),
                (
                    MacContextInjection::LocalPartSha256,
                    Some(local_part_sha256.as_slice()),
                ),
            ] {
                if let Some(validation) = try_pinned_matrix_padding_variant(
                    &transformed_mac_key,
                    mac_key_transform,
                    context_injection,
                    context,
                    header,
                    ciphertext,
                    plaintext,
                    expected_mac,
                    original_blob,
                    trimmed_len,
                    MacPadding::Pkcs7Trimmed,
                    found,
                ) {
                    return Some(validation);
                }
            }
        }
    }

    None
}

fn try_pinned_matrix_padding_variant(
    mac_key: &[u8],
    mac_key_transform: MacKeyTransform,
    context_injection: MacContextInjection,
    context: Option<&[u8]>,
    header: &[u8],
    ciphertext: &[u8],
    plaintext: &[u8; CIPHERTEXT_BYTES],
    expected_mac: &[u8],
    original_blob: &[u8],
    plaintext_len: usize,
    padding: MacPadding,
    found: &AtomicBool,
) -> Option<MacValidation> {
    if found.load(Ordering::Acquire) {
        return None;
    }
    if let Some(validation) = try_mac_scope(
        MacKeyStrategy::CandidateTail16,
        mac_key,
        header,
        ciphertext,
        plaintext,
        plaintext_len,
        MacScope::HeaderPlaintext,
        expected_mac,
        None,
        padding,
        mac_key_transform,
        context_injection,
        context,
        true,
    ) {
        return Some(validation);
    }

    for &nonce_length in &PINNED_MATRIX_NONCE_LENGTHS {
        for nonce_offset in 0..=PINNED_MATRIX_NONCE_MAX_OFFSET {
            if found.load(Ordering::Acquire) {
                return None;
            }
            let nonce_slice = &original_blob[nonce_offset..nonce_offset + nonce_length];
            if let Some(validation) = try_mac_scope(
                MacKeyStrategy::CandidateTail16,
                mac_key,
                header,
                ciphertext,
                plaintext,
                plaintext_len,
                MacScope::PlaintextNonce,
                expected_mac,
                Some((nonce_offset, nonce_length, nonce_slice)),
                padding,
                mac_key_transform,
                context_injection,
                context,
                true,
            ) {
                return Some(validation);
            }

            if let Some(validation) = try_mac_scope(
                MacKeyStrategy::CandidateTail16,
                mac_key,
                header,
                ciphertext,
                plaintext,
                plaintext_len,
                MacScope::HeaderPlaintextNonce,
                expected_mac,
                Some((nonce_offset, nonce_length, nonce_slice)),
                padding,
                mac_key_transform,
                context_injection,
                context,
                true,
            ) {
                return Some(validation);
            }
        }
    }

    None
}

fn try_mac_key_strategy(
    key_strategy: MacKeyStrategy,
    mac_key: &[u8],
    header: &[u8],
    ciphertext: &[u8],
    plaintext: &[u8; CIPHERTEXT_BYTES],
    expected_mac: &[u8],
    original_blob: &[u8],
) -> Option<MacValidation> {
    for &(scope, plaintext_len) in &[
        (MacScope::Ciphertext, CIPHERTEXT_BYTES),
        (MacScope::HeaderCiphertext, CIPHERTEXT_BYTES),
        (MacScope::Plaintext, CIPHERTEXT_BYTES),
        (MacScope::HeaderPlaintext, CIPHERTEXT_BYTES),
        (MacScope::HeaderPlaintextCiphertext, CIPHERTEXT_BYTES),
    ] {
        if let Some(validation) = try_mac_scope(
            key_strategy,
            mac_key,
            header,
            ciphertext,
            plaintext,
            plaintext_len,
            scope,
            expected_mac,
            None,
            MacPadding::None,
            MacKeyTransform::Identity,
            MacContextInjection::None,
            None,
            false,
        ) {
            return Some(validation);
        }
    }

    for &nonce_len in &[12_usize, CTR_NONCE_BYTES] {
        for nonce_offset in 0..=BLOB_BYTES - nonce_len {
            if let Some(validation) = try_mac_scope(
                key_strategy,
                mac_key,
                header,
                ciphertext,
                plaintext,
                CIPHERTEXT_BYTES,
                MacScope::HeaderPlaintextNonce,
                expected_mac,
                Some((
                    nonce_offset,
                    nonce_len,
                    &original_blob[nonce_offset..nonce_offset + nonce_len],
                )),
                MacPadding::None,
                MacKeyTransform::Identity,
                MacContextInjection::None,
                None,
                false,
            ) {
                return Some(validation);
            }
        }
    }

    if !needs_padding_scan(plaintext) {
        return None;
    }

    if let Some(trimmed_len) = zero_trimmed_len(plaintext) {
        if let Some(validation) = try_padding_variants(
            key_strategy,
            mac_key,
            header,
            ciphertext,
            plaintext,
            expected_mac,
            original_blob,
            trimmed_len,
            MacPadding::ZeroTrimmed,
        ) {
            return Some(validation);
        }
    }

    if let Some(trimmed_len) = pkcs7_trimmed_len(plaintext) {
        if let Some(validation) = try_padding_variants(
            key_strategy,
            mac_key,
            header,
            ciphertext,
            plaintext,
            expected_mac,
            original_blob,
            trimmed_len,
            MacPadding::Pkcs7Trimmed,
        ) {
            return Some(validation);
        }
    }

    None
}

fn try_padding_variants(
    key_strategy: MacKeyStrategy,
    mac_key: &[u8],
    header: &[u8],
    ciphertext: &[u8],
    plaintext: &[u8; CIPHERTEXT_BYTES],
    expected_mac: &[u8],
    original_blob: &[u8],
    trimmed_len: usize,
    padding: MacPadding,
) -> Option<MacValidation> {
    for &scope in &[
        MacScope::Plaintext,
        MacScope::HeaderPlaintext,
        MacScope::HeaderPlaintextCiphertext,
    ] {
        if let Some(validation) = try_mac_scope(
            key_strategy,
            mac_key,
            header,
            ciphertext,
            plaintext,
            trimmed_len,
            scope,
            expected_mac,
            None,
            padding,
            MacKeyTransform::Identity,
            MacContextInjection::None,
            None,
            false,
        ) {
            return Some(validation);
        }
    }

    for &nonce_len in &[12_usize, CTR_NONCE_BYTES] {
        for nonce_offset in 0..=BLOB_BYTES - nonce_len {
            if let Some(validation) = try_mac_scope(
                key_strategy,
                mac_key,
                header,
                ciphertext,
                plaintext,
                trimmed_len,
                MacScope::HeaderPlaintextNonce,
                expected_mac,
                Some((
                    nonce_offset,
                    nonce_len,
                    &original_blob[nonce_offset..nonce_offset + nonce_len],
                )),
                padding,
                MacKeyTransform::Identity,
                MacContextInjection::None,
                None,
                false,
            ) {
                return Some(validation);
            }
        }
    }

    None
}

fn try_mac_scope(
    key_strategy: MacKeyStrategy,
    mac_key: &[u8],
    header: &[u8],
    ciphertext: &[u8],
    plaintext: &[u8; CIPHERTEXT_BYTES],
    plaintext_len: usize,
    scope: MacScope,
    expected_mac: &[u8],
    nonce_slice: Option<(usize, usize, &[u8])>,
    padding: MacPadding,
    mac_key_transform: MacKeyTransform,
    context_injection: MacContextInjection,
    context: Option<&[u8]>,
    include_raw_hash: bool,
) -> Option<MacValidation> {
    let nonce_meta = nonce_slice.map(|(offset, len, _)| (offset as u8, len as u8));
    let nonce_bytes = nonce_slice.map(|(_, _, bytes)| bytes);

    if hmac_md5_matches_scope(
        mac_key,
        scope,
        header,
        ciphertext,
        plaintext,
        plaintext_len,
        nonce_bytes,
        context,
        expected_mac,
    ) {
        return Some(MacValidation {
            algorithm: MacAlgorithm::HmacMd5,
            key_strategy,
            scope,
            nonce_offset: nonce_meta.map(|(offset, _)| offset),
            nonce_length: nonce_meta.map(|(_, len)| len),
            padding,
            mac_key_transform,
            context_injection,
        });
    }

    if hmac_sha256_truncated_matches_scope(
        mac_key,
        scope,
        header,
        ciphertext,
        plaintext,
        plaintext_len,
        nonce_bytes,
        context,
        expected_mac,
    ) {
        return Some(MacValidation {
            algorithm: MacAlgorithm::HmacSha256Truncated16,
            key_strategy,
            scope,
            nonce_offset: nonce_meta.map(|(offset, _)| offset),
            nonce_length: nonce_meta.map(|(_, len)| len),
            padding,
            mac_key_transform,
            context_injection,
        });
    }

    if include_raw_hash {
        if raw_md5_matches_scope(
            mac_key,
            scope,
            header,
            ciphertext,
            plaintext,
            plaintext_len,
            nonce_bytes,
            context,
            expected_mac,
        ) {
            return Some(MacValidation {
                algorithm: MacAlgorithm::RawMd5,
                key_strategy,
                scope,
                nonce_offset: nonce_meta.map(|(offset, _)| offset),
                nonce_length: nonce_meta.map(|(_, len)| len),
                padding,
                mac_key_transform,
                context_injection,
            });
        }

        if raw_sha256_truncated_matches_scope(
            mac_key,
            scope,
            header,
            ciphertext,
            plaintext,
            plaintext_len,
            nonce_bytes,
            context,
            expected_mac,
        ) {
            return Some(MacValidation {
                algorithm: MacAlgorithm::RawSha256Truncated16,
                key_strategy,
                scope,
                nonce_offset: nonce_meta.map(|(offset, _)| offset),
                nonce_length: nonce_meta.map(|(_, len)| len),
                padding,
                mac_key_transform,
                context_injection,
            });
        }
    }

    None
}

fn derive_mac_key_hmac_sha256(candidate_key: &[u8; 32], out: &mut [u8; 32]) {
    let mut mac = Hmac::<Sha256>::new_from_slice(candidate_key).expect("valid hmac-sha256 key");
    mac.update(b"pmta-mac-key");
    let derived = mac.finalize().into_bytes();
    out.copy_from_slice(&derived);
}

#[inline(always)]
fn apply_mac_key_transform(transform: MacKeyTransform, key: &mut [u8; TAG_BYTES]) {
    match transform {
        MacKeyTransform::Identity => {}
        MacKeyTransform::Reversed => key.reverse(),
        MacKeyTransform::SwapU32 => swap_mac_key_endianness(key, 4),
        MacKeyTransform::SwapU64 => swap_mac_key_endianness(key, 8),
    }
}

#[inline(always)]
fn swap_mac_key_endianness(key: &mut [u8; TAG_BYTES], chunk_len: usize) {
    for chunk in key.chunks_exact_mut(chunk_len) {
        chunk.reverse();
    }
}

fn hmac_md5_matches_scope(
    key: &[u8],
    scope: MacScope,
    header: &[u8],
    ciphertext: &[u8],
    plaintext: &[u8; CIPHERTEXT_BYTES],
    plaintext_len: usize,
    nonce_slice: Option<&[u8]>,
    context: Option<&[u8]>,
    expected_mac: &[u8],
) -> bool {
    let mut mac = Hmac::<Md5>::new_from_slice(key).expect("valid hmac-md5 key");
    update_hmac_scope(
        &mut mac,
        scope,
        header,
        ciphertext,
        plaintext,
        plaintext_len,
        nonce_slice,
        context,
    );
    let calculated = mac.finalize().into_bytes();
    calculated.as_slice() == expected_mac
}

fn hmac_sha256_truncated_matches_scope(
    key: &[u8],
    scope: MacScope,
    header: &[u8],
    ciphertext: &[u8],
    plaintext: &[u8; CIPHERTEXT_BYTES],
    plaintext_len: usize,
    nonce_slice: Option<&[u8]>,
    context: Option<&[u8]>,
    expected_mac: &[u8],
) -> bool {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("valid hmac-sha256 key");
    update_hmac_scope(
        &mut mac,
        scope,
        header,
        ciphertext,
        plaintext,
        plaintext_len,
        nonce_slice,
        context,
    );
    let calculated = mac.finalize().into_bytes();
    &calculated[..TAG_BYTES] == expected_mac
}

fn raw_md5_matches_scope(
    key: &[u8],
    scope: MacScope,
    header: &[u8],
    ciphertext: &[u8],
    plaintext: &[u8; CIPHERTEXT_BYTES],
    plaintext_len: usize,
    nonce_slice: Option<&[u8]>,
    context: Option<&[u8]>,
    expected_mac: &[u8],
) -> bool {
    let mut hash = Md5::new();
    hash.update(key);
    if let Some(context) = context {
        hash.update(context);
    }
    update_md5_scope(
        &mut hash,
        scope,
        header,
        ciphertext,
        plaintext,
        plaintext_len,
        nonce_slice,
    );
    let calculated = hash.finalize();
    calculated.as_slice() == expected_mac
}

fn raw_sha256_truncated_matches_scope(
    key: &[u8],
    scope: MacScope,
    header: &[u8],
    ciphertext: &[u8],
    plaintext: &[u8; CIPHERTEXT_BYTES],
    plaintext_len: usize,
    nonce_slice: Option<&[u8]>,
    context: Option<&[u8]>,
    expected_mac: &[u8],
) -> bool {
    let mut hash = Sha256::new();
    hash.update(key);
    if let Some(context) = context {
        hash.update(context);
    }
    update_sha256_scope(
        &mut hash,
        scope,
        header,
        ciphertext,
        plaintext,
        plaintext_len,
        nonce_slice,
    );
    let calculated = hash.finalize();
    &calculated[..TAG_BYTES] == expected_mac
}

fn update_hmac_scope<M: Mac>(
    mac: &mut M,
    scope: MacScope,
    header: &[u8],
    ciphertext: &[u8],
    plaintext: &[u8; CIPHERTEXT_BYTES],
    plaintext_len: usize,
    nonce_slice: Option<&[u8]>,
    context: Option<&[u8]>,
) {
    if let Some(context) = context {
        mac.update(context);
    }
    match scope {
        MacScope::Ciphertext => mac.update(ciphertext),
        MacScope::HeaderCiphertext => {
            mac.update(header);
            mac.update(ciphertext);
        }
        MacScope::Plaintext => mac.update(&plaintext[..plaintext_len]),
        MacScope::PlaintextNonce => {
            mac.update(&plaintext[..plaintext_len]);
            mac.update(nonce_slice.expect("nonce slice required for nonce scope"));
        }
        MacScope::HeaderPlaintext => {
            mac.update(header);
            mac.update(&plaintext[..plaintext_len]);
        }
        MacScope::HeaderPlaintextCiphertext => {
            mac.update(header);
            mac.update(&plaintext[..plaintext_len]);
            mac.update(ciphertext);
        }
        MacScope::HeaderPlaintextNonce => {
            mac.update(header);
            mac.update(&plaintext[..plaintext_len]);
            mac.update(nonce_slice.expect("nonce slice required for nonce scope"));
        }
    }
}

fn update_md5_scope(
    hash: &mut Md5,
    scope: MacScope,
    header: &[u8],
    ciphertext: &[u8],
    plaintext: &[u8; CIPHERTEXT_BYTES],
    plaintext_len: usize,
    nonce_slice: Option<&[u8]>,
) {
    match scope {
        MacScope::Ciphertext => hash.update(ciphertext),
        MacScope::HeaderCiphertext => {
            hash.update(header);
            hash.update(ciphertext);
        }
        MacScope::Plaintext => hash.update(&plaintext[..plaintext_len]),
        MacScope::PlaintextNonce => {
            hash.update(&plaintext[..plaintext_len]);
            hash.update(nonce_slice.expect("nonce slice required for nonce scope"));
        }
        MacScope::HeaderPlaintext => {
            hash.update(header);
            hash.update(&plaintext[..plaintext_len]);
        }
        MacScope::HeaderPlaintextCiphertext => {
            hash.update(header);
            hash.update(&plaintext[..plaintext_len]);
            hash.update(ciphertext);
        }
        MacScope::HeaderPlaintextNonce => {
            hash.update(header);
            hash.update(&plaintext[..plaintext_len]);
            hash.update(nonce_slice.expect("nonce slice required for nonce scope"));
        }
    }
}

fn update_sha256_scope(
    hash: &mut Sha256,
    scope: MacScope,
    header: &[u8],
    ciphertext: &[u8],
    plaintext: &[u8; CIPHERTEXT_BYTES],
    plaintext_len: usize,
    nonce_slice: Option<&[u8]>,
) {
    match scope {
        MacScope::Ciphertext => hash.update(ciphertext),
        MacScope::HeaderCiphertext => {
            hash.update(header);
            hash.update(ciphertext);
        }
        MacScope::Plaintext => hash.update(&plaintext[..plaintext_len]),
        MacScope::PlaintextNonce => {
            hash.update(&plaintext[..plaintext_len]);
            hash.update(nonce_slice.expect("nonce slice required for nonce scope"));
        }
        MacScope::HeaderPlaintext => {
            hash.update(header);
            hash.update(&plaintext[..plaintext_len]);
        }
        MacScope::HeaderPlaintextCiphertext => {
            hash.update(header);
            hash.update(&plaintext[..plaintext_len]);
            hash.update(ciphertext);
        }
        MacScope::HeaderPlaintextNonce => {
            hash.update(header);
            hash.update(&plaintext[..plaintext_len]);
            hash.update(nonce_slice.expect("nonce slice required for nonce scope"));
        }
    }
}

fn needs_padding_scan(plaintext: &[u8; CIPHERTEXT_BYTES]) -> bool {
    plaintext.starts_with(b"t=") && !matches!(plaintext[CIPHERTEXT_BYTES - 1], b'&' | b'}')
}

fn zero_trimmed_len(plaintext: &[u8; CIPHERTEXT_BYTES]) -> Option<usize> {
    let trimmed_len = plaintext.iter().rposition(|&byte| byte != 0)? + 1;
    (trimmed_len < CIPHERTEXT_BYTES).then_some(trimmed_len)
}

fn pkcs7_trimmed_len(plaintext: &[u8; CIPHERTEXT_BYTES]) -> Option<usize> {
    let pad_len = *plaintext.last()? as usize;
    if pad_len == 0 || pad_len > CTR_NONCE_BYTES || pad_len > CIPHERTEXT_BYTES {
        return None;
    }
    let pad_start = CIPHERTEXT_BYTES - pad_len;
    if plaintext[pad_start..]
        .iter()
        .all(|&byte| byte as usize == pad_len)
    {
        Some(pad_start)
    } else {
        None
    }
}

fn print_validated_hit(finding: &BruteForceFinding) {
    let validation = match finding.mac_validation {
        Some(validation) => validation,
        None => return,
    };

    let plaintext = std::str::from_utf8(&finding.decoded_bytes[..finding.decoded_len as usize])
        .unwrap_or("<non-utf8>");

    println!("\x1b[1;92mOPPS!\x1b[0m");
    println!(
        "validated blob={} key={} algorithm={:?} key_strategy={:?} scope={:?} mac_key_transform={:?} context={:?} blob_transform={:?} key_transform={:?} nonce@{}+{} plaintext={}",
        finding.blob_index + 1,
        finding.key_index + 1,
        validation.algorithm,
        validation.key_strategy,
        validation.scope,
        validation.mac_key_transform,
        validation.context_injection,
        finding.blob_transform,
        finding.key_transform,
        finding.nonce_offset,
        finding.nonce_length,
        plaintext
    );
    if let Some(nonce_offset) = validation.nonce_offset {
        println!(
            "mac_nonce@{}+{} padding={:?}",
            nonce_offset,
            validation.nonce_length.unwrap_or_default(),
            validation.padding
        );
    }
}

fn write_escaped_preview(bytes: &[u8], output: &mut [u8; PREVIEW_BYTES]) -> usize {
    let mut written = 0_usize;

    for &byte in bytes {
        if is_printable_ascii(byte) {
            if written >= output.len() {
                break;
            }
            output[written] = byte;
            written += 1;
            continue;
        }

        if written + 4 > output.len() {
            break;
        }

        output[written] = b'\\';
        output[written + 1] = b'x';
        output[written + 2] = hex_nibble(byte >> 4);
        output[written + 3] = hex_nibble(byte & 0x0f);
        written += 4;
    }

    written
}

fn hex_nibble(value: u8) -> u8 {
    match value {
        0..=9 => b'0' + value,
        _ => b'a' + (value - 10),
    }
}

fn is_printable_ascii(byte: u8) -> bool {
    (0x20..=0x7e).contains(&byte)
}

fn try_aes256_ctr(key: &[u8; 32], iv: &[u8], ciphertext: &[u8]) -> Option<[u8; CIPHERTEXT_BYTES]> {
    let mut buffer = [0_u8; CIPHERTEXT_BYTES];
    buffer.copy_from_slice(ciphertext);
    let mut cipher = Ctr128BE::<Aes256>::new_from_slices(key, iv).ok()?;
    cipher.apply_keystream(&mut buffer);
    Some(buffer)
}

fn try_aes128_ctr_split(
    key: &[u8; 32],
    iv: &[u8],
    ciphertext: &[u8],
) -> Option<[u8; CIPHERTEXT_BYTES]> {
    let mut buffer = [0_u8; CIPHERTEXT_BYTES];
    buffer.copy_from_slice(ciphertext);
    let mut cipher = Ctr128BE::<Aes128>::new_from_slices(&key[..16], iv).ok()?;
    cipher.apply_keystream(&mut buffer);
    Some(buffer)
}

#[cfg(test)]
mod tests {
    use super::{
        entropy_check, evaluate_candidate, normalize_candidate_keys, scan_blob, BlobTransform,
        BruteForceInputError, CipherAlgorithm, FindingKind, FindingPriority, KeyTransform,
        MacAlgorithm, MacContextInjection, MacKeyStrategy, MacKeyTransform, MacPadding, MacScope,
        BINARY_ENTROPY_THRESHOLD_MILLI, BLOB_BYTES, CIPHERTEXT_BYTES, CTR_NONCE_BYTES,
        HEADER_BYTES, HEURISTIC_SCORE_THRESHOLD,
    };
    use aes::cipher::{KeyIvInit, StreamCipher};
    use aes::{Aes128, Aes256};
    use ctr::Ctr128BE;
    use hmac::{Hmac, Mac};
    use md5::Digest as _;
    use md5::Md5;
    use sha2::Sha256;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn rejects_invalid_key_lengths() {
        let error = normalize_candidate_keys([vec![1_u8; 31]]).expect_err("should fail");
        assert!(matches!(
            error,
            BruteForceInputError::InvalidKeyLength {
                index: 0,
                length: 31
            }
        ));
    }

    #[test]
    fn exposes_ctr_nonce_size() {
        assert_eq!(CTR_NONCE_BYTES, 16);
    }

    #[test]
    fn finds_ctr_exact_match_on_strict_ciphertext_segment() {
        let header = header_bytes();
        let key = [7_u8; 32];
        let plaintext = *b"t=1740982512&u=9876&c=042&z=1234";
        let tag = [0xA5_u8; 16];
        let blob = seal_aes_ctr(&header, &key, &plaintext, 8, &tag);
        let found = AtomicBool::new(false);

        let hit = scan_blob(0, &blob, &[key], &found).expect("match");
        assert_eq!(hit.algorithm, CipherAlgorithm::Aes256Ctr);
        assert_eq!(hit.kind, FindingKind::ExactUtf8);
        assert_eq!(hit.priority, FindingPriority::High);
        assert_eq!(hit.decoded_len as usize, CIPHERTEXT_BYTES);
        assert_eq!(hit.decoded_bytes, plaintext);
        assert!(hit.heuristic.score >= HEURISTIC_SCORE_THRESHOLD);
        assert!(hit.heuristic.t_eq_hits > 0);
    }

    #[test]
    fn reversed_key_variant_is_tested() {
        let header = header_bytes();
        let mut key = [0_u8; 32];
        for (index, byte) in key.iter_mut().enumerate() {
            *byte = index as u8;
        }
        let plaintext = *b"t=1740982512&u=9876&c=042&z=1234";
        let tag = [0x3C_u8; 16];
        let blob = seal_aes_ctr(&header, &key, &plaintext, 4, &tag);
        let found = AtomicBool::new(false);
        let mut reversed = key;
        reversed.reverse();

        let hit = scan_blob(0, &blob, &[reversed], &found).expect("match");
        assert_eq!(hit.key_transform, KeyTransform::Reversed);
    }

    #[test]
    fn query_scoring_rewards_compact_kv_payloads() {
        let header = header_bytes();
        let key = [19_u8; 32];
        let plaintext = *b"t=1740982512&u=8821&c=017&id=901";
        let tag = [0x55_u8; 16];
        let blob = seal_aes_ctr(&header, &key, &plaintext, 0, &tag);
        let found = AtomicBool::new(false);

        let hit = scan_blob(0, &blob, &[key], &found).expect("match");
        assert!(hit.heuristic.equals_count >= 4);
        assert!(hit.heuristic.ampersand_count >= 3);
        assert!(hit.heuristic.longest_digit_run >= 4);
        assert!(hit.heuristic.score >= HEURISTIC_SCORE_THRESHOLD);
    }

    #[test]
    fn dense_prefix_controls_are_rejected() {
        let plaintext = [
            0, 1, 2, 3, 4, 5, 6, 7, b't', b'=', b'1', b'7', b'0', b'4', b'0', b'6', b'7', b'2',
            b'0', b'0', b'&', b'u', b'=', b'9', b'8', b'7', b'6', b'&', b'c', b'=', b'4', b'2',
        ];
        assert!(evaluate_candidate(
            0,
            0,
            CipherAlgorithm::Aes256Ctr,
            BlobTransform::Identity,
            KeyTransform::Identity,
            0,
            &plaintext,
        )
        .is_none());
    }

    #[test]
    fn low_entropy_binary_candidate_is_reported() {
        let header = header_bytes();
        let key = [23_u8; 32];
        let plaintext = *b"AAAAAAAABBBBBBBBCCCCCCCCDDDDDDDD";
        let tag = [0x11_u8; 16];
        let blob = seal_aes_ctr(&header, &key, &plaintext, 6, &tag);
        let found = AtomicBool::new(false);

        let hit = scan_blob(0, &blob, &[key], &found).expect("match");
        assert!(hit.heuristic.entropy_milli_bits < BINARY_ENTROPY_THRESHOLD_MILLI);
    }

    #[test]
    fn validates_hmac_md5_over_plaintext() {
        let header = header_bytes();
        let key = [31_u8; 32];
        let plaintext = *b"t=1740982512&u=9876&c=042&id=100";
        let blob = seal_aes_ctr_hmac_md5(
            &header,
            &key,
            &plaintext,
            6,
            MacScope::Plaintext,
            MacKeyStrategy::DirectCandidateKey,
        );
        let found = AtomicBool::new(false);

        let hit = scan_blob(0, &blob, &[key], &found).expect("validated match");
        assert_eq!(
            hit.mac_validation,
            Some(super::MacValidation {
                algorithm: MacAlgorithm::HmacMd5,
                key_strategy: MacKeyStrategy::DirectCandidateKey,
                scope: MacScope::Plaintext,
                nonce_offset: None,
                nonce_length: None,
                padding: MacPadding::None,
                mac_key_transform: MacKeyTransform::Identity,
                context_injection: MacContextInjection::None,
            })
        );
    }

    #[test]
    fn validates_hmac_sha256_truncated_over_ciphertext() {
        let header = header_bytes();
        let key = [37_u8; 32];
        let plaintext = *b"t=1740982512&u=4321&c=077&q=abcd";
        let blob = seal_aes_ctr_hmac_sha256(
            &header,
            &key,
            &plaintext,
            4,
            MacScope::Ciphertext,
            MacKeyStrategy::DirectCandidateKey,
        );
        let found = AtomicBool::new(false);

        let hit = scan_blob(0, &blob, &[key], &found).expect("validated match");
        assert_eq!(
            hit.mac_validation,
            Some(super::MacValidation {
                algorithm: MacAlgorithm::HmacSha256Truncated16,
                key_strategy: MacKeyStrategy::DirectCandidateKey,
                scope: MacScope::Ciphertext,
                nonce_offset: None,
                nonce_length: None,
                padding: MacPadding::None,
                mac_key_transform: MacKeyTransform::Identity,
                context_injection: MacContextInjection::None,
            })
        );
    }

    #[test]
    fn validates_hmac_md5_over_header_plaintext_with_tail_key() {
        let header = header_bytes();
        let mut key = [0_u8; 32];
        for (index, byte) in key.iter_mut().enumerate() {
            *byte = (index as u8).wrapping_mul(7).wrapping_add(3);
        }
        let plaintext = *b"t=1740982512&u=9999&c=099&id=110";
        let blob = seal_aes_ctr_hmac_md5(
            &header,
            &key,
            &plaintext,
            2,
            MacScope::HeaderPlaintext,
            MacKeyStrategy::CandidateTail16,
        );
        let found = AtomicBool::new(false);

        let hit = scan_blob(0, &blob, &[key], &found).expect("validated match");
        assert_eq!(
            hit.mac_validation,
            Some(super::MacValidation {
                algorithm: MacAlgorithm::HmacMd5,
                key_strategy: MacKeyStrategy::CandidateTail16,
                scope: MacScope::HeaderPlaintext,
                nonce_offset: None,
                nonce_length: None,
                padding: MacPadding::None,
                mac_key_transform: MacKeyTransform::Identity,
                context_injection: MacContextInjection::None,
            })
        );
    }

    #[test]
    fn validates_hmac_sha256_over_header_plaintext_ciphertext_with_derived_key() {
        let header = header_bytes();
        let key = [41_u8; 32];
        let plaintext = *b"t=1740982512&u=1111&c=055&id=220";
        let blob = seal_aes_ctr_hmac_sha256(
            &header,
            &key,
            &plaintext,
            10,
            MacScope::HeaderPlaintextCiphertext,
            MacKeyStrategy::DerivedHmacSha256,
        );
        let found = AtomicBool::new(false);

        let hit = scan_blob(0, &blob, &[key], &found).expect("validated match");
        assert_eq!(
            hit.mac_validation,
            Some(super::MacValidation {
                algorithm: MacAlgorithm::HmacSha256Truncated16,
                key_strategy: MacKeyStrategy::DerivedHmacSha256,
                scope: MacScope::HeaderPlaintextCiphertext,
                nonce_offset: None,
                nonce_length: None,
                padding: MacPadding::None,
                mac_key_transform: MacKeyTransform::Identity,
                context_injection: MacContextInjection::None,
            })
        );
    }

    #[test]
    fn finds_aes128_split_key_with_nonce_scoped_mac() {
        let header = header_bytes();
        let mut key = [0_u8; 32];
        for (index, byte) in key.iter_mut().enumerate() {
            *byte = (index as u8).wrapping_mul(11).wrapping_add(5);
        }
        let plaintext = *b"t=1740982512&u=8821&c=017&id=901";
        let blob = seal_ctr_hmac_md5(
            CipherAlgorithm::Aes128CtrSplit,
            &header,
            &key,
            &plaintext,
            7,
            MacScope::HeaderPlaintextNonce,
            MacKeyStrategy::CandidateTail16,
            Some((18, 12)),
            CIPHERTEXT_BYTES,
        );
        let found = AtomicBool::new(false);

        let hit = scan_blob(0, &blob, &[key], &found).expect("validated split-key match");
        assert_eq!(hit.algorithm, CipherAlgorithm::Aes128CtrSplit);
        assert_eq!(
            hit.mac_validation,
            Some(super::MacValidation {
                algorithm: MacAlgorithm::HmacMd5,
                key_strategy: MacKeyStrategy::CandidateTail16,
                scope: MacScope::HeaderPlaintextNonce,
                nonce_offset: Some(18),
                nonce_length: Some(12),
                padding: MacPadding::None,
                mac_key_transform: MacKeyTransform::Identity,
                context_injection: MacContextInjection::None,
            })
        );
    }

    #[test]
    fn pinned_matrix_runner_finds_plaintext_nonce_match() {
        let header = header_bytes();
        let mut key = [0_u8; 32];
        for (index, byte) in key.iter_mut().enumerate() {
            *byte = (index as u8).wrapping_mul(9).wrapping_add(1);
        }
        let plaintext = *b"t=1740982512&u=7001&c=333&id=880";
        let blob = seal_ctr_hmac_sha256(
            CipherAlgorithm::Aes128CtrSplit,
            &header,
            &key,
            &plaintext,
            10,
            MacScope::PlaintextNonce,
            MacKeyStrategy::CandidateTail16,
            Some((24, 16)),
            CIPHERTEXT_BYTES,
        );
        let found = AtomicBool::new(false);
        let target = super::PinnedMatrixTarget {
            blob_index: 0,
            key_index: 42,
            decrypt_nonce_offset: 10,
            blob_transform: BlobTransform::Identity,
            key_transform: KeyTransform::Identity,
        };

        let hit =
            super::scan_pinned_matrix_target(target, &blob, &key, &found).expect("matrix match");
        assert_eq!(hit.algorithm, CipherAlgorithm::Aes128CtrSplit);
        assert_eq!(
            hit.mac_validation,
            Some(super::MacValidation {
                algorithm: MacAlgorithm::HmacSha256Truncated16,
                key_strategy: MacKeyStrategy::CandidateTail16,
                scope: MacScope::PlaintextNonce,
                nonce_offset: Some(24),
                nonce_length: Some(16),
                padding: MacPadding::None,
                mac_key_transform: MacKeyTransform::Identity,
                context_injection: MacContextInjection::None,
            })
        );
    }

    #[test]
    fn pinned_matrix_runner_finds_transformed_raw_hash_with_context() {
        let header = header_bytes();
        let mut key = [0_u8; 32];
        for (index, byte) in key.iter_mut().enumerate() {
            *byte = (index as u8).wrapping_mul(5).wrapping_add(9);
        }
        let plaintext = *b"t=1740982512&u=4444&c=123&id=901";
        let blob = seal_ctr_raw_sha256(
            CipherAlgorithm::Aes128CtrSplit,
            &header,
            &key,
            &plaintext,
            10,
            MacScope::HeaderPlaintextNonce,
            MacKeyTransform::SwapU32,
            MacContextInjection::LocalPartMd5,
            Some((32, 12)),
            CIPHERTEXT_BYTES,
        );
        let found = AtomicBool::new(false);
        let target = super::PinnedMatrixTarget {
            blob_index: 0,
            key_index: 42,
            decrypt_nonce_offset: 10,
            blob_transform: BlobTransform::Identity,
            key_transform: KeyTransform::Identity,
        };

        let hit =
            super::scan_pinned_matrix_target(target, &blob, &key, &found).expect("matrix match");
        assert_eq!(
            hit.mac_validation,
            Some(super::MacValidation {
                algorithm: MacAlgorithm::RawSha256Truncated16,
                key_strategy: MacKeyStrategy::CandidateTail16,
                scope: MacScope::HeaderPlaintextNonce,
                nonce_offset: Some(32),
                nonce_length: Some(12),
                padding: MacPadding::None,
                mac_key_transform: MacKeyTransform::SwapU32,
                context_injection: MacContextInjection::LocalPartMd5,
            })
        );
    }

    #[test]
    fn validates_zero_trimmed_padding_scope() {
        let header = header_bytes();
        let key = [53_u8; 32];
        let mut plaintext = [0_u8; CIPHERTEXT_BYTES];
        plaintext[..21].copy_from_slice(b"t=1740982512&c=042&id");
        let blob = seal_ctr_hmac_sha256(
            CipherAlgorithm::Aes256Ctr,
            &header,
            &key,
            &plaintext,
            5,
            MacScope::HeaderPlaintext,
            MacKeyStrategy::DirectCandidateKey,
            None,
            21,
        );
        let found = AtomicBool::new(false);

        let hit = scan_blob(0, &blob, &[key], &found).expect("validated padding match");
        assert_eq!(
            hit.mac_validation,
            Some(super::MacValidation {
                algorithm: MacAlgorithm::HmacSha256Truncated16,
                key_strategy: MacKeyStrategy::DirectCandidateKey,
                scope: MacScope::HeaderPlaintext,
                nonce_offset: None,
                nonce_length: None,
                padding: MacPadding::ZeroTrimmed,
                mac_key_transform: MacKeyTransform::Identity,
                context_injection: MacContextInjection::None,
            })
        );
    }

    #[test]
    fn entropy_check_marks_unique_tag_as_randomish() {
        let tag: Vec<u8> = (0_u8..16).collect();
        let report = entropy_check(&tag);
        assert!(report.looks_random);
        assert_eq!(report.unique_bytes, 16);
    }

    #[test]
    fn reverse_ciphertext_transform_keeps_segmentation() {
        let mut blob = [0_u8; BLOB_BYTES];
        for (index, byte) in blob.iter_mut().enumerate() {
            *byte = index as u8;
        }

        super::apply_blob_transform(BlobTransform::ReverseCiphertext, &mut blob);

        let mut expected_header = [0_u8; HEADER_BYTES];
        for (index, byte) in expected_header.iter_mut().enumerate() {
            *byte = index as u8;
        }
        assert_eq!(&blob[..HEADER_BYTES], &expected_header);
        assert_eq!(blob[HEADER_BYTES], 63);
        assert_eq!(blob[HEADER_BYTES + CIPHERTEXT_BYTES - 1], 32);
        assert_eq!(blob[HEADER_BYTES + CIPHERTEXT_BYTES], 64);
    }

    fn header_bytes() -> [u8; HEADER_BYTES] {
        let mut header = [0_u8; HEADER_BYTES];
        for (index, byte) in header.iter_mut().enumerate() {
            *byte = (index as u8).wrapping_mul(13).wrapping_add(17);
        }
        header
    }

    fn seal_aes_ctr(
        header: &[u8; HEADER_BYTES],
        key: &[u8; 32],
        plaintext: &[u8; CIPHERTEXT_BYTES],
        nonce_offset: usize,
        tag: &[u8; 16],
    ) -> Vec<u8> {
        let mut ciphertext = *plaintext;
        let nonce = &header[nonce_offset..nonce_offset + CTR_NONCE_BYTES];
        let mut cipher = Ctr128BE::<Aes256>::new_from_slices(key, nonce).expect("ctr");
        cipher.apply_keystream(&mut ciphertext);

        let mut blob = Vec::with_capacity(BLOB_BYTES);
        blob.extend_from_slice(header);
        blob.extend_from_slice(&ciphertext);
        blob.extend_from_slice(tag);
        blob
    }

    fn seal_aes_ctr_hmac_md5(
        header: &[u8; HEADER_BYTES],
        key: &[u8; 32],
        plaintext: &[u8; CIPHERTEXT_BYTES],
        nonce_offset: usize,
        scope: MacScope,
        key_strategy: MacKeyStrategy,
    ) -> Vec<u8> {
        seal_ctr_hmac_md5(
            CipherAlgorithm::Aes256Ctr,
            header,
            key,
            plaintext,
            nonce_offset,
            scope,
            key_strategy,
            None,
            CIPHERTEXT_BYTES,
        )
    }

    fn seal_aes_ctr_hmac_sha256(
        header: &[u8; HEADER_BYTES],
        key: &[u8; 32],
        plaintext: &[u8; CIPHERTEXT_BYTES],
        nonce_offset: usize,
        scope: MacScope,
        key_strategy: MacKeyStrategy,
    ) -> Vec<u8> {
        seal_ctr_hmac_sha256(
            CipherAlgorithm::Aes256Ctr,
            header,
            key,
            plaintext,
            nonce_offset,
            scope,
            key_strategy,
            None,
            CIPHERTEXT_BYTES,
        )
    }

    fn seal_ctr_hmac_md5(
        algorithm: CipherAlgorithm,
        header: &[u8; HEADER_BYTES],
        key: &[u8; 32],
        plaintext: &[u8; CIPHERTEXT_BYTES],
        nonce_offset: usize,
        scope: MacScope,
        key_strategy: MacKeyStrategy,
        mac_nonce: Option<(usize, usize)>,
        plaintext_len: usize,
    ) -> Vec<u8> {
        let mut ciphertext = *plaintext;
        let nonce = &header[nonce_offset..nonce_offset + CTR_NONCE_BYTES];
        match algorithm {
            CipherAlgorithm::Aes256Ctr => {
                let mut cipher = Ctr128BE::<Aes256>::new_from_slices(key, nonce).expect("ctr");
                cipher.apply_keystream(&mut ciphertext);
            }
            CipherAlgorithm::Aes128CtrSplit => {
                let mut cipher =
                    Ctr128BE::<Aes128>::new_from_slices(&key[..16], nonce).expect("ctr");
                cipher.apply_keystream(&mut ciphertext);
            }
        }

        let mut blob = Vec::with_capacity(BLOB_BYTES);
        blob.extend_from_slice(header);
        blob.extend_from_slice(&ciphertext);

        let mut derived_mac_key = [0_u8; 32];
        let mac_key = match key_strategy {
            MacKeyStrategy::DirectCandidateKey => key.as_slice(),
            MacKeyStrategy::CandidateTail16 => &key[16..],
            MacKeyStrategy::DerivedHmacSha256 => {
                super::derive_mac_key_hmac_sha256(key, &mut derived_mac_key);
                derived_mac_key.as_slice()
            }
        };
        let nonce_slice = mac_nonce.map(|(offset, len)| &blob[offset..offset + len]);
        let mut mac = Hmac::<Md5>::new_from_slice(mac_key).expect("hmac-md5");
        super::update_hmac_scope(
            &mut mac,
            scope,
            header,
            &ciphertext,
            plaintext,
            plaintext_len,
            nonce_slice,
            None,
        );
        let tag = mac.finalize().into_bytes();
        blob.extend_from_slice(tag.as_slice());
        blob
    }

    fn seal_ctr_hmac_sha256(
        algorithm: CipherAlgorithm,
        header: &[u8; HEADER_BYTES],
        key: &[u8; 32],
        plaintext: &[u8; CIPHERTEXT_BYTES],
        nonce_offset: usize,
        scope: MacScope,
        key_strategy: MacKeyStrategy,
        mac_nonce: Option<(usize, usize)>,
        plaintext_len: usize,
    ) -> Vec<u8> {
        let mut ciphertext = *plaintext;
        let nonce = &header[nonce_offset..nonce_offset + CTR_NONCE_BYTES];
        match algorithm {
            CipherAlgorithm::Aes256Ctr => {
                let mut cipher = Ctr128BE::<Aes256>::new_from_slices(key, nonce).expect("ctr");
                cipher.apply_keystream(&mut ciphertext);
            }
            CipherAlgorithm::Aes128CtrSplit => {
                let mut cipher =
                    Ctr128BE::<Aes128>::new_from_slices(&key[..16], nonce).expect("ctr");
                cipher.apply_keystream(&mut ciphertext);
            }
        }

        let mut blob = Vec::with_capacity(BLOB_BYTES);
        blob.extend_from_slice(header);
        blob.extend_from_slice(&ciphertext);

        let mut derived_mac_key = [0_u8; 32];
        let mac_key = match key_strategy {
            MacKeyStrategy::DirectCandidateKey => key.as_slice(),
            MacKeyStrategy::CandidateTail16 => &key[16..],
            MacKeyStrategy::DerivedHmacSha256 => {
                super::derive_mac_key_hmac_sha256(key, &mut derived_mac_key);
                derived_mac_key.as_slice()
            }
        };
        let nonce_slice = mac_nonce.map(|(offset, len)| &blob[offset..offset + len]);
        let mut mac = Hmac::<Sha256>::new_from_slice(mac_key).expect("hmac-sha256");
        super::update_hmac_scope(
            &mut mac,
            scope,
            header,
            &ciphertext,
            plaintext,
            plaintext_len,
            nonce_slice,
            None,
        );
        let tag = mac.finalize().into_bytes();
        blob.extend_from_slice(&tag[..super::TAG_BYTES]);
        blob
    }

    fn seal_ctr_raw_sha256(
        algorithm: CipherAlgorithm,
        header: &[u8; HEADER_BYTES],
        key: &[u8; 32],
        plaintext: &[u8; CIPHERTEXT_BYTES],
        nonce_offset: usize,
        scope: MacScope,
        mac_key_transform: MacKeyTransform,
        context_injection: MacContextInjection,
        mac_nonce: Option<(usize, usize)>,
        plaintext_len: usize,
    ) -> Vec<u8> {
        let mut ciphertext = *plaintext;
        let nonce = &header[nonce_offset..nonce_offset + CTR_NONCE_BYTES];
        match algorithm {
            CipherAlgorithm::Aes256Ctr => {
                let mut cipher = Ctr128BE::<Aes256>::new_from_slices(key, nonce).expect("ctr");
                cipher.apply_keystream(&mut ciphertext);
            }
            CipherAlgorithm::Aes128CtrSplit => {
                let mut cipher =
                    Ctr128BE::<Aes128>::new_from_slices(&key[..16], nonce).expect("ctr");
                cipher.apply_keystream(&mut ciphertext);
            }
        }

        let mut blob = Vec::with_capacity(BLOB_BYTES);
        blob.extend_from_slice(header);
        blob.extend_from_slice(&ciphertext);

        let mut mac_key = [0_u8; super::TAG_BYTES];
        mac_key.copy_from_slice(&key[16..]);
        super::apply_mac_key_transform(mac_key_transform, &mut mac_key);

        let context_md5 = Md5::digest(super::PINNED_MATRIX_LOCAL_PART);
        let context_sha256 = Sha256::digest(super::PINNED_MATRIX_LOCAL_PART);
        let context = match context_injection {
            MacContextInjection::None => None,
            MacContextInjection::LocalPart => Some(super::PINNED_MATRIX_LOCAL_PART),
            MacContextInjection::LocalPartMd5 => Some(context_md5.as_slice()),
            MacContextInjection::LocalPartSha256 => Some(context_sha256.as_slice()),
        };

        let nonce_slice = mac_nonce.map(|(offset, len)| &blob[offset..offset + len]);
        let mut hash = Sha256::new();
        hash.update(mac_key);
        if let Some(context) = context {
            hash.update(context);
        }
        super::update_sha256_scope(
            &mut hash,
            scope,
            header,
            &ciphertext,
            plaintext,
            plaintext_len,
            nonce_slice,
        );
        let tag = hash.finalize();
        blob.extend_from_slice(&tag[..super::TAG_BYTES]);
        blob
    }
}
