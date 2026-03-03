use mps::{
    parse_strict_key_record, Base64Alphabet, CorePreference, MpsScheduler, NativeTask,
    ParsedKeyRecord, TaskPriority,
};
use std::env;
use std::fs;
use std::sync::mpsc;
use std::time::Duration;

#[derive(Debug)]
struct LineOutcome {
    line_number: usize,
    raw: String,
    result: Result<ParsedKeyRecord, mps::ParseError>,
}

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| "key.txt".to_owned());
    let input = fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!("failed to read {path}: {error}");
    });

    let lines: Vec<(usize, String)> = input
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let trimmed = line.trim();
            (!trimmed.is_empty()).then(|| (index + 1, trimmed.to_owned()))
        })
        .collect();

    if lines.is_empty() {
        println!("{path} is empty; add one colon-separated key per line.");
        return;
    }

    let scheduler = MpsScheduler::new();
    let (tx, rx) = mpsc::channel();

    let tasks: Vec<NativeTask> = lines
        .into_iter()
        .map(|(line_number, raw)| {
            let tx = tx.clone();
            Box::new(move || {
                let result = parse_strict_key_record(&raw);
                let _ = tx.send(LineOutcome {
                    line_number,
                    raw,
                    result,
                });
            }) as NativeTask
        })
        .collect();
    drop(tx);

    let submitted = scheduler.submit_batch_native(TaskPriority::High, CorePreference::Auto, tasks);
    let idle = scheduler.wait_for_idle(Duration::from_secs(10));

    let mut outcomes: Vec<_> = rx.into_iter().collect();
    outcomes.sort_by_key(|outcome| outcome.line_number);

    println!("Input path: {path}");
    println!("Submitted tasks: {}", submitted.len());
    println!("Scheduler idle: {idle}");
    println!();

    for outcome in outcomes {
        print_outcome(outcome);
    }

    let metrics = scheduler.metrics();
    println!("Metrics:");
    println!(
        "  submitted={}, completed={}, failed={}",
        metrics.submitted, metrics.completed, metrics.failed
    );
    println!(
        "  queue_depth(perf={}, eff={}, shared={}, total={})",
        metrics.queue_depth.performance,
        metrics.queue_depth.efficient,
        metrics.queue_depth.shared,
        metrics.queue_depth.total
    );
}

fn print_outcome(outcome: LineOutcome) {
    println!("Line {}:", outcome.line_number);

    match outcome.result {
        Ok(record) => {
            println!(
                "  raw_len={} normalized_len={} kind={} expected_len_match={}",
                record.raw_length,
                record.normalized_length,
                record.kind,
                record.has_expected_length
            );
            let payload = &record.payload;
            println!(
                "  payload alphabet={} encoded_len={} normalized_len={} padded_len={} pad_added={}",
                alphabet_name(payload.alphabet),
                payload.encoded.len(),
                payload.normalized_encoded.len(),
                payload.padded_encoded.len(),
                payload.pad_added
            );
            println!("  payload_hex={}", hex_string(record.payload_bytes()));
            println!("  payload_text={}", escaped_text(record.payload_bytes()));
        }
        Err(error) => {
            println!("  parse_error={error:?}");
            println!("  raw={}", outcome.raw);
        }
    }

    println!();
}

fn alphabet_name(alphabet: Base64Alphabet) -> &'static str {
    match alphabet {
        Base64Alphabet::Standard => "standard",
        Base64Alphabet::UrlSafe => "url-safe",
    }
}

fn hex_string(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        use std::fmt::Write;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn escaped_text(bytes: &[u8]) -> String {
    let mut output = String::new();
    for &byte in bytes {
        match byte {
            b' '..=b'~' => output.push(byte as char),
            b'\n' => output.push_str("\\n"),
            b'\r' => output.push_str("\\r"),
            b'\t' => output.push_str("\\t"),
            _ => {
                use std::fmt::Write;
                let _ = write!(output, "\\x{byte:02x}");
            }
        }
    }
    output
}
