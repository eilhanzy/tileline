use base64::engine::general_purpose::{STANDARD, URL_SAFE};
use base64::Engine;

pub const EXPECTED_RECORD_LENGTHS: [usize; 3] = [121, 129, 133];
pub const KEY_PREFIX: &str = "lu:";
pub const KEY_KIND: &str = "pmta";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Base64Alphabet {
    Standard,
    UrlSafe,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedSegment {
    pub encoded: String,
    pub normalized_encoded: String,
    pub padded_encoded: String,
    pub alphabet: Base64Alphabet,
    pub pad_added: usize,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedKeyRecord {
    pub raw: String,
    pub raw_length: usize,
    pub normalized: String,
    pub normalized_length: usize,
    pub has_expected_length: bool,
    pub kind: String,
    pub payload: DecodedSegment,
}

impl ParsedKeyRecord {
    pub fn payload_bytes(&self) -> &[u8] {
        &self.payload.bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    EmptyInput,
    MissingPrefix,
    MissingSeparator,
    UnexpectedSeparatorCount(usize),
    UnexpectedKind(String),
    EmptySegment { index: usize },
    InvalidBase64Length { index: usize, length: usize },
    DecodeFailed { index: usize },
}

// Enforce the literal `lu:pmta:<payload>` envelope first so later stages can
// reason about payload shape without re-checking prefix semantics.
pub fn parse_strict_key_record(input: &str) -> Result<ParsedKeyRecord, ParseError> {
    let raw = input.trim();
    if raw.is_empty() {
        return Err(ParseError::EmptyInput);
    }

    let normalized = raw
        .strip_prefix(KEY_PREFIX)
        .ok_or(ParseError::MissingPrefix)?;
    let separator_count = normalized.matches(':').count();
    match separator_count {
        0 => return Err(ParseError::MissingSeparator),
        1 => {}
        count => return Err(ParseError::UnexpectedSeparatorCount(count)),
    }

    let (left, right) = normalized
        .split_once(':')
        .expect("separator count already checked");
    if left.is_empty() {
        return Err(ParseError::EmptySegment { index: 0 });
    }
    if right.is_empty() {
        return Err(ParseError::EmptySegment { index: 1 });
    }

    if left != KEY_KIND {
        return Err(ParseError::UnexpectedKind(left.to_owned()));
    }

    Ok(ParsedKeyRecord {
        raw: raw.to_owned(),
        raw_length: raw.len(),
        normalized: normalized.to_owned(),
        normalized_length: normalized.len(),
        has_expected_length: EXPECTED_RECORD_LENGTHS.contains(&normalized.len()),
        kind: left.to_owned(),
        payload: decode_segment(right, 1)?,
    })
}

// PMTA samples omit `=` padding and mix escaped standard / URL-safe alphabets.
// Normalize first, then force padding instead of failing on short tails.
fn decode_segment(segment: &str, index: usize) -> Result<DecodedSegment, ParseError> {
    let normalized_encoded = normalize_segment(segment);
    let remainder = normalized_encoded.len() % 4;
    if remainder == 1 {
        return Err(ParseError::InvalidBase64Length {
            index,
            length: normalized_encoded.len(),
        });
    }

    let pad_added = (4 - remainder) % 4;
    let mut padded_encoded = String::with_capacity(normalized_encoded.len() + pad_added);
    padded_encoded.push_str(&normalized_encoded);
    for _ in 0..pad_added {
        padded_encoded.push('=');
    }

    let (alphabet, bytes) = match STANDARD.decode(&padded_encoded) {
        Ok(bytes) => (Base64Alphabet::Standard, bytes),
        Err(_) => match URL_SAFE.decode(&padded_encoded) {
            Ok(bytes) => (Base64Alphabet::UrlSafe, bytes),
            Err(_) => return Err(ParseError::DecodeFailed { index }),
        },
    };

    Ok(DecodedSegment {
        encoded: segment.to_owned(),
        normalized_encoded,
        padded_encoded,
        alphabet,
        pad_added,
        bytes,
    })
}

// The dataset encodes `/`, `+`, and `=` through placeholder tokens inside the
// raw record. Convert them back before base64 decoding.
fn normalize_segment(segment: &str) -> String {
    segment
        .replace("_s_l_", "/")
        .replace("_p_l_", "+")
        .replace("_e_q_", "=")
}

#[cfg(test)]
mod tests {
    use super::{decode_segment, parse_strict_key_record, Base64Alphabet, ParseError, KEY_KIND};
    use base64::engine::general_purpose::{STANDARD, URL_SAFE};
    use base64::Engine;

    #[test]
    fn standard_base64_padding_is_completed() {
        let encoded = STANDARD.encode(b"tileline");
        let unpadded = encoded.trim_end_matches('=');

        let decoded = decode_segment(unpadded, 0).expect("standard base64 should decode");
        assert_eq!(decoded.alphabet, Base64Alphabet::Standard);
        assert_eq!(decoded.bytes, b"tileline");
        assert_eq!(decoded.pad_added, encoded.len() - unpadded.len());
    }

    #[test]
    fn url_safe_base64_is_supported() {
        let encoded = URL_SAFE.encode([251_u8, 255_u8, 254_u8, 0_u8]);
        let unpadded = encoded.trim_end_matches('=');

        let decoded = decode_segment(unpadded, 0).expect("url-safe base64 should decode");
        assert_eq!(decoded.alphabet, Base64Alphabet::UrlSafe);
        assert_eq!(decoded.bytes, vec![251_u8, 255_u8, 254_u8, 0_u8]);
    }

    #[test]
    fn escaped_base64_symbols_are_normalized() {
        let decoded = decode_segment("_p_l__s_l_8_e_q_", 0).expect("escaped symbols should decode");
        assert_eq!(decoded.normalized_encoded, "+/8=");
        assert_eq!(decoded.bytes, vec![251_u8, 255_u8]);
    }

    #[test]
    fn strict_record_marks_expected_length() {
        let right = STANDARD.encode(vec![b'B'; 45]);
        let record = format!("lu:{KEY_KIND}:{right}");

        let parsed = parse_strict_key_record(&record).expect("record should parse");
        assert_eq!(parsed.raw_length, 68);
        assert_eq!(parsed.normalized_length, 65);
        assert!(!parsed.has_expected_length);
        assert_eq!(parsed.kind, KEY_KIND);
        assert_eq!(parsed.payload.bytes, vec![b'B'; 45]);
    }

    #[test]
    fn multiple_separators_are_rejected() {
        let error = parse_strict_key_record("lu:abcd:efgh:ijkl").expect_err("should fail");
        assert_eq!(error, ParseError::UnexpectedSeparatorCount(2));
    }

    #[test]
    fn missing_prefix_is_rejected() {
        let error = parse_strict_key_record("pmta:abcd").expect_err("should fail");
        assert_eq!(error, ParseError::MissingPrefix);
    }

    #[test]
    fn unexpected_kind_is_rejected() {
        let error = parse_strict_key_record("lu:noop:YWJj").expect_err("should fail");
        assert_eq!(error, ParseError::UnexpectedKind("noop".to_owned()));
    }
}
