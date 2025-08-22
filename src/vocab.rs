//! Vocabulary loading utilities for tiktoken encodings

use crate::core::Rank;
use crate::errors::{Result, TiktokenError};
use base64::Engine;
use std::collections::HashMap;

/// Vocabulary file information
#[derive(Debug, Clone)]
pub struct VocabInfo {
    pub url: &'static str,
    pub expected_hash: &'static str,
}

/// Registry of vocabulary files for different encodings
pub fn get_vocab_info(encoding: &str) -> Option<VocabInfo> {
    match encoding {
        "r50k_base" => Some(VocabInfo {
            url: "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
            expected_hash: "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
        }),
        "p50k_base" => Some(VocabInfo {
            url: "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
            expected_hash: "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069",
        }),
        "cl100k_base" => Some(VocabInfo {
            url: "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
            expected_hash: "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
        }),
        "o200k_base" => Some(VocabInfo {
            url: "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
            expected_hash: "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
        }),
        _ => None,
    }
}

/// Load tiktoken BPE vocabulary from a URL or embedded data
#[cfg(feature = "download")]
pub fn load_tiktoken_bpe(encoding: &str) -> Result<HashMap<Vec<u8>, Rank>> {
    use sha2::{Digest, Sha256};

    let vocab_info = get_vocab_info(encoding)
        .ok_or_else(|| TiktokenError::UnknownEncoding(encoding.to_string()))?;

    // Try to download the vocabulary file
    let response = reqwest::blocking::get(vocab_info.url)
        .map_err(|e| TiktokenError::DataError(format!("Failed to download vocabulary: {e}")))?;

    let content = response
        .text()
        .map_err(|e| TiktokenError::DataError(format!("Failed to read vocabulary content: {e}")))?;

    // Verify the hash
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let hash = format!("{:x}", hasher.finalize());

    if hash != vocab_info.expected_hash {
        return Err(TiktokenError::DataError(format!(
            "Vocabulary hash mismatch. Expected: {}, Got: {}",
            vocab_info.expected_hash, hash
        )));
    }

    parse_tiktoken_bpe(&content)
}

/// Load tiktoken BPE vocabulary without download feature (uses embedded fallback)
#[cfg(not(feature = "download"))]
pub fn load_tiktoken_bpe(encoding: &str) -> Result<HashMap<Vec<u8>, Rank>> {
    // For now, fall back to the basic vocabulary when download is disabled
    // In a production implementation, you would embed the actual vocabulary files
    create_basic_vocabulary()
}

/// Parse tiktoken BPE format
pub fn parse_tiktoken_bpe(content: &str) -> Result<HashMap<Vec<u8>, Rank>> {
    let mut ranks = HashMap::new();

    for (rank, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }

        // tiktoken format is base64 encoded bytes followed by a space and rank
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 2 {
            return Err(TiktokenError::DataError(format!(
                "Invalid tiktoken format at line {}: {}",
                rank + 1,
                line
            )));
        }

        let token_bytes =
            base64::engine::general_purpose::STANDARD.decode(parts[0]).map_err(|e| {
                TiktokenError::DataError(format!("Invalid base64 in tiktoken file: {e}"))
            })?;

        let token_rank: Rank = parts[1]
            .parse()
            .map_err(|e| TiktokenError::DataError(format!("Invalid rank in tiktoken file: {e}")))?;

        ranks.insert(token_bytes, token_rank);
    }

    Ok(ranks)
}

/// Create a basic vocabulary for demonstration purposes
/// This creates a minimal BPE vocabulary that can handle basic ASCII text
pub fn create_basic_vocabulary() -> Result<HashMap<Vec<u8>, Rank>> {
    let mut ranks = HashMap::new();
    let mut rank = 0;

    // Add single byte tokens (0-255)
    for i in 0..=255u8 {
        ranks.insert(vec![i], rank);
        rank += 1;
    }

    // Add some common two-byte sequences
    let common_pairs = [
        b"th".to_vec(),
        b"he".to_vec(),
        b"in".to_vec(),
        b"er".to_vec(),
        b"an".to_vec(),
        b"re".to_vec(),
        b"ed".to_vec(),
        b"nd".to_vec(),
        b"on".to_vec(),
        b"en".to_vec(),
        b"at".to_vec(),
        b"ou".to_vec(),
        b"it".to_vec(),
        b"is".to_vec(),
        b"or".to_vec(),
        b"ti".to_vec(),
        b"as".to_vec(),
        b"te".to_vec(),
        b"et".to_vec(),
        b"ng".to_vec(),
        b"of".to_vec(),
        b"al".to_vec(),
        b"de".to_vec(),
        b"se".to_vec(),
        b"le".to_vec(),
        b"to".to_vec(),
        b"nt".to_vec(),
        b"ha".to_vec(),
        b"ar".to_vec(),
        b"his".to_vec(),
        b"for".to_vec(),
        b"are".to_vec(),
        b"with".to_vec(),
        b"that".to_vec(),
        b"you".to_vec(),
        b"this".to_vec(),
        b"but".to_vec(),
        b"his".to_vec(),
        b"from".to_vec(),
        b"they".to_vec(),
        b"she".to_vec(),
        b"her".to_vec(),
        b"been".to_vec(),
        b"than".to_vec(),
        b"its".to_vec(),
        b"who".to_vec(),
        b"oil".to_vec(),
        b"sit".to_vec(),
        b" the".to_vec(),
        b" and".to_vec(),
        b" to".to_vec(),
        b" of".to_vec(),
        b" a".to_vec(),
        b" in".to_vec(),
        b" is".to_vec(),
        b" it".to_vec(),
        b" you".to_vec(),
        b" that".to_vec(),
        b" he".to_vec(),
        b" was".to_vec(),
        b" for".to_vec(),
        b" are".to_vec(),
        b" with".to_vec(),
        b" as".to_vec(),
        b" I".to_vec(),
        b" his".to_vec(),
        b" they".to_vec(),
        b" be".to_vec(),
        b" at".to_vec(),
        b" one".to_vec(),
        b" have".to_vec(),
        b" this".to_vec(),
        b" from".to_vec(),
        b" or".to_vec(),
        b" had".to_vec(),
        b" by".to_vec(),
        b" hot".to_vec(),
        b" word".to_vec(),
        b" but".to_vec(),
        b" what".to_vec(),
        b" some".to_vec(),
        b" we".to_vec(),
        b" can".to_vec(),
        b" out".to_vec(),
        b" other".to_vec(),
        b" were".to_vec(),
        b" all".to_vec(),
        b" there".to_vec(),
        b" when".to_vec(),
        b" up".to_vec(),
        b" use".to_vec(),
        b" your".to_vec(),
        b" how".to_vec(),
        b" said".to_vec(),
        b" an".to_vec(),
        b" each".to_vec(),
        b" which".to_vec(),
        b" she".to_vec(),
        b" do".to_vec(),
        b" has".to_vec(),
        b" will".to_vec(),
        b" if".to_vec(),
        b" about".to_vec(),
        b" get".to_vec(),
        b" go".to_vec(),
        b" me".to_vec(),
        b" would".to_vec(),
        b" make".to_vec(),
        b" like".to_vec(),
        b" into".to_vec(),
        b" him".to_vec(),
        b" time".to_vec(),
        b" two".to_vec(),
        b" more".to_vec(),
        b" very".to_vec(),
        b" after".to_vec(),
        b" back".to_vec(),
        b" other".to_vec(),
        b" many".to_vec(),
        b" than".to_vec(),
        b" first".to_vec(),
        b" well".to_vec(),
        b" way".to_vec(),
        b" even".to_vec(),
        b" new".to_vec(),
        b" want".to_vec(),
        b" because".to_vec(),
        b" any".to_vec(),
        b" these".to_vec(),
        b" give".to_vec(),
        b" day".to_vec(),
        b" most".to_vec(),
        b" us".to_vec(),
    ];

    for pair in common_pairs {
        ranks.insert(pair, rank);
        rank += 1;
    }

    Ok(ranks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_vocabulary() {
        let vocab = create_basic_vocabulary().unwrap();

        // Should have at least 256 single-byte tokens
        assert!(vocab.len() >= 256);

        // Should contain single byte tokens
        assert!(vocab.contains_key(&vec![b'a']));
        assert!(vocab.contains_key(&vec![b' ']));

        // Should contain some common pairs
        assert!(vocab.contains_key(&b" the".to_vec()));
        assert!(vocab.contains_key(&b"th".to_vec()));
    }

    #[test]
    fn test_parse_tiktoken_bpe() {
        let content = "aGVsbG8= 0\nd29ybGQ= 1\n";
        let vocab = parse_tiktoken_bpe(content).unwrap();

        assert_eq!(vocab.len(), 2);
        assert_eq!(vocab.get(&b"hello".to_vec()), Some(&0));
        assert_eq!(vocab.get(&b"world".to_vec()), Some(&1));
    }

    #[test]
    fn test_get_vocab_info() {
        assert!(get_vocab_info("cl100k_base").is_some());
        assert!(get_vocab_info("r50k_base").is_some());
        assert!(get_vocab_info("p50k_base").is_some());
        assert!(get_vocab_info("o200k_base").is_some());
        assert!(get_vocab_info("unknown").is_none());
    }
}
