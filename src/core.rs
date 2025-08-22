//! Core BPE implementation and encoding structures

use crate::errors::{Result, TiktokenError};
use regex::Regex;
use std::collections::HashMap;

/// Token type - using u32 to match original tiktoken
pub type Token = u32;

/// Rank type for BPE merge priorities
pub type Rank = u32;

/// Core BPE encoder/decoder with all necessary data
#[derive(Clone)]
pub struct CoreBPE {
    /// Maps byte sequences to their token IDs
    encoder: HashMap<Vec<u8>, Token>,
    /// Maps token IDs to their byte sequences
    decoder: HashMap<Token, Vec<u8>>,
    /// Maps special token strings to their token IDs
    special_tokens_encoder: HashMap<String, Token>,
    /// Maps special token IDs to their byte sequences
    special_tokens_decoder: HashMap<Token, Vec<u8>>,
    /// Regex for splitting text into pieces
    regex: Regex,
    /// Regex for finding special tokens
    special_regex: Option<Regex>,
}

impl CoreBPE {
    /// Create a new CoreBPE instance
    pub fn new(
        mergeable_ranks: HashMap<Vec<u8>, Rank>,
        special_tokens: HashMap<String, Token>,
        pattern: &str,
    ) -> Result<Self> {
        let regex = Regex::new(pattern)?;

        // Build encoder from mergeable ranks
        let encoder: HashMap<Vec<u8>, Token> = mergeable_ranks;

        // Build decoder
        let decoder: HashMap<Token, Vec<u8>> =
            encoder.iter().map(|(bytes, &token)| (token, bytes.clone())).collect();

        // Build special token decoder
        let special_tokens_decoder: HashMap<Token, Vec<u8>> = special_tokens
            .iter()
            .map(|(text, &token)| (token, text.as_bytes().to_vec()))
            .collect();

        // Build special token regex if we have special tokens
        let special_regex = if special_tokens.is_empty() {
            None
        } else {
            let escaped_tokens: Vec<String> =
                special_tokens.keys().map(|s| regex::escape(s)).collect();
            Some(Regex::new(&escaped_tokens.join("|"))?)
        };

        Ok(CoreBPE {
            encoder,
            decoder,
            special_tokens_encoder: special_tokens,
            special_tokens_decoder,
            regex,
            special_regex,
        })
    }

    /// Encode text to tokens, ignoring special tokens
    pub fn encode_ordinary(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::new();

        for mat in self.regex.find_iter(text) {
            let piece = mat.as_str().as_bytes();

            // Check if this piece is a single token
            if let Some(&token) = self.encoder.get(piece) {
                tokens.push(token);
            } else {
                // Apply BPE to this piece
                tokens.extend(self.byte_pair_encode(piece));
            }
        }

        tokens
    }

    /// Encode text to tokens with special token handling
    pub fn encode(
        &self,
        text: &str,
        allowed_special: &[&str],
        disallowed_special: &[&str],
    ) -> Result<Vec<Token>> {
        // Check for disallowed special tokens
        if !disallowed_special.is_empty() {
            if let Some(ref special_regex) = self.special_regex {
                for mat in special_regex.find_iter(text) {
                    let token_text = mat.as_str();
                    if disallowed_special.contains(&token_text)
                        && !allowed_special.contains(&token_text)
                    {
                        return Err(TiktokenError::EncodingError(format!(
                            "Disallowed special token: {token_text}"
                        )));
                    }
                }
            }
        }

        let mut tokens = Vec::new();
        let mut start = 0;

        // Process text, handling special tokens
        while start < text.len() {
            let mut next_special_start = text.len();
            let mut next_special_end = text.len();
            let mut found_special = None;

            // Find the next allowed special token
            if let Some(ref special_regex) = self.special_regex {
                for mat in special_regex.find_iter(&text[start..]) {
                    let token_text = &text[start + mat.start()..start + mat.end()];
                    if allowed_special.contains(&token_text) {
                        next_special_start = start + mat.start();
                        next_special_end = start + mat.end();
                        found_special = Some(token_text);
                        break;
                    }
                }
            }

            // Encode the text before the special token
            if next_special_start > start {
                let ordinary_text = &text[start..next_special_start];
                tokens.extend(self.encode_ordinary(ordinary_text));
            }

            // Add the special token if found
            if let Some(special_token) = found_special {
                if let Some(&token) = self.special_tokens_encoder.get(special_token) {
                    tokens.push(token);
                }
                start = next_special_end;
            } else {
                // No more special tokens, we're done
                break;
            }
        }

        Ok(tokens)
    }

    /// Decode tokens back to bytes
    pub fn decode_bytes(&self, tokens: &[Token]) -> Result<Vec<u8>> {
        let mut result = Vec::new();

        for &token in tokens {
            if let Some(bytes) = self.decoder.get(&token) {
                result.extend_from_slice(bytes);
            } else if let Some(bytes) = self.special_tokens_decoder.get(&token) {
                result.extend_from_slice(bytes);
            } else {
                return Err(TiktokenError::InvalidToken(token));
            }
        }

        Ok(result)
    }

    /// Decode tokens back to string
    pub fn decode(&self, tokens: &[Token]) -> Result<String> {
        let bytes = self.decode_bytes(tokens)?;
        String::from_utf8(bytes).map_err(TiktokenError::from)
    }

    /// Get the byte sequence for a single token
    pub fn decode_single_token_bytes(&self, token: Token) -> Result<&[u8]> {
        if let Some(bytes) = self.decoder.get(&token) {
            Ok(bytes)
        } else if let Some(bytes) = self.special_tokens_decoder.get(&token) {
            Ok(bytes)
        } else {
            Err(TiktokenError::InvalidToken(token))
        }
    }

    /// Apply byte pair encoding to a piece of text
    fn byte_pair_encode(&self, piece: &[u8]) -> Vec<Token> {
        if piece.len() == 1 {
            return vec![self.encoder[piece]];
        }

        let parts = self.byte_pair_merge(piece);
        parts
            .windows(2)
            .map(|window| {
                let start = window[0].0;
                let end = window[1].0;
                self.encoder[&piece[start..end]]
            })
            .collect()
    }

    /// Core BPE merge algorithm
    fn byte_pair_merge(&self, piece: &[u8]) -> Vec<(usize, Rank)> {
        // This is a vector of (start, rank).
        // The rank is of the pair starting at position start.
        let mut parts = Vec::with_capacity(piece.len() + 1);

        // Find initial ranks for all adjacent pairs
        let mut min_rank = (Rank::MAX, usize::MAX);
        for i in 0..piece.len() - 1 {
            let pair = &piece[i..i + 2];
            let rank = self.encoder.get(pair).copied().unwrap_or(Rank::MAX);

            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
            parts.push((i, rank));
        }
        parts.push((piece.len() - 1, Rank::MAX));
        parts.push((piece.len(), Rank::MAX));

        // Iteratively merge the pair with the lowest rank
        while min_rank.0 != Rank::MAX {
            let i = min_rank.1;

            // Update ranks for adjacent pairs before removing the middle element
            if i > 0 {
                parts[i - 1].1 = self.get_pair_rank(piece, &parts, i - 1);
            }
            parts[i].1 = self.get_pair_rank(piece, &parts, i);

            // Remove the middle element
            parts.remove(i + 1);

            // Find the new minimum rank
            min_rank = (Rank::MAX, usize::MAX);
            for (idx, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
                if rank < min_rank.0 {
                    min_rank = (rank, idx);
                }
            }
        }

        parts
    }

    /// Get the rank of a pair at a given position
    fn get_pair_rank(&self, piece: &[u8], parts: &[(usize, Rank)], i: usize) -> Rank {
        if i + 3 < parts.len() {
            let start = parts[i].0;
            let end = parts[i + 3].0;
            self.encoder.get(&piece[start..end]).copied().unwrap_or(Rank::MAX)
        } else {
            Rank::MAX
        }
    }

    /// Get all special tokens
    pub fn special_tokens(&self) -> &HashMap<String, Token> {
        &self.special_tokens_encoder
    }

    /// Check if a token is a special token
    pub fn is_special_token(&self, token: Token) -> bool {
        self.special_tokens_decoder.contains_key(&token)
    }

    /// Get the maximum token value
    pub fn max_token_value(&self) -> Token {
        let max_regular = self.decoder.keys().max().copied().unwrap_or(0);
        let max_special = self.special_tokens_decoder.keys().max().copied().unwrap_or(0);
        max_regular.max(max_special)
    }

    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.decoder.len() + self.special_tokens_decoder.len()
    }
}

/// High-level encoding interface that wraps CoreBPE
#[derive(Clone)]
pub struct Encoding {
    /// The name of this encoding
    pub name: String,
    /// The core BPE implementation
    core: CoreBPE,
}

impl Encoding {
    /// Create a new encoding
    pub fn new(
        name: String,
        mergeable_ranks: HashMap<Vec<u8>, Rank>,
        special_tokens: HashMap<String, Token>,
        pattern: &str,
    ) -> Result<Self> {
        let core = CoreBPE::new(mergeable_ranks, special_tokens, pattern)?;
        Ok(Encoding { name, core })
    }

    /// Encode text to tokens, ignoring special tokens
    pub fn encode_ordinary(&self, text: &str) -> Vec<Token> {
        self.core.encode_ordinary(text)
    }

    /// Encode text to tokens with special token handling
    pub fn encode(
        &self,
        text: &str,
        allowed_special: &[&str],
        disallowed_special: &[&str],
    ) -> Result<Vec<Token>> {
        self.core.encode(text, allowed_special, disallowed_special)
    }

    /// Decode tokens back to string
    pub fn decode(&self, tokens: &[Token]) -> Result<String> {
        self.core.decode(tokens)
    }

    /// Decode tokens back to bytes
    pub fn decode_bytes(&self, tokens: &[Token]) -> Result<Vec<u8>> {
        self.core.decode_bytes(tokens)
    }

    /// Get the byte sequence for a single token
    pub fn decode_single_token_bytes(&self, token: Token) -> Result<&[u8]> {
        self.core.decode_single_token_bytes(token)
    }

    /// Get all special tokens
    pub fn special_tokens(&self) -> &HashMap<String, Token> {
        self.core.special_tokens()
    }

    /// Check if a token is a special token
    pub fn is_special_token(&self, token: Token) -> bool {
        self.core.is_special_token(token)
    }

    /// Get the maximum token value
    pub fn max_token_value(&self) -> Token {
        self.core.max_token_value()
    }

    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.core.vocab_size()
    }

    /// Count tokens in text (convenience method)
    pub fn count_tokens(&self, text: &str) -> usize {
        self.encode_ordinary(text).len()
    }

    /// Encode a batch of texts
    pub fn encode_batch(
        &self,
        texts: &[&str],
        allowed_special: &[&str],
        disallowed_special: &[&str],
    ) -> Result<Vec<Vec<Token>>> {
        texts
            .iter()
            .map(|&text| self.encode(text, allowed_special, disallowed_special))
            .collect()
    }

    /// Encode a batch of texts, ignoring special tokens
    pub fn encode_ordinary_batch(&self, texts: &[&str]) -> Vec<Vec<Token>> {
        texts.iter().map(|&text| self.encode_ordinary(text)).collect()
    }

    /// Decode a batch of token sequences
    pub fn decode_batch(&self, token_sequences: &[&[Token]]) -> Result<Vec<String>> {
        token_sequences.iter().map(|&tokens| self.decode(tokens)).collect()
    }

    /// Decode a batch of token sequences to bytes
    pub fn decode_bytes_batch(&self, token_sequences: &[&[Token]]) -> Result<Vec<Vec<u8>>> {
        token_sequences.iter().map(|&tokens| self.decode_bytes(tokens)).collect()
    }

    /// Encode a single token from text
    pub fn encode_single_token(&self, text: &str) -> Result<Token> {
        let bytes = text.as_bytes();

        // Check if it's a special token first
        if let Some(&token) = self.special_tokens().get(text) {
            return Ok(token);
        }

        // Check if it's a regular token
        if let Some(&token) = self.core.encoder.get(bytes) {
            return Ok(token);
        }

        Err(TiktokenError::EncodingError(format!(
            "Text '{text}' does not correspond to a single token"
        )))
    }

    /// Get all token byte values
    pub fn token_byte_values(&self) -> Vec<Vec<u8>> {
        let mut values: Vec<Vec<u8>> = self.core.decoder.values().cloned().collect();
        values.extend(self.core.special_tokens_decoder.values().cloned());
        values.sort();
        values
    }

    /// Get the end-of-text token if it exists
    pub fn eot_token(&self) -> Option<Token> {
        self.special_tokens().get("<|endoftext|>").copied()
    }
}
