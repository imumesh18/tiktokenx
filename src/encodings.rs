//! Encoding definitions and registry

use crate::core::{Encoding, Rank};
use crate::errors::{Result, TiktokenError};
use crate::vocab;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Type alias for encoding constructor functions
type EncodingConstructor = fn() -> Result<Encoding>;

/// Registry of all available encodings
static ENCODING_REGISTRY: OnceLock<HashMap<String, EncodingConstructor>> = OnceLock::new();

/// Initialize the encoding registry
fn init_registry() -> HashMap<String, EncodingConstructor> {
    let mut registry = HashMap::new();

    registry.insert("r50k_base".to_string(), r50k_base as EncodingConstructor);
    registry.insert("p50k_base".to_string(), p50k_base as EncodingConstructor);
    registry.insert("p50k_edit".to_string(), p50k_edit as EncodingConstructor);
    registry.insert("cl100k_base".to_string(), cl100k_base as EncodingConstructor);
    registry.insert("o200k_base".to_string(), o200k_base as EncodingConstructor);
    registry.insert("gpt2".to_string(), gpt2 as EncodingConstructor);

    registry
}

/// Get an encoding by name
pub fn get_encoding(name: &str) -> Result<Encoding> {
    let registry = ENCODING_REGISTRY.get_or_init(init_registry);

    if let Some(constructor) = registry.get(name) {
        constructor()
    } else {
        Err(TiktokenError::UnknownEncoding(name.to_string()))
    }
}

/// List all available encoding names
pub fn list_encodings() -> Vec<String> {
    let registry = ENCODING_REGISTRY.get_or_init(init_registry);
    registry.keys().cloned().collect()
}

// Special token constants
const ENDOFTEXT: &str = "<|endoftext|>";
const FIM_PREFIX: &str = "<|fim_prefix|>";
const FIM_MIDDLE: &str = "<|fim_middle|>";
const FIM_SUFFIX: &str = "<|fim_suffix|>";
const ENDOFPROMPT: &str = "<|endofprompt|>";

// Regex patterns
const R50K_PAT_STR: &str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+";

const CL100K_PAT_STR: &str = r"\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+|\s+";

const O200K_PAT_STR: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+";

/// Create the r50k_base encoding
pub fn r50k_base() -> Result<Encoding> {
    let mergeable_ranks = load_r50k_base_ranks()?;
    let mut special_tokens = HashMap::new();
    special_tokens.insert(ENDOFTEXT.to_string(), 50256);

    Encoding::new("r50k_base".to_string(), mergeable_ranks, special_tokens, R50K_PAT_STR)
}

/// Create the p50k_base encoding
pub fn p50k_base() -> Result<Encoding> {
    let mergeable_ranks = load_p50k_base_ranks()?;
    let mut special_tokens = HashMap::new();
    special_tokens.insert(ENDOFTEXT.to_string(), 50256);

    Encoding::new("p50k_base".to_string(), mergeable_ranks, special_tokens, R50K_PAT_STR)
}

/// Create the p50k_edit encoding
pub fn p50k_edit() -> Result<Encoding> {
    let mergeable_ranks = load_p50k_base_ranks()?;
    let mut special_tokens = HashMap::new();
    special_tokens.insert(ENDOFTEXT.to_string(), 50256);
    special_tokens.insert(FIM_PREFIX.to_string(), 50281);
    special_tokens.insert(FIM_MIDDLE.to_string(), 50282);
    special_tokens.insert(FIM_SUFFIX.to_string(), 50283);

    Encoding::new("p50k_edit".to_string(), mergeable_ranks, special_tokens, R50K_PAT_STR)
}

/// Create the cl100k_base encoding
pub fn cl100k_base() -> Result<Encoding> {
    let mergeable_ranks = load_cl100k_base_ranks()?;
    let mut special_tokens = HashMap::new();
    special_tokens.insert(ENDOFTEXT.to_string(), 100257);
    special_tokens.insert(FIM_PREFIX.to_string(), 100258);
    special_tokens.insert(FIM_MIDDLE.to_string(), 100259);
    special_tokens.insert(FIM_SUFFIX.to_string(), 100260);
    special_tokens.insert(ENDOFPROMPT.to_string(), 100276);

    Encoding::new("cl100k_base".to_string(), mergeable_ranks, special_tokens, CL100K_PAT_STR)
}

/// Create the o200k_base encoding
pub fn o200k_base() -> Result<Encoding> {
    let mergeable_ranks = load_o200k_base_ranks()?;
    let mut special_tokens = HashMap::new();
    special_tokens.insert(ENDOFTEXT.to_string(), 199999);
    special_tokens.insert(ENDOFPROMPT.to_string(), 200018);

    Encoding::new("o200k_base".to_string(), mergeable_ranks, special_tokens, O200K_PAT_STR)
}

/// Create the gpt2 encoding (same as r50k_base)
pub fn gpt2() -> Result<Encoding> {
    let mergeable_ranks = load_r50k_base_ranks()?;
    let mut special_tokens = HashMap::new();
    special_tokens.insert(ENDOFTEXT.to_string(), 50256);

    Encoding::new("gpt2".to_string(), mergeable_ranks, special_tokens, R50K_PAT_STR)
}

// Vocabulary loading functions
// In a real implementation, these would load from embedded data or external files

/// Load r50k_base vocabulary ranks
fn load_r50k_base_ranks() -> Result<HashMap<Vec<u8>, Rank>> {
    vocab::load_tiktoken_bpe("r50k_base")
}

/// Load p50k_base vocabulary ranks
fn load_p50k_base_ranks() -> Result<HashMap<Vec<u8>, Rank>> {
    vocab::load_tiktoken_bpe("p50k_base")
}

/// Load cl100k_base vocabulary ranks
fn load_cl100k_base_ranks() -> Result<HashMap<Vec<u8>, Rank>> {
    vocab::load_tiktoken_bpe("cl100k_base")
}

/// Load o200k_base vocabulary ranks
fn load_o200k_base_ranks() -> Result<HashMap<Vec<u8>, Rank>> {
    vocab::load_tiktoken_bpe("o200k_base")
}
