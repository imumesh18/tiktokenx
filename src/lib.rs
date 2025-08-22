//! # tiktoken_rust
//!
//! A high-performance Rust implementation of OpenAI's tiktoken library.
//!
//! This library provides fast byte pair encoding (BPE) tokenization compatible with OpenAI's models.
//! It supports all current OpenAI encodings including cl100k_base, p50k_base, r50k_base, and o200k_base.
//!
//! ## Features
//!
//! - **High Performance**: Optimized Rust implementation with zero-copy operations where possible
//! - **Full Compatibility**: Exact compatibility with OpenAI's tiktoken library
//! - **All Encodings**: Support for all OpenAI model encodings
//! - **Pure Rust**: Minimal dependencies using only standard library and well-maintained crates
//!
//! ## Quick Start
//!
//! ```rust
//! use tiktoken_rust::{get_encoding, encoding_for_model};
//!
//! // Get encoding by name
//! let enc = get_encoding("cl100k_base").unwrap();
//! let tokens = enc.encode("hello world", &[], &[]).unwrap();
//! let text = enc.decode(&tokens).unwrap();
//! assert_eq!(text, "hello world");
//!
//! // Get encoding for a specific model
//! let enc = encoding_for_model("gpt-4").unwrap();
//! let token_count = enc.encode("Hello, world!", &[], &[]).unwrap().len();
//! ```

pub mod core;
pub mod encodings;
pub mod errors;
pub mod models;
pub mod vendors;
pub mod vocab;

// Re-export main types and functions for convenience
pub use core::{CoreBPE, Encoding};
pub use encodings::{get_encoding, list_encodings};
pub use errors::{Result, TiktokenError};
pub use models::{encoding_for_model, encoding_name_for_model};
pub use vendors::{VendorProvider, VendorRegistry};

/// The main result type used throughout the library
pub type TiktokenResult<T> = std::result::Result<T, TiktokenError>;

/// Get encoding for any model from any supported vendor
pub fn get_encoding_for_any_model(model: &str) -> Result<Encoding> {
    let registry = VendorRegistry::new();
    if let Some(vendor) = registry.find_vendor_for_model(model) {
        let encoding_name = vendor.encoding_for_model(model)?;
        vendor.create_encoding(&encoding_name)
    } else {
        Err(TiktokenError::UnknownModel(model.to_string()))
    }
}

/// Get encoding from any supported vendor
pub fn get_encoding_from_any_vendor(encoding: &str) -> Result<Encoding> {
    let registry = VendorRegistry::new();
    if let Some(vendor) = registry.find_vendor_for_encoding(encoding) {
        vendor.create_encoding(encoding)
    } else {
        Err(TiktokenError::UnknownEncoding(encoding.to_string()))
    }
}

/// List all supported models from all vendors
pub fn list_all_supported_models() -> Vec<(String, String)> {
    let registry = VendorRegistry::new();
    registry
        .list_all_models()
        .into_iter()
        .map(|(vendor, model)| (vendor.to_string(), model.to_string()))
        .collect()
}

/// List all supported encodings from all vendors
pub fn list_all_supported_encodings() -> Vec<(String, String)> {
    let registry = VendorRegistry::new();
    registry
        .list_all_encodings()
        .into_iter()
        .map(|(vendor, encoding)| (vendor.to_string(), encoding.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_encoding() {
        let enc = get_encoding("cl100k_base").unwrap();
        let tokens = enc.encode("hello world", &[], &[]).unwrap();
        let decoded = enc.decode(&tokens).unwrap();
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_model_encoding() {
        let enc = encoding_for_model("gpt-4").unwrap();
        let tokens = enc.encode("test", &[], &[]).unwrap();
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_roundtrip() {
        let enc = get_encoding("cl100k_base").unwrap();
        let original = "The quick brown fox jumps over the lazy dog.";
        let tokens = enc.encode(original, &[], &[]).unwrap();
        let decoded = enc.decode(&tokens).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_all_encodings() {
        let encodings =
            ["r50k_base", "p50k_base", "p50k_edit", "cl100k_base", "o200k_base", "gpt2"];

        for encoding_name in &encodings {
            let enc = get_encoding(encoding_name).unwrap();
            let text = "Hello, world! This is a test.";
            let tokens = enc.encode(text, &[], &[]).unwrap();
            let decoded = enc.decode(&tokens).unwrap();
            assert_eq!(decoded, text, "Failed for encoding: {}", encoding_name);
        }
    }

    #[test]
    fn test_special_tokens() {
        let enc = get_encoding("cl100k_base").unwrap();
        let special_tokens = enc.special_tokens();

        // cl100k_base should have endoftext token
        assert!(special_tokens.contains_key("<|endoftext|>"));

        // Test encoding with special tokens
        let text = "Hello<|endoftext|>World";
        let tokens = enc.encode(text, &["<|endoftext|>"], &[]).unwrap();
        let decoded = enc.decode(&tokens).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_empty_text() {
        let enc = get_encoding("cl100k_base").unwrap();
        let tokens = enc.encode("", &[], &[]).unwrap();
        assert!(tokens.is_empty());

        let decoded = enc.decode(&[]).unwrap();
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_unicode_text() {
        let enc = get_encoding("cl100k_base").unwrap();
        let text = "Hello 世界! 🌍 Здравствуй мир!";
        let tokens = enc.encode(text, &[], &[]).unwrap();
        let decoded = enc.decode(&tokens).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_long_text() {
        let enc = get_encoding("cl100k_base").unwrap();
        let text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(100);
        let tokens = enc.encode(&text, &[], &[]).unwrap();
        let decoded = enc.decode(&tokens).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encode_ordinary() {
        let enc = get_encoding("cl100k_base").unwrap();
        let text = "Hello<|endoftext|>World";

        // encode_ordinary should ignore special tokens
        let tokens = enc.encode_ordinary(text);
        let decoded = enc.decode(&tokens).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_count_tokens() {
        let enc = get_encoding("cl100k_base").unwrap();
        let text = "Hello, world!";
        let token_count = enc.count_tokens(text);
        let tokens = enc.encode_ordinary(text);
        assert_eq!(token_count, tokens.len());
    }

    #[test]
    fn test_batch_operations() {
        let enc = get_encoding("cl100k_base").unwrap();
        let texts = vec!["Hello", "World", "Test"];

        let token_batches = enc.encode_ordinary_batch(&texts);
        assert_eq!(token_batches.len(), 3);

        let decoded_batch = enc
            .decode_batch(&token_batches.iter().map(|v| v.as_slice()).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(decoded_batch, texts);
    }

    #[test]
    fn test_vendor_registry() {
        let registry = VendorRegistry::new();

        // Test OpenAI models
        assert!(registry.find_vendor_for_model("gpt-4").is_some());
        assert!(registry.find_vendor_for_encoding("cl100k_base").is_some());

        // Test listing functions
        let models = list_all_supported_models();
        assert!(!models.is_empty());
        assert!(models.iter().any(|(vendor, model)| vendor == "openai" && model == "gpt-4"));

        let encodings = list_all_supported_encodings();
        assert!(!encodings.is_empty());
        assert!(encodings
            .iter()
            .any(|(vendor, encoding)| vendor == "openai" && encoding == "cl100k_base"));
    }

    #[test]
    fn test_multi_vendor_encoding() {
        // Test OpenAI model
        let enc = get_encoding_for_any_model("gpt-4").unwrap();
        assert_eq!(enc.name, "cl100k_base");

        // Test direct encoding access
        let enc = get_encoding_from_any_vendor("cl100k_base").unwrap();
        assert_eq!(enc.name, "cl100k_base");
    }

    #[test]
    fn test_unknown_vendor_model() {
        let result = get_encoding_for_any_model("unknown-model-12345");
        assert!(result.is_err());
    }
}
