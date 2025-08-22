//! Model to encoding mappings

use crate::core::Encoding;
use crate::encodings::get_encoding;
use crate::errors::{Result, TiktokenError};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Registry of model name prefixes to encoding names
static MODEL_PREFIX_REGISTRY: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();

/// Registry of exact model names to encoding names
static MODEL_REGISTRY: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();

/// Initialize the model prefix registry
fn init_prefix_registry() -> HashMap<&'static str, &'static str> {
    let mut registry = HashMap::new();

    // Reasoning models
    registry.insert("o1-", "o200k_base");
    registry.insert("o3-", "o200k_base");
    registry.insert("o4-mini-", "o200k_base");

    // Chat models
    registry.insert("gpt-5-", "o200k_base");
    registry.insert("gpt-4.5-", "o200k_base");
    registry.insert("gpt-4.1-", "o200k_base");
    registry.insert("chatgpt-4o-", "o200k_base");
    registry.insert("gpt-4o-", "o200k_base");
    registry.insert("gpt-4-", "cl100k_base");
    registry.insert("gpt-3.5-turbo-", "cl100k_base");
    registry.insert("gpt-35-turbo-", "cl100k_base"); // Azure deployment name
    registry.insert("gpt-oss-", "o200k_harmony");

    // Fine-tuned models
    registry.insert("ft:gpt-4o", "o200k_base");
    registry.insert("ft:gpt-4", "cl100k_base");
    registry.insert("ft:gpt-3.5-turbo", "cl100k_base");
    registry.insert("ft:davinci-002", "cl100k_base");
    registry.insert("ft:babbage-002", "cl100k_base");

    registry
}

/// Initialize the exact model registry
fn init_model_registry() -> HashMap<&'static str, &'static str> {
    let mut registry = HashMap::new();

    // Reasoning models
    registry.insert("o1", "o200k_base");
    registry.insert("o3", "o200k_base");
    registry.insert("o4-mini", "o200k_base");

    // Chat models
    registry.insert("gpt-5", "o200k_base");
    registry.insert("gpt-4.1", "o200k_base");
    registry.insert("gpt-4o", "o200k_base");
    registry.insert("gpt-4", "cl100k_base");
    registry.insert("gpt-3.5-turbo", "cl100k_base");
    registry.insert("gpt-3.5", "cl100k_base");
    registry.insert("gpt-35-turbo", "cl100k_base"); // Azure deployment name

    // Base models
    registry.insert("davinci-002", "cl100k_base");
    registry.insert("babbage-002", "cl100k_base");

    // Embedding models
    registry.insert("text-embedding-ada-002", "cl100k_base");
    registry.insert("text-embedding-3-small", "cl100k_base");
    registry.insert("text-embedding-3-large", "cl100k_base");

    // DEPRECATED MODELS
    // Text models (DEPRECATED)
    registry.insert("text-davinci-003", "p50k_base");
    registry.insert("text-davinci-002", "p50k_base");
    registry.insert("text-davinci-001", "r50k_base");
    registry.insert("text-curie-001", "r50k_base");
    registry.insert("text-babbage-001", "r50k_base");
    registry.insert("text-ada-001", "r50k_base");
    registry.insert("davinci", "r50k_base");
    registry.insert("curie", "r50k_base");
    registry.insert("babbage", "r50k_base");
    registry.insert("ada", "r50k_base");

    // Code models (DEPRECATED)
    registry.insert("code-davinci-002", "p50k_base");
    registry.insert("code-davinci-001", "p50k_base");
    registry.insert("code-cushman-002", "p50k_base");
    registry.insert("code-cushman-001", "p50k_base");
    registry.insert("davinci-codex", "p50k_base");
    registry.insert("cushman-codex", "p50k_base");

    // Edit models (DEPRECATED)
    registry.insert("text-davinci-edit-001", "p50k_edit");
    registry.insert("code-davinci-edit-001", "p50k_edit");

    // Old embedding models (DEPRECATED)
    registry.insert("text-similarity-davinci-001", "r50k_base");
    registry.insert("text-similarity-curie-001", "r50k_base");
    registry.insert("text-similarity-babbage-001", "r50k_base");
    registry.insert("text-similarity-ada-001", "r50k_base");
    registry.insert("text-search-davinci-doc-001", "r50k_base");
    registry.insert("text-search-curie-doc-001", "r50k_base");
    registry.insert("text-search-babbage-doc-001", "r50k_base");
    registry.insert("text-search-ada-doc-001", "r50k_base");
    registry.insert("code-search-babbage-code-001", "r50k_base");
    registry.insert("code-search-ada-code-001", "r50k_base");

    // Open source models
    registry.insert("gpt2", "gpt2");
    registry.insert("gpt-2", "gpt2");

    registry
}

/// Get the encoding name for a model
pub fn encoding_name_for_model(model_name: &str) -> Result<String> {
    let model_registry = MODEL_REGISTRY.get_or_init(init_model_registry);
    let prefix_registry = MODEL_PREFIX_REGISTRY.get_or_init(init_prefix_registry);

    // First check exact matches
    if let Some(&encoding_name) = model_registry.get(model_name) {
        return Ok(encoding_name.to_string());
    }

    // Then check prefix matches
    for (&prefix, &encoding_name) in prefix_registry.iter() {
        if model_name.starts_with(prefix) {
            return Ok(encoding_name.to_string());
        }
    }

    Err(TiktokenError::UnknownModel(model_name.to_string()))
}

/// Get the encoding for a model
pub fn encoding_for_model(model_name: &str) -> Result<Encoding> {
    let encoding_name = encoding_name_for_model(model_name)?;
    get_encoding(&encoding_name)
}

/// List all supported model names
pub fn list_supported_models() -> Vec<String> {
    let model_registry = MODEL_REGISTRY.get_or_init(init_model_registry);
    model_registry.keys().map(|&s| s.to_string()).collect()
}

/// Check if a model is supported
pub fn is_model_supported(model_name: &str) -> bool {
    encoding_name_for_model(model_name).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_model_mapping() {
        assert_eq!(encoding_name_for_model("gpt-4").unwrap(), "cl100k_base");
        assert_eq!(encoding_name_for_model("gpt-3.5-turbo").unwrap(), "cl100k_base");
        assert_eq!(encoding_name_for_model("gpt-4o").unwrap(), "o200k_base");
        assert_eq!(encoding_name_for_model("text-davinci-003").unwrap(), "p50k_base");
    }

    #[test]
    fn test_prefix_model_mapping() {
        assert_eq!(encoding_name_for_model("gpt-4-0314").unwrap(), "cl100k_base");
        assert_eq!(encoding_name_for_model("gpt-4o-2024-05-13").unwrap(), "o200k_base");
        assert_eq!(encoding_name_for_model("gpt-3.5-turbo-0301").unwrap(), "cl100k_base");
    }

    #[test]
    fn test_unknown_model() {
        assert!(encoding_name_for_model("unknown-model").is_err());
    }

    #[test]
    fn test_encoding_for_model() {
        let encoding = encoding_for_model("gpt-4").unwrap();
        assert_eq!(encoding.name, "cl100k_base");
    }
}
