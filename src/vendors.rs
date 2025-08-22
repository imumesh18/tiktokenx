//! Vendor-specific implementations for different AI providers

use crate::core::Encoding;
use crate::errors::{Result, TiktokenError};
use crate::vocab;
use std::collections::HashMap;

/// Trait for vendor-specific encoding implementations
pub trait VendorProvider {
    /// Get the vendor name
    fn name(&self) -> &'static str;

    /// Get all available encodings for this vendor
    fn available_encodings(&self) -> Vec<&'static str>;

    /// Get all available models for this vendor
    fn available_models(&self) -> Vec<&'static str>;

    /// Get encoding name for a model
    fn encoding_for_model(&self, model: &str) -> Result<String>;

    /// Create an encoding by name
    fn create_encoding(&self, name: &str) -> Result<Encoding>;

    /// Check if this vendor supports a given model
    fn supports_model(&self, model: &str) -> bool {
        self.available_models().contains(&model)
    }

    /// Check if this vendor supports a given encoding
    fn supports_encoding(&self, encoding: &str) -> bool {
        self.available_encodings().contains(&encoding)
    }
}

/// OpenAI vendor implementation
pub struct OpenAIProvider;

impl VendorProvider for OpenAIProvider {
    fn name(&self) -> &'static str {
        "openai"
    }

    fn available_encodings(&self) -> Vec<&'static str> {
        vec!["r50k_base", "p50k_base", "p50k_edit", "cl100k_base", "o200k_base", "gpt2"]
    }

    fn available_models(&self) -> Vec<&'static str> {
        vec![
            // Reasoning models
            "o1",
            "o3",
            "o4-mini",
            // Chat models
            "gpt-4.1",
            "gpt-4o",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-3.5",
            "gpt-35-turbo",
            // Base models
            "davinci-002",
            "babbage-002",
            // Embedding models
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
            // Legacy models
            "text-davinci-003",
            "text-davinci-002",
            "text-davinci-001",
            "text-curie-001",
            "text-babbage-001",
            "text-ada-001",
            "davinci",
            "curie",
            "babbage",
            "ada",
            "code-davinci-002",
            "code-davinci-001",
            "code-cushman-002",
            "code-cushman-001",
            "davinci-codex",
            "cushman-codex",
            "text-davinci-edit-001",
            "code-davinci-edit-001",
            "gpt2",
            "gpt-2",
        ]
    }

    fn encoding_for_model(&self, model: &str) -> Result<String> {
        crate::models::encoding_name_for_model(model)
    }

    fn create_encoding(&self, name: &str) -> Result<Encoding> {
        crate::encodings::get_encoding(name)
    }
}

/// Anthropic vendor implementation (placeholder for future support)
pub struct AnthropicProvider;

impl VendorProvider for AnthropicProvider {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    fn available_encodings(&self) -> Vec<&'static str> {
        vec![
            // Placeholder - would be implemented when Anthropic releases their tokenizer
            "claude_base",
        ]
    }

    fn available_models(&self) -> Vec<&'static str> {
        vec![
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
        ]
    }

    fn encoding_for_model(&self, model: &str) -> Result<String> {
        match model {
            m if m.starts_with("claude") => Ok("claude_base".to_string()),
            _ => Err(TiktokenError::UnknownModel(model.to_string())),
        }
    }

    fn create_encoding(&self, name: &str) -> Result<Encoding> {
        match name {
            "claude_base" => {
                // Placeholder implementation - would load actual Claude tokenizer
                let ranks = vocab::create_basic_vocabulary()?;
                let special_tokens = HashMap::new();
                Encoding::new(
                    name.to_string(),
                    ranks,
                    special_tokens,
                    r"\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+|\s+",
                )
            }
            _ => Err(TiktokenError::UnknownEncoding(name.to_string())),
        }
    }
}

/// xAI vendor implementation (placeholder for future support)
pub struct XAIProvider;

impl VendorProvider for XAIProvider {
    fn name(&self) -> &'static str {
        "xai"
    }

    fn available_encodings(&self) -> Vec<&'static str> {
        vec![
            // Placeholder - would be implemented when xAI releases their tokenizer
            "grok_base",
        ]
    }

    fn available_models(&self) -> Vec<&'static str> {
        vec!["grok-1", "grok-1.5", "grok-2"]
    }

    fn encoding_for_model(&self, model: &str) -> Result<String> {
        match model {
            m if m.starts_with("grok") => Ok("grok_base".to_string()),
            _ => Err(TiktokenError::UnknownModel(model.to_string())),
        }
    }

    fn create_encoding(&self, name: &str) -> Result<Encoding> {
        match name {
            "grok_base" => {
                // Placeholder implementation - would load actual Grok tokenizer
                let ranks = vocab::create_basic_vocabulary()?;
                let special_tokens = HashMap::new();
                Encoding::new(
                    name.to_string(),
                    ranks,
                    special_tokens,
                    r"\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+|\s+",
                )
            }
            _ => Err(TiktokenError::UnknownEncoding(name.to_string())),
        }
    }
}

/// Registry for all vendor providers
pub struct VendorRegistry {
    providers: HashMap<&'static str, Box<dyn VendorProvider>>,
}

impl VendorRegistry {
    /// Create a new vendor registry with default providers
    pub fn new() -> Self {
        let mut registry = VendorRegistry { providers: HashMap::new() };

        // Register default providers
        registry.register_provider(Box::new(OpenAIProvider));
        registry.register_provider(Box::new(AnthropicProvider));
        registry.register_provider(Box::new(XAIProvider));

        registry
    }

    /// Register a new vendor provider
    pub fn register_provider(&mut self, provider: Box<dyn VendorProvider>) {
        self.providers.insert(provider.name(), provider);
    }

    /// Get a provider by name
    pub fn get_provider(&self, name: &str) -> Option<&dyn VendorProvider> {
        self.providers.get(name).map(|p| p.as_ref())
    }

    /// Find which vendor supports a given model
    pub fn find_vendor_for_model(&self, model: &str) -> Option<&dyn VendorProvider> {
        self.providers
            .values()
            .find(|provider| provider.supports_model(model))
            .map(|p| p.as_ref())
    }

    /// Find which vendor supports a given encoding
    pub fn find_vendor_for_encoding(&self, encoding: &str) -> Option<&dyn VendorProvider> {
        self.providers
            .values()
            .find(|provider| provider.supports_encoding(encoding))
            .map(|p| p.as_ref())
    }

    /// List all available vendors
    pub fn list_vendors(&self) -> Vec<&'static str> {
        self.providers.keys().copied().collect()
    }

    /// List all available models across all vendors
    pub fn list_all_models(&self) -> Vec<(&'static str, &'static str)> {
        let mut models = Vec::new();
        for provider in self.providers.values() {
            for model in provider.available_models() {
                models.push((provider.name(), model));
            }
        }
        models
    }

    /// List all available encodings across all vendors
    pub fn list_all_encodings(&self) -> Vec<(&'static str, &'static str)> {
        let mut encodings = Vec::new();
        for provider in self.providers.values() {
            for encoding in provider.available_encodings() {
                encodings.push((provider.name(), encoding));
            }
        }
        encodings
    }
}

impl Default for VendorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_provider() {
        let provider = OpenAIProvider;
        assert_eq!(provider.name(), "openai");
        assert!(provider.supports_model("gpt-4"));
        assert!(provider.supports_encoding("cl100k_base"));
        assert!(!provider.supports_model("claude-3-opus"));
    }

    #[test]
    fn test_vendor_registry() {
        let registry = VendorRegistry::new();

        // Test finding vendors
        assert!(registry.find_vendor_for_model("gpt-4").is_some());
        assert!(registry.find_vendor_for_model("claude-3-opus").is_some());
        assert!(registry.find_vendor_for_model("grok-1").is_some());

        // Test listing
        let vendors = registry.list_vendors();
        assert!(vendors.contains(&"openai"));
        assert!(vendors.contains(&"anthropic"));
        assert!(vendors.contains(&"xai"));
    }

    #[test]
    fn test_encoding_creation() {
        let provider = OpenAIProvider;
        let encoding = provider.create_encoding("cl100k_base").unwrap();
        assert_eq!(encoding.name, "cl100k_base");
    }
}
