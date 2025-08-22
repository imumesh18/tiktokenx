//! Error types for the tiktoken_rust library

use std::fmt;

/// The main error type for tiktoken operations
#[derive(Debug, Clone)]
pub enum TiktokenError {
    /// Error when an encoding name is not recognized
    UnknownEncoding(String),

    /// Error when a model name is not recognized
    UnknownModel(String),

    /// Error when a token cannot be decoded
    InvalidToken(u32),

    /// Error when text cannot be encoded
    EncodingError(String),

    /// Error when tokens cannot be decoded to valid UTF-8
    DecodingError(String),

    /// Error when regex compilation fails
    RegexError(String),

    /// Error when loading encoding data
    DataError(String),

    /// Generic error for other cases
    Other(String),
}

impl fmt::Display for TiktokenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TiktokenError::UnknownEncoding(name) => {
                write!(f, "Unknown encoding: {name}")
            }
            TiktokenError::UnknownModel(name) => {
                write!(f, "Unknown model: {name}")
            }
            TiktokenError::InvalidToken(token) => {
                write!(f, "Invalid token: {token}")
            }
            TiktokenError::EncodingError(msg) => {
                write!(f, "Encoding error: {msg}")
            }
            TiktokenError::DecodingError(msg) => {
                write!(f, "Decoding error: {msg}")
            }
            TiktokenError::RegexError(msg) => {
                write!(f, "Regex error: {msg}")
            }
            TiktokenError::DataError(msg) => {
                write!(f, "Data error: {msg}")
            }
            TiktokenError::Other(msg) => {
                write!(f, "Error: {msg}")
            }
        }
    }
}

impl std::error::Error for TiktokenError {}

impl From<regex::Error> for TiktokenError {
    fn from(err: regex::Error) -> Self {
        TiktokenError::RegexError(err.to_string())
    }
}

impl From<std::string::FromUtf8Error> for TiktokenError {
    fn from(err: std::string::FromUtf8Error) -> Self {
        TiktokenError::DecodingError(err.to_string())
    }
}

/// Convenience type alias for Results in this library
pub type Result<T> = std::result::Result<T, TiktokenError>;
