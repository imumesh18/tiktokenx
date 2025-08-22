# tiktoken_rust

[![Crates.io](https://img.shields.io/crates/v/tiktoken_rust.svg)](https://crates.io/crates/tiktoken_rust)
[![Documentation](https://docs.rs/tiktoken_rust/badge.svg)](https://docs.rs/tiktoken_rust)
[![Build Status](https://github.com/imumesh18/tiktoken_rust/workflows/CI/badge.svg)](https://github.com/imumesh18/tiktoken_rust/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Fast Rust implementation of OpenAI's tiktoken tokenizer.

## Features

- Drop-in replacement for Python tiktoken
- All OpenAI models supported (GPT-4, GPT-5, o1, etc.)
- Real vocabulary file loading with hash verification
- Zero-copy operations for optimal performance
- Comprehensive test suite (24 tests)

## Installation

```toml
[dependencies]
tiktoken_rust = "0.1"
```

## Usage

```rust
use tiktoken_rust::{get_encoding, encoding_for_model};

// Get encoding by name
let enc = get_encoding("cl100k_base").unwrap();
let tokens = enc.encode("hello world", &[], &[]).unwrap();
let text = enc.decode(&tokens).unwrap();

// Get encoding for a model
let enc = encoding_for_model("gpt-4").unwrap();
let token_count = enc.encode("Hello, world!", &[], &[]).unwrap().len();
```

## Supported Models

| Model Family | Models                             | Encoding                 |
| ------------ | ---------------------------------- | ------------------------ |
| GPT-5        | gpt-5                              | o200k_base               |
| GPT-4        | gpt-4, gpt-4-turbo, gpt-4o         | cl100k_base / o200k_base |
| GPT-3.5      | gpt-3.5-turbo                      | cl100k_base              |
| o1           | o1, o1-mini, o1-preview            | o200k_base               |
| Legacy       | text-davinci-003, code-davinci-002 | p50k_base                |

## Performance

Benchmarks on Apple M1 Pro comparing tiktoken_rust vs Python tiktoken:

| Implementation  | Operation         | Time     | Throughput | vs Python |
| --------------- | ----------------- | -------- | ---------- | --------- |
| Python tiktoken | Encode short text | 5.5 μs   | 5.0 MiB/s  | 1.0x      |
| tiktoken_rust   | Encode short text | 4.1 μs   | 6.7 MiB/s  | **1.3x**  |
| Python tiktoken | Encode long text  | 472.8 μs | 9.1 MiB/s  | 1.0x      |
| tiktoken_rust   | Encode long text  | 175.4 μs | 24.5 MiB/s | **2.7x**  |

**tiktoken_rust is 2.0x faster at encoding on average!**

## Development

```bash
# Run tests
cargo test

# Run benchmarks
cargo bench

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy -- -D warnings
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT
