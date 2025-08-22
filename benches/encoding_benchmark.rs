use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use peak_alloc::PeakAlloc;
use std::hint::black_box;
use tiktoken_rust::{encoding_for_model, get_encoding};

#[global_allocator]
static PEAK_ALLOC: PeakAlloc = PeakAlloc;

fn bench_encoding_short_text(c: &mut Criterion) {
    let enc = get_encoding("cl100k_base").unwrap();
    let text = "Hello, world! This is a short test.";

    let mut group = c.benchmark_group("encoding_short_text");
    group.throughput(Throughput::Bytes(text.len() as u64));

    group.bench_function("encode", |b| {
        b.iter(|| {
            let tokens = enc.encode(black_box(text), &[], &[]).unwrap();
            black_box(tokens)
        })
    });

    group.bench_function("encode_ordinary", |b| {
        b.iter(|| {
            let tokens = enc.encode_ordinary(black_box(text));
            black_box(tokens)
        })
    });

    let tokens = enc.encode(text, &[], &[]).unwrap();
    group.bench_function("decode", |b| {
        b.iter(|| {
            let decoded = enc.decode(black_box(&tokens)).unwrap();
            black_box(decoded)
        })
    });

    group.finish();
}

fn bench_encoding_medium_text(c: &mut Criterion) {
    let enc = get_encoding("cl100k_base").unwrap();
    let text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";

    let mut group = c.benchmark_group("encoding_medium_text");
    group.throughput(Throughput::Bytes(text.len() as u64));

    group.bench_function("encode", |b| {
        b.iter(|| {
            let tokens = enc.encode(black_box(text), &[], &[]).unwrap();
            black_box(tokens)
        })
    });

    group.bench_function("encode_ordinary", |b| {
        b.iter(|| {
            let tokens = enc.encode_ordinary(black_box(text));
            black_box(tokens)
        })
    });

    let tokens = enc.encode(text, &[], &[]).unwrap();
    group.bench_function("decode", |b| {
        b.iter(|| {
            let decoded = enc.decode(black_box(&tokens)).unwrap();
            black_box(decoded)
        })
    });

    group.finish();
}

fn bench_encoding_long_text(c: &mut Criterion) {
    let enc = get_encoding("cl100k_base").unwrap();
    let base_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. ";
    let text = base_text.repeat(10); // Create a longer text

    let mut group = c.benchmark_group("encoding_long_text");
    group.throughput(Throughput::Bytes(text.len() as u64));

    group.bench_function("encode", |b| {
        b.iter(|| {
            let tokens = enc.encode(black_box(&text), &[], &[]).unwrap();
            black_box(tokens)
        })
    });

    group.bench_function("encode_ordinary", |b| {
        b.iter(|| {
            let tokens = enc.encode_ordinary(black_box(&text));
            black_box(tokens)
        })
    });

    let tokens = enc.encode(&text, &[], &[]).unwrap();
    group.bench_function("decode", |b| {
        b.iter(|| {
            let decoded = enc.decode(black_box(&tokens)).unwrap();
            black_box(decoded)
        })
    });

    group.finish();
}

fn bench_batch_operations(c: &mut Criterion) {
    let enc = get_encoding("cl100k_base").unwrap();
    let texts: Vec<&str> = vec![
        "Hello, world!",
        "This is a test.",
        "Rust is awesome!",
        "Benchmarking performance.",
        "tiktoken_rust is fast!",
    ];
    let texts_100: Vec<&str> = (0..100).map(|i| texts[i % texts.len()]).collect();

    let mut group = c.benchmark_group("batch_operations");

    group.bench_with_input(BenchmarkId::new("encode_batch", texts.len()), &texts, |b, texts| {
        b.iter(|| {
            let results = enc.encode_batch(black_box(texts), &[], &[]).unwrap();
            black_box(results)
        })
    });

    group.bench_with_input(
        BenchmarkId::new("encode_ordinary_batch", texts.len()),
        &texts,
        |b, texts| {
            b.iter(|| {
                let results = enc.encode_ordinary_batch(black_box(texts));
                black_box(results)
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("encode_batch", texts_100.len()),
        &texts_100,
        |b, texts| {
            b.iter(|| {
                let results = enc.encode_batch(black_box(texts), &[], &[]).unwrap();
                black_box(results)
            })
        },
    );

    group.finish();
}

fn bench_different_encodings(c: &mut Criterion) {
    let text =
        "The quick brown fox jumps over the lazy dog. This is a test of different encodings.";
    let encodings = ["r50k_base", "p50k_base", "cl100k_base", "o200k_base"];

    let mut group = c.benchmark_group("different_encodings");
    group.throughput(Throughput::Bytes(text.len() as u64));

    for encoding_name in &encodings {
        let enc = get_encoding(encoding_name).unwrap();
        group.bench_with_input(BenchmarkId::new("encode", encoding_name), encoding_name, |b, _| {
            b.iter(|| {
                let tokens = enc.encode(black_box(text), &[], &[]).unwrap();
                black_box(tokens)
            })
        });
    }

    group.finish();
}

fn bench_model_encodings(c: &mut Criterion) {
    let text = "This is a test message for different model encodings.";
    let models = ["gpt-4", "gpt-3.5-turbo", "text-davinci-003"];

    let mut group = c.benchmark_group("model_encodings");
    group.throughput(Throughput::Bytes(text.len() as u64));

    for model in &models {
        let enc = encoding_for_model(model).unwrap();
        group.bench_with_input(BenchmarkId::new("encode", model), model, |b, _| {
            b.iter(|| {
                let tokens = enc.encode(black_box(text), &[], &[]).unwrap();
                black_box(tokens)
            })
        });
    }

    group.finish();
}

fn bench_special_tokens(c: &mut Criterion) {
    let enc = get_encoding("cl100k_base").unwrap();
    let text_with_special = "Hello <|endoftext|> World <|endoftext|> Test";
    let text_without_special = "Hello World Test";

    let mut group = c.benchmark_group("special_tokens");

    group.bench_function("with_special_tokens", |b| {
        b.iter(|| {
            let tokens = enc.encode(black_box(text_with_special), &["<|endoftext|>"], &[]).unwrap();
            black_box(tokens)
        })
    });

    group.bench_function("without_special_tokens", |b| {
        b.iter(|| {
            let tokens = enc.encode_ordinary(black_box(text_with_special));
            black_box(tokens)
        })
    });

    group.bench_function("regular_text", |b| {
        b.iter(|| {
            let tokens = enc.encode(black_box(text_without_special), &[], &[]).unwrap();
            black_box(tokens)
        })
    });

    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    let long_text = "Long text ".repeat(100);
    let texts = [
        "Short text",
        "This is a medium length text that should use more memory for tokenization and processing.",
        &long_text,
    ];

    for (i, text) in texts.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("encode_with_memory", i), text, |b, text| {
            b.iter_custom(|iters| {
                let enc = get_encoding("cl100k_base").unwrap();

                // Reset peak memory counter
                PEAK_ALLOC.reset_peak_usage();

                let start = std::time::Instant::now();
                for _ in 0..iters {
                    let tokens = enc.encode_ordinary(black_box(text));
                    black_box(tokens);
                }
                let duration = start.elapsed();

                // Get peak memory usage
                let peak_memory = PEAK_ALLOC.peak_usage_as_mb();
                println!("Peak memory usage for text {i}: {peak_memory:.2} MB");

                duration
            })
        });
    }

    group.finish();
}

fn bench_cpu_intensive_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_intensive");

    // Test with different text sizes to measure CPU scaling
    let sizes = [100, 1000, 10000];

    for size in sizes {
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(size / 45);

        group.bench_with_input(BenchmarkId::new("large_text_encoding", size), &text, |b, text| {
            let enc = get_encoding("cl100k_base").unwrap();
            b.iter(|| {
                let tokens = enc.encode_ordinary(black_box(text));
                black_box(tokens)
            })
        });

        // Also test decoding performance
        let enc = get_encoding("cl100k_base").unwrap();
        let tokens = enc.encode_ordinary(&text);

        group.bench_with_input(
            BenchmarkId::new("large_text_decoding", size),
            &tokens,
            |b, tokens| {
                b.iter(|| {
                    let decoded = enc.decode(black_box(tokens)).unwrap();
                    black_box(decoded)
                })
            },
        );
    }

    group.finish();
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    // Test memory efficiency of different operations
    let enc = get_encoding("cl100k_base").unwrap();
    let text = "This is a test text for memory efficiency benchmarking. ".repeat(100);

    group.bench_function("encoding_creation", |b| {
        b.iter_custom(|iters| {
            PEAK_ALLOC.reset_peak_usage();

            let start = std::time::Instant::now();
            for _ in 0..iters {
                let enc = get_encoding("cl100k_base").unwrap();
                black_box(enc);
            }
            let duration = start.elapsed();

            let peak_memory = PEAK_ALLOC.peak_usage_as_mb();
            println!("Peak memory for encoding creation: {peak_memory:.2} MB");

            duration
        })
    });

    group.bench_function("batch_vs_individual", |b| {
        let texts: Vec<&str> = (0..100).map(|_| text.as_str()).collect();

        b.iter_custom(|iters| {
            PEAK_ALLOC.reset_peak_usage();

            let start = std::time::Instant::now();
            for _ in 0..iters {
                // Batch operation
                let batch_results = enc.encode_ordinary_batch(black_box(&texts));
                black_box(batch_results);
            }
            let duration = start.elapsed();

            let peak_memory = PEAK_ALLOC.peak_usage_as_mb();
            println!("Peak memory for batch operations: {peak_memory:.2} MB");

            duration
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_encoding_short_text,
    bench_encoding_medium_text,
    bench_encoding_long_text,
    bench_batch_operations,
    bench_different_encodings,
    bench_model_encodings,
    bench_special_tokens,
    bench_memory_usage,
    bench_cpu_intensive_operations,
    bench_memory_efficiency
);
criterion_main!(benches);
