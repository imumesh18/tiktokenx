#!/bin/bash
# Cleanup script for tiktoken_rust project

echo "🧹 Cleaning up tiktoken_rust project..."

# Clean Cargo build artifacts
echo "Cleaning Cargo build artifacts..."
cargo clean

# Remove any temporary files
echo "Removing temporary files..."
find . -name "*.tmp" -delete 2>/dev/null || true
find . -name "*.temp" -delete 2>/dev/null || true
find . -name "*~" -delete 2>/dev/null || true

# Remove any benchmark output files
echo "Removing benchmark output files..."
rm -f benchmark_output.txt 2>/dev/null || true
rm -f *.bench 2>/dev/null || true

# Remove any Python cache files
echo "Removing Python cache files..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Remove any editor backup files
echo "Removing editor backup files..."
find . -name ".#*" -delete 2>/dev/null || true
find . -name "#*#" -delete 2>/dev/null || true

echo "✅ Cleanup complete!"
echo ""
echo "📊 To run benchmarks: python3 scripts/fair_benchmark.py"
echo "🧪 To run tests: cargo test --all-features"
echo "📦 To build: cargo build --release"
echo "🚀 To publish: cargo publish --dry-run --allow-dirty"
