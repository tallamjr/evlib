# Comprehensive Benchmarking Suite for evlib

This document describes the comprehensive benchmarking system implemented for the evlib event camera processing library. The benchmarks are designed to measure performance characteristics, detect regressions, and provide insights into optimal usage patterns.

## Overview

The benchmarking suite consists of multiple benchmark files, each focusing on different aspects of the library's performance:

- **Memory Efficiency**: Direct vs streaming loading patterns
- **Streaming Performance**: Chunk size optimization and adaptive algorithms
- **Format Comparison**: Performance across different event data formats
- **Basic Performance**: Fundamental operations and data structures
- **Standalone Benchmark**: Self-contained performance measurements

## Running Benchmarks

### Prerequisites

1. Ensure you have the Rust toolchain installed
2. Install the library dependencies:
   ```bash
   cargo build --release
   ```

### Basic Benchmark Execution

```bash
# Run all benchmarks
cargo bench

# Run a specific benchmark
cargo bench --bench standalone_benchmark

# Run benchmarks with HTML reports
cargo bench --bench standalone_benchmark -- --output-format html

# Run benchmarks with quick profile (fewer iterations)
cargo bench --bench standalone_benchmark -- --profile-time 10
```

### Benchmark Categories

#### 1. Memory Efficiency Benchmarks

**File**: `benches/memory_efficiency.rs`

Tests different approaches to memory management:

- **Direct vs Streaming**: Compare loading entire datasets vs chunk-based processing
- **Memory Usage Patterns**: Analyze memory consumption with different data sizes
- **Chunk Size Optimization**: Find optimal chunk sizes for different memory constraints
- **Data Type Efficiency**: Compare memory usage of different data type strategies

**Key Metrics**:
- Events processed per second
- Memory usage per event
- Optimal chunk sizes for different memory constraints
- Memory efficiency ratios

**Example Results**:
```
memory_efficiency_direct_vs_streaming/direct_loading/1000000
                        time:   [45.2 ms 45.8 ms 46.4 ms]
                        thrpt:  [21.6 Mevents/s 21.8 Mevents/s 22.1 Mevents/s]

memory_efficiency_direct_vs_streaming/streaming_loading/1000000
                        time:   [52.1 ms 52.7 ms 53.3 ms]
                        thrpt:  [18.8 Mevents/s 19.0 Mevents/s 19.2 Mevents/s]
```

#### 2. Streaming Performance Benchmarks

**File**: `benches/streaming_performance.rs`

Focuses on streaming algorithms and chunked processing:

- **Chunk Size Performance**: Test different chunk sizes for streaming
- **Adaptive Chunk Sizing**: Benchmark automatic chunk size calculation
- **Concatenation Overhead**: Measure cost of combining chunks
- **Crossover Analysis**: Find the point where streaming becomes beneficial

**Key Metrics**:
- Throughput (events/second) for different chunk sizes
- Optimal chunk sizes for different file sizes
- Concatenation performance overhead
- Streaming vs direct performance crossover point

#### 3. Format Comparison Benchmarks

**File**: `benches/format_comparison.rs`

Compares performance across different event data formats:

- **Format Detection Speed**: How quickly formats are identified
- **Loading Performance**: Compare loading speed across formats
- **Polarity Encoding**: Performance impact of different polarity encodings
- **Format-Specific Optimizations**: Measure format-specific performance characteristics

**Key Metrics**:
- Format detection time
- Loading throughput per format
- Polarity conversion overhead
- Format-specific memory usage

#### 4. Standalone Benchmarks

**File**: `benches/standalone_benchmark.rs`

Self-contained benchmarks that don't depend on the full library:

- **Core Algorithm Performance**: Fundamental operations
- **Memory Efficiency**: Direct memory usage measurements
- **Data Processing**: Basic event processing algorithms
- **Crossover Point Analysis**: When to use streaming vs direct processing

**Key Metrics**:
- Basic operation performance
- Memory usage patterns
- Algorithm efficiency
- Performance crossover points

## Benchmark Configuration

### Criterion Configuration

The benchmarks use the Criterion crate with the following configuration:

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};

// Sample benchmark configuration
fn configure_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("benchmark_name");

    // Configure throughput measurement
    group.throughput(Throughput::Elements(event_count as u64));

    // Configure sample size for large operations
    group.sample_size(10);

    // Configure measurement time
    group.measurement_time(Duration::from_secs(30));

    // Configure warm-up time
    group.warm_up_time(Duration::from_secs(3));
}
```

### Performance Thresholds

The benchmarks include performance regression detection with the following thresholds:

- **Loading Performance**: > 1M events/second for basic operations
- **Memory Usage**: < 30 bytes per event including overhead
- **Streaming Crossover**: ~5M events (configurable)
- **Format Detection**: < 1ms for typical files

## Interpreting Results

### Throughput Measurements

Throughput is measured in events per second (events/s) and provides the primary performance metric:

```
Events/second = Total Events / Execution Time
```

### Memory Efficiency

Memory efficiency is calculated as:

```
Memory Efficiency = (Theoretical Minimum Memory) / (Actual Memory Usage)
```

Where theoretical minimum is the raw size of event data.

### Performance Regression Detection

The benchmarks include automated regression detection:

1. **Baseline Performance**: Establish performance baselines
2. **Threshold Monitoring**: Alert when performance drops below thresholds
3. **Comparative Analysis**: Compare against previous runs
4. **Memory Usage Bounds**: Enforce memory usage limits

## Optimization Recommendations

Based on benchmark results, the following recommendations are generated:

### File Size Recommendations

- **< 1M events**: Use direct loading for optimal performance
- **1M - 5M events**: Consider streaming based on memory constraints
- **5M - 50M events**: Use streaming with optimized chunk sizes
- **> 50M events**: Always use streaming with adaptive chunk sizing

### Memory Optimization

- **Available Memory > 1GB**: Use larger chunk sizes (1-2M events)
- **Available Memory < 512MB**: Use smaller chunk sizes (100-500K events)
- **Memory Constrained**: Enable streaming mode regardless of file size

### Format-Specific Optimizations

- **HDF5 Files**: Optimal for large datasets, good compression
- **EVT2 Files**: Fast loading, moderate memory usage
- **Text Files**: Good for debugging, slower for large datasets

## Continuous Integration

The benchmarks can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions configuration
- name: Run Benchmarks
  run: |
    cargo bench --bench standalone_benchmark -- --output-format json > bench_results.json

- name: Check Performance Regression
  run: |
    # Compare against baseline results
    python scripts/check_benchmark_regression.py bench_results.json
```

## Performance Monitoring

### Key Performance Indicators (KPIs)

1. **Loading Throughput**: Events/second for different file sizes
2. **Memory Efficiency**: Memory usage per event
3. **Streaming Overhead**: Performance cost of chunked processing
4. **Format Detection Speed**: Time to identify file formats

### Alerting Thresholds

- **Performance Degradation**: > 20% slower than baseline
- **Memory Usage Increase**: > 50% more memory per event
- **Detection Failure**: Format detection taking > 10ms

## Best Practices

### Benchmark Design

1. **Reproducible Data**: Use deterministic random number generation
2. **Realistic Workloads**: Test with data patterns similar to real usage
3. **Multiple Scenarios**: Test edge cases and typical usage patterns
4. **Proper Warm-up**: Allow JIT compilation and cache warming

### Result Interpretation

1. **Statistical Significance**: Use confidence intervals and multiple runs
2. **Variance Analysis**: Understand performance variability
3. **Trend Monitoring**: Track performance over time
4. **Comparative Analysis**: Compare against alternatives

## Future Enhancements

### Planned Improvements

1. **Real Data Integration**: Benchmark with actual event camera data
2. **GPU Acceleration**: Add GPU performance benchmarks
3. **Network Streaming**: Benchmark network-based streaming
4. **Compression Analysis**: Compare different compression algorithms

### Advanced Metrics

1. **Cache Performance**: L1/L2 cache hit rates
2. **Branch Prediction**: Branch misses and prediction accuracy
3. **Memory Access Patterns**: Memory access efficiency
4. **Power Consumption**: Energy efficiency measurements

## Conclusion

The comprehensive benchmarking suite provides detailed insights into evlib's performance characteristics. The benchmarks enable:

- **Performance Optimization**: Identify bottlenecks and optimization opportunities
- **Regression Detection**: Automatically detect performance regressions
- **Usage Guidance**: Provide recommendations for optimal usage patterns
- **Continuous Improvement**: Track performance improvements over time

Regular benchmark execution ensures the library maintains high performance standards and helps guide future development decisions.
