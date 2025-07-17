# evlib: Advanced Memory Optimization & Streaming Implementation Plan

## **🎉 PHASE 1 COMPLETED: Direct Polars Series Construction**

### **✅ ACHIEVEMENTS**
- **Memory Optimization**: Direct Polars Series builders implemented
- **Zero-Copy Architecture**: Single iteration, no intermediate collections
- **Format-Specific Encoding**: EVT2 (-1/1), HDF5/Text (0/1) polarity handling
- **Regression Tests**: All existing tests pass without modification
- **Performance**: 600k+ events/s loading, 400M+ events/s filtering
- **API Compatibility**: Zero breaking changes to public API

---

## **🚀 PHASE 2: STREAMING PATTERN IMPLEMENTATION (NEXT PRIORITY)**

### **2.1 IMMEDIATE TASKS**

#### **Task 2.1.1: Implement PolarsEventStreamer Struct**
**Priority**: HIGH  
**Estimated Time**: 2-3 hours  
**Description**: Create streaming infrastructure for large files (>100M events)

**Detailed Steps**:
1. Create new file: `src/ev_formats/streaming.rs`
2. Implement `PolarsEventStreamer` struct with:
   ```rust
   pub struct PolarsEventStreamer {
       chunk_size: usize,
       format: EventFormat,
       memory_limit_mb: usize,
   }
   ```
3. Add methods:
   - `new(chunk_size, format)` - Constructor
   - `stream_to_polars(events)` - Main streaming function
   - `build_chunk(events)` - Build single chunk DataFrame
   - `calculate_optimal_chunk_size()` - Adaptive sizing
4. Handle chunk concatenation with `polars::concat()`
5. Add comprehensive error handling for memory exhaustion

**Validation Criteria**:
- [ ] Can process 100M+ events without memory issues
- [ ] Chunk concatenation produces identical results to direct loading
- [ ] Memory usage stays below configured limits
- [ ] All existing functionality preserved

#### **Task 2.1.2: Integrate Streaming into load_events_py**
**Priority**: HIGH  
**Estimated Time**: 1-2 hours  
**Description**: Add automatic streaming detection to main API

**Detailed Steps**:
1. Modify `load_events_py()` in `src/ev_formats/mod.rs`
2. Add threshold detection (default: 5M events)
3. Implement fallback logic:
   ```rust
   let df = if events.len() > 5_000_000 {
       // Use streaming for large datasets
       let streamer = PolarsEventStreamer::new(1_000_000, format_result.format);
       streamer.stream_to_polars(events.into_iter())?.collect()?
   } else {
       // Direct construction for smaller datasets
       build_polars_dataframe(&events, format_result.format)?
   };
   ```
4. Add configuration option for streaming threshold
5. Ensure seamless transition between modes

**Validation Criteria**:
- [ ] Small files (<5M events) use direct construction
- [ ] Large files (>5M events) automatically use streaming
- [ ] API remains unchanged for end users
- [ ] Performance maintained or improved

#### **Task 2.1.3: Memory Management & Adaptive Chunking**
**Priority**: MEDIUM  
**Estimated Time**: 2 hours  
**Description**: Implement intelligent memory management

**Detailed Steps**:
1. Add system memory detection:
   ```rust
   fn get_available_memory() -> usize {
       // Platform-specific memory detection
       // Use sysinfo crate or similar
   }
   ```
2. Implement adaptive chunk sizing:
   ```rust
   fn calculate_optimal_chunk_size(total_events: usize, available_memory_mb: usize) -> usize {
       const BYTES_PER_EVENT: usize = 15; // Updated estimate
       let target_memory = (available_memory_mb * 1024 * 1024) / 4; // 25% of RAM
       (target_memory / BYTES_PER_EVENT).clamp(100_000, 10_000_000)
   }
   ```
3. Add progress reporting for large files
4. Implement graceful degradation for low-memory systems

**Validation Criteria**:
- [ ] Chunk size adapts to available system memory
- [ ] Progress reporting works for files >1GB
- [ ] Graceful handling of memory-constrained environments
- [ ] No memory leaks during streaming

---

## **🔧 PHASE 3: PERFORMANCE INTEGRATION & BENCHMARKING**

### **3.1 AUTOMATED BENCHMARKING SYSTEM**

#### **Task 3.1.1: Create Comprehensive Benchmark Suite**
**Priority**: MEDIUM  
**Estimated Time**: 3-4 hours  
**Description**: Automated performance regression detection

**Detailed Steps**:
1. Create `benches/` directory with Rust criterion benchmarks
2. Implement benchmarks:
   ```rust
   // benches/memory_efficiency.rs
   use criterion::{black_box, criterion_group, criterion_main, Criterion};
   
   fn benchmark_memory_usage(c: &mut Criterion) {
       let events = generate_test_events(1_000_000);
       
       c.bench_function("load_events_direct", |b| {
           b.iter(|| build_polars_dataframe(black_box(&events), EventFormat::HDF5))
       });
       
       c.bench_function("load_events_streaming", |b| {
           b.iter(|| {
               let streamer = PolarsEventStreamer::new(100_000, EventFormat::HDF5);
               streamer.stream_to_polars(black_box(events.iter().cloned()))
           })
       });
   }
   ```
3. Add memory profiling integration
4. Create performance regression detection
5. Add CI/CD integration with GitHub Actions

**Validation Criteria**:
- [ ] Benchmarks run in <10 minutes
- [ ] Memory usage tracking accurate within 5%
- [ ] Performance regression alerts functional
- [ ] Integration with CI/CD pipeline

#### **Task 3.1.2: Real-World Performance Testing**
**Priority**: MEDIUM  
**Estimated Time**: 2 hours  
**Description**: Test with actual large datasets

**Detailed Steps**:
1. Create test suite for large files:
   - 10M events (small streaming test)
   - 100M events (medium streaming test)
   - 500M+ events (large streaming test)
2. Add benchmark for different file formats:
   - EVT2 streaming performance
   - HDF5 streaming performance
   - Text file streaming performance
3. Create memory efficiency comparisons
4. Add performance baselines

**Validation Criteria**:
- [ ] Streaming handles 500M+ events efficiently
- [ ] Memory usage linear with chunk size, not total events
- [ ] Performance within 10% of direct loading for small files
- [ ] All formats supported in streaming mode

---

## **📚 PHASE 4: DOCUMENTATION & USER EXPERIENCE**

### **4.1 DOCUMENTATION UPDATES**

#### **Task 4.1.1: Update README with Performance Metrics**
**Priority**: MEDIUM  
**Estimated Time**: 1 hour  
**Description**: Showcase optimization achievements

**Detailed Steps**:
1. Add performance section to README.md:
   ```markdown
   ## Performance Optimizations
   
   ### Memory Efficiency
   - **Direct Polars Integration**: Zero-copy architecture
   - **Memory Usage**: ~110 bytes/event (includes overhead)
   - **Streaming Support**: Files >100M events automatically streamed
   
   ### Processing Speed  
   - **Load Speed**: 600k+ events/s
   - **Filter Speed**: 400M+ events/s (LazyFrame operations)
   - **Format Support**: All formats optimized
   ```
2. Add benchmarking instructions
3. Update installation guide for optimal performance
4. Add troubleshooting section for large files

#### **Task 4.1.2: Create Performance Monitoring Tools**
**Priority**: LOW  
**Estimated Time**: 2 hours  
**Description**: User-facing performance analysis tools

**Detailed Steps**:
1. Enhanced benchmark script (`benchmark_advanced.py`):
   ```python
   def profile_loading_performance(file_path, streaming_threshold=5_000_000):
       """Compare direct vs streaming performance"""
       # Memory usage profiling
       # Loading speed comparison  
       # Streaming threshold optimization
   ```
2. Memory monitoring dashboard
3. Performance recommendation system
4. Usage pattern analysis

---

## **🔬 PHASE 5: ADVANCED OPTIMIZATIONS (FUTURE)**

### **5.1 MEMORY MAPPING INTEGRATION**

#### **Task 5.1.1: Direct File-to-Polars Pipeline**
**Priority**: LOW  
**Estimated Time**: 1-2 days  
**Description**: Ultimate zero-copy file loading

**Technical Approach**:
1. Implement memory-mapped file readers
2. Direct Arrow array construction from mapped memory
3. Bypass intermediate Event structs entirely
4. Platform-specific optimizations (Linux, macOS, Windows)

### **5.2 PARALLEL PROCESSING**

#### **Task 5.2.1: Multi-threaded Chunk Processing**
**Priority**: LOW  
**Estimated Time**: 2-3 days  
**Description**: Parallel streaming for massive files

**Technical Approach**:
1. Thread-safe chunk processing
2. Parallel Series builder construction
3. Lock-free data structure optimization
4. NUMA-aware memory allocation

---

## **📋 IMPLEMENTATION PRIORITIES & TIMELINE**

### **IMMEDIATE (This Week)**
1. **Task 2.1.1**: Implement PolarsEventStreamer ⚡ HIGH
2. **Task 2.1.2**: Integrate streaming into main API ⚡ HIGH
3. **Task 2.1.3**: Memory management & adaptive chunking 🔧 MEDIUM

### **SHORT TERM (Next 2 Weeks)**
4. **Task 3.1.1**: Automated benchmark suite 📊 MEDIUM
5. **Task 3.1.2**: Real-world performance testing 📊 MEDIUM
6. **Task 4.1.1**: Documentation updates 📚 MEDIUM

### **FUTURE (When Resources Available)**
7. **Task 4.1.2**: Performance monitoring tools 📚 LOW
8. **Task 5.1.1**: Memory mapping integration 🔬 LOW
9. **Task 5.2.1**: Parallel processing 🔬 LOW

---

## **🎯 SUCCESS METRICS**

### **Performance Targets**
- **Streaming Throughput**: >1M events/s for large files
- **Memory Efficiency**: <150 bytes/event total memory usage
- **Scalability**: Handle 1B+ events without memory exhaustion
- **Compatibility**: Zero breaking changes to existing API

### **Quality Gates**
- [ ] All regression tests pass
- [ ] Streaming functionality tested with >100M events
- [ ] Memory usage benchmarks show continued improvement
- [ ] Documentation updated with performance metrics
- [ ] CI/CD integration functional

### **User Experience Goals**
- **Automatic Optimization**: Users don't need to think about streaming
- **Predictable Performance**: Consistent behavior across file sizes
- **Clear Feedback**: Progress reporting for large operations
- **Error Recovery**: Graceful handling of memory constraints

---

## **🔧 TECHNICAL IMPLEMENTATION DETAILS**

### **Dependencies to Add**
```toml
[dependencies]
sysinfo = "0.29"  # For system memory detection (optional)

[dev-dependencies]
criterion = "0.5"  # For benchmarking
```

### **File Structure**
```
src/
├── ev_formats/
│   ├── mod.rs              # Updated with streaming integration
│   ├── streaming.rs        # New: PolarsEventStreamer implementation
│   └── polarity_conversion.rs  # Helper functions
benches/
├── memory_efficiency.rs   # New: Memory benchmarks
├── streaming_performance.rs # New: Streaming benchmarks
└── regression_detection.rs # New: Performance regression tests
```

### **Configuration Options**
```rust
// Add to LoadConfig
pub struct LoadConfig {
    // ... existing fields
    pub streaming_threshold: Option<usize>,  // Default: 5_000_000
    pub chunk_size: Option<usize>,           // Default: adaptive
    pub memory_limit_mb: Option<usize>,      // Default: 25% of system RAM
    pub progress_callback: Option<Box<dyn Fn(f64)>>, // Progress reporting
}
```

---

## **🚨 RISK MITIGATION**

### **Technical Risks**
- **Memory Fragmentation**: Use pre-allocated chunk sizes
- **Concatenation Overhead**: Benchmark chunk concatenation performance  
- **Type Consistency**: Ensure streaming produces identical types to direct loading
- **Error Propagation**: Robust error handling in streaming pipeline

### **Performance Risks**
- **Regression Detection**: Automated benchmarking catches performance drops
- **Memory Monitoring**: Track memory usage in CI/CD
- **Scalability Testing**: Regular testing with large datasets

### **Compatibility Risks**
- **API Stability**: No changes to public interface
- **Test Coverage**: Comprehensive regression test suite
- **Gradual Rollout**: Feature flags for streaming functionality

This plan provides a clear roadmap for implementing advanced streaming capabilities while maintaining the excellent foundation we've built with the direct Polars integration.