#!/usr/bin/env python3
"""
Benchmark validation script for evlib
Validates that the benchmarking infrastructure is working correctly
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

def run_command(cmd, timeout=300):
    """Run a command with timeout and return result"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout} seconds"

def validate_benchmark_compilation():
    """Validate that benchmarks can be compiled"""
    print("🔨 Validating benchmark compilation...")
    
    # Test compilation of standalone benchmark
    cmd = "cargo check --bench standalone_benchmark"
    returncode, stdout, stderr = run_command(cmd, timeout=180)
    
    if returncode == 0:
        print("✅ Standalone benchmark compiles successfully")
        return True
    else:
        print(f"❌ Benchmark compilation failed:")
        print(f"   Return code: {returncode}")
        print(f"   Stderr: {stderr}")
        return False

def validate_benchmark_execution():
    """Validate that benchmarks can be executed"""
    print("🚀 Validating benchmark execution...")
    
    # Try to run a quick benchmark test
    cmd = "cargo bench --bench standalone_benchmark -- --test"
    returncode, stdout, stderr = run_command(cmd, timeout=60)
    
    if returncode == 0:
        print("✅ Benchmark execution successful")
        return True
    else:
        print(f"⚠️  Benchmark execution failed (this may be expected due to library linking issues)")
        print(f"   Return code: {returncode}")
        print(f"   Stderr: {stderr}")
        return False

def validate_benchmark_structure():
    """Validate benchmark file structure"""
    print("📁 Validating benchmark structure...")
    
    project_root = Path(__file__).parent.parent
    benches_dir = project_root / "benches"
    
    expected_files = [
        "memory_efficiency.rs",
        "streaming_performance.rs", 
        "format_comparison.rs",
        "basic_performance.rs",
        "standalone_benchmark.rs"
    ]
    
    missing_files = []
    for file in expected_files:
        if not (benches_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing benchmark files: {missing_files}")
        return False
    else:
        print("✅ All benchmark files present")
        return True

def validate_cargo_configuration():
    """Validate Cargo.toml benchmark configuration"""
    print("⚙️  Validating Cargo.toml configuration...")
    
    project_root = Path(__file__).parent.parent
    cargo_toml = project_root / "Cargo.toml"
    
    if not cargo_toml.exists():
        print("❌ Cargo.toml not found")
        return False
    
    with open(cargo_toml, 'r') as f:
        content = f.read()
    
    # Check for benchmark configurations
    required_configs = [
        "criterion",
        "[[bench]]",
        "harness = false"
    ]
    
    missing_configs = []
    for config in required_configs:
        if config not in content:
            missing_configs.append(config)
    
    if missing_configs:
        print(f"❌ Missing Cargo.toml configurations: {missing_configs}")
        return False
    else:
        print("✅ Cargo.toml properly configured for benchmarks")
        return True

def validate_benchmark_dependencies():
    """Validate benchmark dependencies"""
    print("📦 Validating benchmark dependencies...")
    
    # Check if criterion is available
    cmd = "cargo metadata --format-version 1"
    returncode, stdout, stderr = run_command(cmd, timeout=30)
    
    if returncode != 0:
        print(f"❌ Failed to get cargo metadata: {stderr}")
        return False
    
    try:
        metadata = json.loads(stdout)
        packages = metadata.get("packages", [])
        
        # Check for criterion dependency
        criterion_found = False
        for package in packages:
            if package.get("name") == "criterion":
                criterion_found = True
                break
        
        if criterion_found:
            print("✅ Criterion dependency found")
            return True
        else:
            print("❌ Criterion dependency not found")
            return False
    except json.JSONDecodeError:
        print("❌ Failed to parse cargo metadata")
        return False

def generate_benchmark_report():
    """Generate a summary report of benchmark capabilities"""
    print("\n📊 Benchmark Capability Report")
    print("=" * 50)
    
    # Performance metrics that can be measured
    metrics = [
        "✅ Event generation throughput",
        "✅ Memory usage estimation", 
        "✅ Chunk size optimization",
        "✅ Format comparison",
        "✅ Streaming vs direct performance",
        "✅ Data type efficiency",
        "✅ Polarity encoding overhead",
        "✅ Timestamp conversion speed"
    ]
    
    print("Available Performance Metrics:")
    for metric in metrics:
        print(f"   {metric}")
    
    # Benchmark categories
    categories = [
        "Memory Efficiency",
        "Streaming Performance", 
        "Format Comparison",
        "Basic Performance",
        "Standalone Algorithms"
    ]
    
    print(f"\nBenchmark Categories ({len(categories)}):")
    for i, category in enumerate(categories, 1):
        print(f"   {i}. {category}")
    
    # Expected performance characteristics
    print("\nExpected Performance Characteristics:")
    print("   • Event processing: > 1M events/second")
    print("   • Memory efficiency: < 30 bytes/event")
    print("   • Streaming crossover: ~5M events")
    print("   • Format detection: < 1ms")
    
    print("\nRecommendations:")
    print("   • Run benchmarks regularly to detect regressions")
    print("   • Use streaming for files > 5M events")
    print("   • Monitor memory usage for large datasets")
    print("   • Profile with realistic data patterns")

def main():
    """Main validation function"""
    print("🔍 evlib Benchmark Validation")
    print("=" * 40)
    
    # Run all validation checks
    checks = [
        ("Benchmark Structure", validate_benchmark_structure),
        ("Cargo Configuration", validate_cargo_configuration),
        ("Benchmark Dependencies", validate_benchmark_dependencies),
        ("Benchmark Compilation", validate_benchmark_compilation),
        ("Benchmark Execution", validate_benchmark_execution),
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        result = check_func()
        results.append((check_name, result))
    
    # Summary
    print("\n📋 Validation Summary")
    print("=" * 25)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {check_name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All validations passed! Benchmarks are ready to use.")
        generate_benchmark_report()
        return 0
    else:
        print("⚠️  Some validations failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())