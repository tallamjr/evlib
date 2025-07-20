#!/bin/bash
set -e

echo "=== evlib Documentation CI Test Script ==="
echo "This script simulates the GitHub Actions documentation CI locally"
echo

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: Run this script from the evlib root directory"
    exit 1
fi

echo "SUCCESS: Running from evlib root directory"

# Check documentation structure
echo
echo "=== Checking Documentation Structure ==="
if [ -d "docs" ]; then
    md_files=$(find docs/ -name "*.md" | wc -l)
    echo "SUCCESS: Found $md_files markdown files in docs/"
else
    echo "ERROR: No docs/ directory found"
    exit 1
fi

# Check for Python code blocks
python_blocks=$(find docs/ README.md -type f -name "*.md" -exec grep -l '```python' {} \; | wc -l)
echo "SUCCESS: Found $python_blocks files with Python code blocks"

if [ $python_blocks -eq 0 ]; then
    echo "ERROR: No Python code blocks found!"
    exit 1
fi

# Check index file
if [ -f "docs/index.md" ]; then
    echo "SUCCESS: docs/index.md exists as documentation index"
    echo "File size: $(du -h docs/index.md | cut -f1)"
    if [ -s "docs/index.md" ]; then
        echo "SUCCESS: docs/index.md has content"
    else
        echo "ERROR: docs/index.md is empty"
        exit 1
    fi
else
    echo "ERROR: docs/index.md is missing"
    exit 1
fi

# Check test data
echo
echo "=== Checking Test Data ==="
if [ -f "data/slider_depth/events.txt" ]; then
    size=$(du -h data/slider_depth/events.txt | cut -f1)
    lines=$(wc -l < data/slider_depth/events.txt)
    echo "SUCCESS: Main test data file exists: $size, $lines lines"
else
    echo "ERROR: Main test data file missing: data/slider_depth/events.txt"
    echo "Available data files:"
    find data/ -name "*.txt" -o -name "*.h5" -o -name "*.raw" 2>/dev/null || echo "No data directory found"
    exit 1
fi

# Check if evlib is built
echo
echo "=== Checking evlib Build ==="
if python -c "import evlib; print(f'SUCCESS: evlib version available')" 2>/dev/null; then
    echo "SUCCESS: evlib is importable"
else
    echo "ERROR: evlib not built or not importable"
    echo "Run: maturin develop --release"
    exit 1
fi

# Run documentation tests
echo
echo "=== Running Documentation Tests ==="

echo "Testing README examples..."
if pytest --markdown-docs README.md -q --tb=no; then
    echo "SUCCESS: README tests passed"
else
    echo "ERROR: README tests failed"
    exit 1
fi

echo "Testing quickstart guide..."
if pytest --markdown-docs docs/getting-started/quickstart.md -q --tb=no; then
    echo "SUCCESS: Quickstart tests passed"
else
    echo "ERROR: Quickstart tests failed"
    exit 1
fi

echo "Testing API documentation..."
if pytest --markdown-docs docs/api/ -q --tb=no --maxfail=5; then
    echo "SUCCESS: API docs tests passed"
else
    echo "ERROR: API docs tests failed"
    exit 1
fi

echo "Testing comprehensive documentation..."
if pytest --markdown-docs README.md docs/ -q --tb=no --maxfail=10; then
    echo "SUCCESS: All documentation tests passed"
else
    echo "ERROR: Some documentation tests failed"
    exit 1
fi

# Performance test
echo
echo "=== Performance Test ==="
start_time=$(date +%s)
pytest --markdown-docs README.md docs/ -q --tb=no >/dev/null 2>&1
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "SUCCESS: Documentation tests completed in ${duration}s"

echo
echo "SUCCESS: All documentation CI checks passed!"
echo "Your documentation is ready for GitHub Actions CI"
