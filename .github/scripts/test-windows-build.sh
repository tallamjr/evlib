#!/bin/bash
# Test script for validating Windows build compatibility using act
# NOTE: This requires Docker and act to be installed

set -e

echo "=== Windows Build Testing Script ==="
echo ""

# Check if act is installed
if ! command -v act &> /dev/null; then
    echo "Error: 'act' is not installed"
    echo "Install with: brew install act (macOS) or see https://github.com/nektos/act"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "Error: Docker is not running"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

echo "✓ act is installed: $(act --version)"
echo "✓ Docker is running"
echo ""

# Important note about Windows containers on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "⚠️  WARNING: Running Windows containers on macOS has limitations"
    echo "   Windows containers require a Windows host or special configuration"
    echo ""
    echo "   Recommended approach:"
    echo "   1. Push to GitHub and let GitHub Actions run the actual Windows build"
    echo "   2. Use this script to test Ubuntu/Linux builds locally"
    echo "   3. For Windows testing, use a Windows machine or GitHub Actions"
    echo ""
    read -p "Continue with Linux container test instead? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi

    # Test with Ubuntu instead
    echo "Testing with Ubuntu runner (Linux build)..."
    act -W .github/workflows/test-windows-build.yml \
        --platform windows-latest=catthehacker/ubuntu:act-latest \
        -j test-windows-wheels \
        --matrix python-version:3.12 \
        --matrix os:ubuntu-latest \
        --dryrun
else
    echo "Testing Windows build workflow..."
    # On Linux/Windows hosts, try the actual Windows test
    act -W .github/workflows/test-windows-build.yml \
        -j test-windows-wheels \
        --matrix python-version:3.12 \
        --dryrun
fi

echo ""
echo "=== Dry run complete ==="
echo ""
echo "To run the full test:"
echo "  act -W .github/workflows/test-windows-build.yml -j test-windows-wheels"
echo ""
echo "To test a specific Python version:"
echo "  act -W .github/workflows/test-windows-build.yml --matrix python-version:3.11"
echo ""
echo "To push and test on GitHub Actions:"
echo "  git push origin <branch>"
echo "  gh workflow run test-windows-build.yml"
