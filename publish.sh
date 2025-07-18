#!/bin/bash

# Publishing script for evlib to PyPI and crates.io
# Usage: ./publish.sh [--test] [--pypi-only] [--crates-only] [--skip-checks]

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
TEST_MODE=false
PYPI_ONLY=false
CRATES_ONLY=false
SKIP_CHECKS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            shift
            ;;
        --pypi-only)
            PYPI_ONLY=true
            shift
            ;;
        --crates-only)
            CRATES_ONLY=true
            shift
            ;;
        --skip-checks)
            SKIP_CHECKS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--test] [--pypi-only] [--crates-only] [--skip-checks]"
            echo "  --test         Publish to test repositories only"
            echo "  --pypi-only    Only publish to PyPI (skip crates.io)"
            echo "  --crates-only  Only publish to crates.io (skip PyPI)"
            echo "  --skip-checks  Skip pre-publication checks"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Helper functions
print_step() {
    echo -e "${BLUE}CONFIG: $1${NC}"
}

print_success() {
    echo -e "${GREEN}SUCCESS: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

# Get version from Cargo.toml
get_version() {
    grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/'
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Pre-publication checks
run_checks() {
    if [ "$SKIP_CHECKS" = true ]; then
        print_warning "Skipping pre-publication checks"
        return 0
    fi

    print_step "Running pre-publication checks..."

    # Check if we're on a clean git state
    if [ -n "$(git status --porcelain)" ]; then
        print_error "Working directory is not clean. Please commit or stash changes."
        git status --short
        exit 1
    fi

    # Check if we're on master/main branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    if [ "$current_branch" != "master" ] && [ "$current_branch" != "main" ]; then
        print_warning "Not on master/main branch (current: $current_branch)"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Check required tools
    for tool in cargo maturin git; do
        if ! command_exists "$tool"; then
            print_error "$tool is required but not installed"
            exit 1
        fi
    done

    # Run cargo check
    print_step "Running cargo check..."
    if ! cargo check --release --features python; then
        print_error "Cargo check failed"
        exit 1
    fi

    # Run tests
    print_step "Running tests..."
    if ! cargo test --release --lib; then
        print_error "Tests failed"
        exit 1
    fi

    # Check if maturin can build
    print_step "Testing maturin build..."
    if ! maturin build --release --features python >/dev/null 2>&1; then
        print_error "Maturin build failed"
        exit 1
    fi

    print_success "All checks passed!"
}

# Publish to crates.io
publish_to_crates() {
    if [ "$PYPI_ONLY" = true ]; then
        return 0
    fi

    print_step "Publishing to crates.io..."

    if [ "$TEST_MODE" = true ]; then
        print_warning "Test mode: Would publish to crates.io with: cargo publish --dry-run"
        cargo publish --dry-run
    else
        # Check if already published
        version=$(get_version)
        if cargo search evlib | grep -q "evlib = \"$version\""; then
            print_warning "Version $version already exists on crates.io"
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                return 0
            fi
        fi

        print_step "Publishing to crates.io (version: $version)..."
        cargo publish
        print_success "Published to crates.io!"
    fi
}

# Publish to PyPI
publish_to_pypi() {
    if [ "$CRATES_ONLY" = true ]; then
        return 0
    fi

    print_step "Publishing to PyPI..."

    # Clean previous builds
    rm -rf dist/ target/wheels/

    # Build wheel with Python features
    print_step "Building wheel with Python features..."
    maturin build --release --features python

    if [ "$TEST_MODE" = true ]; then
        print_step "Publishing to TestPyPI..."
        if ! maturin publish --repository testpypi --features python; then
            print_error "Failed to publish to TestPyPI"
            exit 1
        fi

        version=$(get_version)
        print_success "Published to TestPyPI!"
        print_step "Test installation with:"
        echo "  pip install --index-url https://test.pypi.org/simple/ evlib==$version"

        read -p "Test installation successful? Continue to real PyPI? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_warning "Stopping at TestPyPI"
            return 0
        fi
    fi

    # Publish to real PyPI
    print_step "Publishing to PyPI..."
    if ! maturin publish --features python; then
        print_error "Failed to publish to PyPI"
        exit 1
    fi

    print_success "Published to PyPI!"
}

# Main execution
main() {
    version=$(get_version)

    echo "LAUNCH: Publishing evlib v$version"
    echo "Target repositories:"
    if [ "$PYPI_ONLY" = true ]; then
        echo "  - PyPI only"
    elif [ "$CRATES_ONLY" = true ]; then
        echo "  - crates.io only"
    else
        echo "  - PyPI"
        echo "  - crates.io"
    fi

    if [ "$TEST_MODE" = true ]; then
        echo "  - TEST MODE: Using test repositories"
    fi

    echo ""

    # Confirmation prompt
    if [ "$TEST_MODE" = false ]; then
        print_warning "This will publish to PRODUCTION repositories!"
        read -p "Are you sure you want to continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_warning "Aborted by user"
            exit 0
        fi
    fi

    # Run checks
    run_checks

    # Publish to crates.io first (faster, and PyPI can depend on it)
    publish_to_crates

    # Publish to PyPI
    publish_to_pypi

    # Success message
    echo ""
    print_success "SUCCESS: Publication complete!"

    if [ "$TEST_MODE" = false ]; then
        echo ""
        echo "Users can now install with:"
        if [ "$CRATES_ONLY" = false ]; then
            echo "  pip install evlib"
        fi
        if [ "$PYPI_ONLY" = false ]; then
            echo "  cargo add evlib"
        fi
        echo ""
        echo "Package links:"
        if [ "$CRATES_ONLY" = false ]; then
            echo "  PyPI: https://pypi.org/project/evlib/"
        fi
        if [ "$PYPI_ONLY" = false ]; then
            echo "  crates.io: https://crates.io/crates/evlib"
        fi
    fi
}

# Check if script is being run from correct directory
if [ ! -f "Cargo.toml" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Run main function
main "$@"
