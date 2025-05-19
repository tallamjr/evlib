#!/bin/bash
set -e

# Function to display error messages and exit
error_exit() {
    echo "ERROR: $1" >&2
    exit 1
}

# Activate the virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Using Python from .venv virtual environment"
    source .venv/bin/activate
fi

# Get Python executable path
PYTHON_PATH=$(which python 2>/dev/null) || error_exit "Python not found in PATH. Ensure Python is installed and in your PATH."
echo "Python path: $PYTHON_PATH"

# Get the real Python executable path (resolve symlinks)
REAL_PYTHON_PATH=$(realpath $PYTHON_PATH 2>/dev/null) || error_exit "Could not resolve real path of Python executable."
echo "Real Python path: $REAL_PYTHON_PATH"

# Get Python version info
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

# Find the Python lib directory by looking at the parent directory of the binary
PYTHON_PREFIX=$(dirname $(dirname $REAL_PYTHON_PATH))
PYTHON_LIB_DIR="$PYTHON_PREFIX/lib"
echo "Python lib dir: $PYTHON_LIB_DIR"

# Verify the lib directory exists
if [ ! -d "$PYTHON_LIB_DIR" ]; then
    echo "Warning: Python lib directory ($PYTHON_LIB_DIR) not found."
    # Try to find lib using Python's sysconfig
    PYTHON_LIB_DIR_ALT=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')" 2>/dev/null)
    if [ -n "$PYTHON_LIB_DIR_ALT" ] && [ -d "$PYTHON_LIB_DIR_ALT" ]; then
        echo "Found alternative Python lib dir: $PYTHON_LIB_DIR_ALT"
        PYTHON_LIB_DIR="$PYTHON_LIB_DIR_ALT"
    else
        # Last resort - look for dylib files in common locations
        echo "Searching for Python libraries in common locations..."
        for dir in /usr/lib /usr/local/lib /opt/homebrew/lib "$HOME/.pyenv/versions"/*/*/lib; do
            if [ -d "$dir" ] && ls "$dir"/libpython*.dylib >/dev/null 2>&1; then
                PYTHON_LIB_DIR="$dir"
                echo "Found Python libraries in: $PYTHON_LIB_DIR"
                break
            fi
        done
    fi
fi

# Check if Python library exists
if [ -f "$PYTHON_LIB_DIR/libpython$PYTHON_VERSION.dylib" ]; then
    PYTHON_LIBRARY="libpython$PYTHON_VERSION.dylib"
    echo "Found Python library: $PYTHON_LIBRARY"
elif [ -f "$PYTHON_LIB_DIR/libpython${PYTHON_VERSION}m.dylib" ]; then
    PYTHON_LIBRARY="libpython${PYTHON_VERSION}m.dylib"
    echo "Found Python library: $PYTHON_LIBRARY"
else
    echo "Searching for Python library in $PYTHON_LIB_DIR..."
    PYTHON_LIBRARY_PATH=$(find $PYTHON_LIB_DIR -name "libpython*.dylib" 2>/dev/null | head -1)
    if [ -n "$PYTHON_LIBRARY_PATH" ]; then
        PYTHON_LIBRARY=$(basename $PYTHON_LIBRARY_PATH)
        echo "Found Python library: $PYTHON_LIBRARY"
    else
        # Check for Linux .so files
        PYTHON_LIBRARY_PATH=$(find $PYTHON_LIB_DIR -name "libpython*.so*" 2>/dev/null | head -1)
        if [ -n "$PYTHON_LIBRARY_PATH" ]; then
            PYTHON_LIBRARY=$(basename $PYTHON_LIBRARY_PATH)
            echo "Found Python library: $PYTHON_LIBRARY"
        else
            echo "Warning: Could not find Python library in $PYTHON_LIB_DIR"
            echo "This might cause linking errors during the build process."
        fi
    fi
fi

# Get Python include directory
PYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null)
if [ -z "$PYTHON_INCLUDE_DIR" ] || [ ! -d "$PYTHON_INCLUDE_DIR" ]; then
    echo "Warning: Python include directory not found or invalid: $PYTHON_INCLUDE_DIR"
    # Try alternate method to find include dir
    PYTHON_INCLUDE_DIR=$(python -c "import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())" 2>/dev/null)
    if [ -n "$PYTHON_INCLUDE_DIR" ] && [ -d "$PYTHON_INCLUDE_DIR" ]; then
        echo "Found Python include dir using alternate method: $PYTHON_INCLUDE_DIR"
    else
        echo "Warning: Could not find valid Python include directory."
        echo "This might cause compilation errors during the build process."
    fi
else
    echo "Python include dir: $PYTHON_INCLUDE_DIR"
fi

# Create a temporary .cargo/config.toml file specific to this build
mkdir -p .cargo
cat > .cargo/config.toml << EOF
[build]
rustflags = [
  "-L", "$PYTHON_LIB_DIR",
  "-l", "python$PYTHON_VERSION"
]

[env]
PYTHON_SYS_EXECUTABLE = { value = "$REAL_PYTHON_PATH", force = true }
EOF

# Display notice about the temporary config file
echo "Created temporary .cargo/config.toml for this build only."
echo "This file will be removed after the build completes."

# Process build arguments
BUILD_TYPE="--release"
CARGO_ARGS=()

for arg in "$@"; do
  case $arg in
    --debug)
      BUILD_TYPE=""
      ;;
    *)
      CARGO_ARGS+=("$arg")
      ;;
  esac
done

# Build the project with the correct environment variables
echo "Building with cargo... (${BUILD_TYPE:-debug mode})"
PYTHONPATH="$PYTHONPATH:$(pwd)" \
DYLD_LIBRARY_PATH="$PYTHON_LIB_DIR:$DYLD_LIBRARY_PATH" \
PYTHON_SYS_EXECUTABLE="$PYTHON_PATH" \
LIBRARY_PATH="$PYTHON_LIB_DIR:$LIBRARY_PATH" \
LDFLAGS="-L$PYTHON_LIB_DIR" \
cargo build $BUILD_TYPE --features "python" "${CARGO_ARGS[@]}"
BUILD_RESULT=$?

# Remove the temporary .cargo/config.toml file
rm .cargo/config.toml

if [ $BUILD_RESULT -eq 0 ]; then
    echo "============================================="
    echo "Build completed successfully! 🎉"
    if [ -n "$BUILD_TYPE" ]; then
        echo "The built library can be found in: $(pwd)/target/release/"
    else
        echo "The built library can be found in: $(pwd)/target/debug/"
    fi
    echo "============================================="
else
    echo "============================================="
    echo "❌ Build failed with error code: $BUILD_RESULT"
    echo "Please check the error messages above for details."
    echo "============================================="
    exit $BUILD_RESULT
fi
