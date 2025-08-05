#!/bin/bash

# Script to add conditional compilation for tracing imports throughout the codebase

files=(
    "src/ev_formats/hdf5_reader.rs"
    "src/ev_formats/prophesee_ecf_codec.rs"
    "src/ev_formats/ecf_codec.rs"
    "src/ev_filtering/spatial.rs"
    "src/ev_filtering/polarity.rs"
    "src/ev_filtering/downsampling.rs"
    "src/ev_filtering/temporal.rs"
    "src/ev_filtering/hot_pixel.rs"
    "src/ev_filtering/denoise.rs"
    "src/ev_filtering/drop_pixel.rs"
    "src/ev_filtering/utils.rs"
    "src/ev_augmentation/drop_event.rs"
    "src/ev_augmentation/geometric_transforms.rs"
    "src/ev_augmentation/time_reversal.rs"
    "src/ev_augmentation/time_jitter.rs"
    "src/ev_augmentation/uniform_noise.rs"
    "src/ev_augmentation/time_skew.rs"
    "src/ev_augmentation/crop.rs"
    "src/ev_visualization/web_server.rs"
)

# Function to add conditional compilation to a file
add_conditional_tracing() {
    local file="$1"
    echo "Processing $file..."

    # Create a backup
    cp "$file" "$file.bak"

    # Replace tracing imports with conditional imports
    sed -i '' 's/^use tracing::#[cfg(feature = "tracing")]\
use tracing::/g' "$file"

    # Add fallback macros after the conditional import
    if grep -q "#\[cfg(feature = \"tracing\")\]" "$file" && grep -q "use tracing::" "$file"; then
        # Find the line number of the conditional import
        import_line=$(grep -n "#\[cfg(feature = \"tracing\")\]" "$file" | grep -A1 "use tracing::" | tail -1 | cut -d: -f1)

        # Add fallback macros after the import
        sed -i '' "${import_line}a\\
\\
#[cfg(not(feature = \"tracing\"))]\\
macro_rules! error {\\
    (\$(\$args:tt)*) => {\\
        eprintln!(\"[ERROR] {}\", format!(\$(\$args)*))\\
    };\\
}\\
\\
#[cfg(not(feature = \"tracing\"))]\\
macro_rules! warn {\\
    (\$(\$args:tt)*) => {\\
        eprintln!(\"[WARN] {}\", format!(\$(\$args)*))\\
    };\\
}\\
\\
#[cfg(not(feature = \"tracing\"))]\\
macro_rules! info {\\
    (\$(\$args:tt)*) => {};\\
}\\
\\
#[cfg(not(feature = \"tracing\"))]\\
macro_rules! debug {\\
    (\$(\$args:tt)*) => {};\\
}\\
\\
#[cfg(not(feature = \"tracing\"))]\\
macro_rules! trace {\\
    (\$(\$args:tt)*) => {};\\
}\\
\\
#[cfg(not(feature = \"tracing\"))]\\
macro_rules! instrument {\\
    (\$(\$args:tt)*) => {};\\
}
" "$file"
    fi
}

# Process each file
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        add_conditional_tracing "$file"
    else
        echo "File $file not found, skipping..."
    fi
done

echo "Done! All files have been processed."
echo "Backup files created with .bak extension"
