#!/usr/bin/env python3

import re
import os
import sys

# Files to update
files_to_update = [
    "src/ev_filtering/downsampling.rs",
    "src/ev_filtering/temporal.rs",
    "src/ev_filtering/hot_pixel.rs",
    "src/ev_filtering/denoise.rs",
    "src/ev_filtering/drop_pixel.rs",
    "src/ev_filtering/utils.rs",
    "src/ev_augmentation/crop.rs",
    "src/ev_augmentation/drop_event.rs",
    "src/ev_augmentation/geometric_transforms.rs",
    "src/ev_augmentation/time_jitter.rs",
    "src/ev_augmentation/time_reversal.rs",
    "src/ev_augmentation/time_skew.rs",
    "src/ev_augmentation/uniform_noise.rs",
    "src/ev_visualization/web_server.rs",
]


def update_file(filename):
    """Update a file to use conditional tracing compilation."""
    if not os.path.exists(filename):
        print(f"File {filename} not found, skipping...")
        return False

    with open(filename, "r") as f:
        content = f.read()

    # Check if already updated
    if '#[cfg(feature = "tracing")]' in content and "use tracing" in content:
        print(f"File {filename} already updated, skipping...")
        return False

    # Find tracing import line
    tracing_pattern = r"^use tracing::\{([^}]+)\};"
    match = re.search(tracing_pattern, content, re.MULTILINE)

    if not match:
        print(f"No tracing import found in {filename}")
        return False

    tracing_imports = match.group(1)
    old_import = match.group(0)

    # Parse the imports (used for validation)
    _imports = [imp.strip() for imp in tracing_imports.split(",")]

    # Create conditional import and fallback macros
    new_import = f"""#[cfg(feature = "tracing")]
use tracing::{{{tracing_imports}}};

#[cfg(not(feature = "tracing"))]
macro_rules! debug {{
    ($($args:tt)*) => {{}};
}}

#[cfg(not(feature = "tracing"))]
macro_rules! info {{
    ($($args:tt)*) => {{}};
}}

#[cfg(not(feature = "tracing"))]
macro_rules! warn {{
    ($($args:tt)*) => {{
        eprintln!("[WARN] {{}}", format!($($args)*))
    }};
}}

#[cfg(not(feature = "tracing"))]
macro_rules! trace {{
    ($($args:tt)*) => {{}};
}}

#[cfg(not(feature = "tracing"))]
macro_rules! error {{
    ($($args:tt)*) => {{
        eprintln!("[ERROR] {{}}", format!($($args)*))
    }};
}}

#[cfg(not(feature = "tracing"))]
macro_rules! instrument {{
    ($($args:tt)*) => {{}};
}}"""

    # Replace the import
    new_content = content.replace(old_import, new_import)

    # Write back to file
    with open(filename, "w") as f:
        f.write(new_content)

    print(f"Updated {filename}")
    return True


def main():
    updated_count = 0
    for filename in files_to_update:
        if update_file(filename):
            updated_count += 1

    print(f"\nUpdated {updated_count} files total")


if __name__ == "__main__":
    main()
