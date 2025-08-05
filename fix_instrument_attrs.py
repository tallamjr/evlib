#!/usr/bin/env python3

import re
import os
import sys

# Files to update
files_to_update = [
    "src/ev_filtering/spatial.rs",
    "src/ev_filtering/polarity.rs",
    "src/ev_filtering/downsampling.rs",
    "src/ev_filtering/temporal.rs",
    "src/ev_filtering/hot_pixel.rs",
    "src/ev_filtering/denoise.rs",
    "src/ev_filtering/drop_pixel.rs",
    "src/ev_filtering/utils.rs",
    "src/ev_augmentation/drop_event.rs",
    "src/ev_augmentation/geometric_transforms.rs",
    "src/ev_augmentation/time_reversal.rs",
    "src/ev_augmentation/time_jitter.rs",
    "src/ev_augmentation/uniform_noise.rs",
    "src/ev_augmentation/time_skew.rs",
    "src/ev_augmentation/spatial_jitter.rs",
    "src/ev_augmentation/crop.rs",
]


def update_file(filename):
    """Update a file to add conditional compilation for instrument attributes."""
    if not os.path.exists(filename):
        print(f"File {filename} not found, skipping...")
        return False

    with open(filename, "r") as f:
        content = f.read()

    # Check if already updated
    if '#[cfg_attr(feature = "tracing", instrument' in content:
        print(f"File {filename} already updated, skipping...")
        return False

    # Replace all #[instrument(...)] with #[cfg_attr(feature = "tracing", instrument(...))]
    # This pattern matches the instrument attribute and captures its contents
    instrument_pattern = r"#\[instrument\(([^]]*)\)\]"

    def replace_instrument(match):
        attrs = match.group(1)
        return f'#[cfg_attr(feature = "tracing", instrument({attrs}))]'

    new_content = re.sub(instrument_pattern, replace_instrument, content)

    if new_content != content:
        # Write back to file
        with open(filename, "w") as f:
            f.write(new_content)

        print(f"Updated {filename}")
        return True
    else:
        print(f"No instrument attributes found in {filename}")
        return False


def main():
    updated_count = 0
    for filename in files_to_update:
        if update_file(filename):
            updated_count += 1

    print(f"\nUpdated {updated_count} files total")


if __name__ == "__main__":
    main()
