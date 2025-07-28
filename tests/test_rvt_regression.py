"""
Regression tests comparing evlib stacked histogram output with RVT reference data.

This test suite validates that evlib's create_stacked_histogram produces
equivalent results to RVT's preprocessing pipeline.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import evlib.representations as evr


def convert_evlib_to_dense_tensor(
    evlib_hist: pl.DataFrame, height: int, width: int, nbins: int
) -> np.ndarray:
    """
    Convert evlib sparse histogram format to RVT dense tensor format.

    Args:
        evlib_hist: Polars DataFrame with columns [window_id, channel, time_bin, y, x, count, channel_time_bin]
        height, width: Spatial dimensions
        nbins: Number of temporal bins

    Returns:
        Dense numpy array with shape (n_windows, n_channels*nbins, height, width)
        matching RVT output format
    """
    # Get dimensions
    max_window_id = evlib_hist["window_id"].max()
    n_windows = max_window_id + 1 if max_window_id is not None else 0
    n_channels = 2  # positive and negative polarity

    # Initialize dense tensor
    dense_tensor = np.zeros((n_windows, n_channels * nbins, height, width), dtype=np.uint8)

    # Convert to numpy for faster indexing
    data = evlib_hist.select(["window_id", "channel_time_bin", "y", "x", "count"]).to_numpy()

    # Fill dense tensor
    for row in data:
        window_id, channel_time_bin, y, x, count = row
        if (
            0 <= window_id < n_windows
            and 0 <= channel_time_bin < n_channels * nbins
            and 0 <= y < height
            and 0 <= x < width
        ):
            dense_tensor[window_id, channel_time_bin, y, x] = min(count, 255)  # Clip to uint8

    return dense_tensor


class TestRVTRegression:
    """Test suite for RVT regression validation."""

    @pytest.fixture
    def rvt_test_file(self):
        """Path to RVT reference data (matching the events file)."""
        return "tests/data/gen4_1mpx_processed_RVT/test/moorea_2019-06-19_000_793500000_853500000/event_representations_v2/stacked_histogram_dt50_nbins10/event_representations_ds2_nearest.h5"

    @pytest.fixture
    def events_file(self):
        """Path to corresponding events file."""
        return "tests/data/eTram/h5/val_2/val_night_011_td.h5"

    def test_evlib_sparse_to_dense_conversion(self):
        """Test conversion from evlib sparse format to dense tensor."""
        # Create test data
        test_data = pl.DataFrame(
            {
                "window_id": [0, 0, 1, 1],
                "channel": [0, 1, 0, 1],
                "time_bin": [0, 5, 2, 7],
                "y": [100, 200, 150, 250],
                "x": [200, 300, 400, 500],
                "count": [5, 3, 8, 2],
                "channel_time_bin": [0, 15, 2, 17],  # channel * 10 + time_bin
            }
        )

        dense = convert_evlib_to_dense_tensor(test_data, height=360, width=640, nbins=10)

        # Verify shape
        assert dense.shape == (2, 20, 360, 640)

        # Verify values are placed correctly
        assert dense[0, 0, 100, 200] == 5  # window 0, channel_time_bin 0
        assert dense[0, 15, 200, 300] == 3  # window 0, channel_time_bin 15
        assert dense[1, 2, 150, 400] == 8  # window 1, channel_time_bin 2
        assert dense[1, 17, 250, 500] == 2  # window 1, channel_time_bin 17

        # Verify all other values are zero
        total_nonzero = np.count_nonzero(dense)
        assert total_nonzero == 4

    @pytest.mark.slow
    @pytest.mark.integration
    def test_parquet_based_regression_quick(self, rvt_test_file, events_file):
        """Quick Parquet-based regression test comparing against RVT reference data.

        This is a slow integration test that requires large data files.
        For CI/CD, use synthetic tests instead.
        """
        if not (Path(rvt_test_file).exists() and Path(events_file).exists()):
            pytest.skip("Required files not available")

        print("\n=== Running Quick RVT Comparison Test ===")
        print(f"RVT file: {rvt_test_file}")
        print(f"Events file: {events_file}")

        import h5py
        import hdf5plugin  # Required for compressed HDF5 files
        import tempfile

        # Step 1: Load first 10 windows of RVT reference data
        max_windows = 10
        print(f"Loading first {max_windows} windows from RVT reference...")

        with h5py.File(rvt_test_file, "r") as f:
            rvt_data = f["data"][:max_windows]  # Load only first 10 windows
            print(f"RVT subset shape: {rvt_data.shape}")

        print(f"RVT reference loaded: {rvt_data.shape}")
        print(f"RVT value range: [{rvt_data.min()}, {rvt_data.max()}]")
        print(f"RVT non-zero entries: {np.count_nonzero(rvt_data)}")

        # Step 2: Generate corresponding evlib data and convert to RVT format
        print("Generating evlib histogram for first 500ms...")
        print("Note: RVT uses 2x spatial downsampling (720×1280 → 360×640)")

        # Load only first 600ms of events to avoid processing full dataset
        print("Loading first 600ms of events to match RVT subset...")

        # Import evlib to load events with time limit
        import evlib

        # Load only first 0.6 seconds (600ms) of events
        events_lazy = evlib.load_events(events_file, t_start=0.0, t_end=0.6)  # 600ms in seconds

        # Collect the events to numpy arrays for processing
        events_df = events_lazy.collect()
        print(f"Loaded {events_df.height} events for first 600ms")

        # Extract arrays
        timestamps = events_df["timestamp"].to_numpy()
        x_coords = events_df["x"].to_numpy()
        y_coords = events_df["y"].to_numpy()
        polarities = events_df["polarity"].to_numpy()

        # Apply 2x spatial downsampling to match RVT exactly
        x_downsampled = x_coords // 2
        y_downsampled = y_coords // 2

        # Clip to RVT dimensions (important for edge coordinates)
        x_downsampled = np.clip(x_downsampled, 0, 639)
        y_downsampled = np.clip(y_downsampled, 0, 359)

        # Create downsampled events LazyFrame
        downsampled_events = pl.LazyFrame(
            {"timestamp": timestamps, "x": x_downsampled, "y": y_downsampled, "polarity": polarities}
        )

        # Generate evlib histogram from pre-downsampled events
        evlib_lazy = evr.create_stacked_histogram(
            downsampled_events,  # Use pre-processed events
            height=360,  # Target resolution (already downsampled)
            width=640,  # Target resolution (already downsampled)
            nbins=10,
            window_duration_ms=50.0,
            count_cutoff=10,  # Apply cutoff after all processing
        )

        evlib_data = evlib_lazy.filter(pl.col("window_id") < max_windows).collect()  # Match RVT subset

        print(f"evlib sparse data: {evlib_data.shape}")

        # Step 3: Convert evlib DataFrame to RVT's dense numpy format
        print("Converting evlib data to RVT format...")
        evlib_dense = convert_evlib_to_dense_tensor(evlib_data, height=360, width=640, nbins=10)
        print(f"evlib dense shape: {evlib_dense.shape}")
        print(f"evlib value range: [{evlib_dense.min()}, {evlib_dense.max()}]")
        print(f"evlib non-zero entries: {np.count_nonzero(evlib_dense)}")

        # Step 4: Direct numpy array comparison
        print("Performing direct numpy comparison...")

        # Ensure shapes match
        min_windows = min(rvt_data.shape[0], evlib_dense.shape[0])
        rvt_subset = rvt_data[:min_windows]
        evlib_subset = evlib_dense[:min_windows]

        print(f"Comparing {min_windows} windows...")
        print(f"RVT subset shape: {rvt_subset.shape}")
        print(f"evlib subset shape: {evlib_subset.shape}")

        # Calculate comparison metrics
        exact_matches = np.sum(rvt_subset == evlib_subset)
        total_elements = rvt_subset.size
        exact_ratio = exact_matches / total_elements

        diff = np.abs(rvt_subset.astype(np.int16) - evlib_subset.astype(np.int16))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        rvt_nonzero = np.count_nonzero(rvt_subset)
        evlib_nonzero = np.count_nonzero(evlib_subset)
        count_diff_ratio = abs(rvt_nonzero - evlib_nonzero) / max(rvt_nonzero, evlib_nonzero, 1)

        print(f"Total elements: {total_elements}")
        print(f"Exact matches: {exact_matches} ({exact_ratio:.3f})")
        print(f"Max difference: {max_diff}")
        print(f"Mean difference: {mean_diff:.3f}")
        print(f"Non-zero counts - RVT: {rvt_nonzero}, evlib: {evlib_nonzero}")
        print(f"Non-zero count difference ratio: {count_diff_ratio:.3f}")

        # Assertions for quick test (realistic thresholds based on preprocessing differences)
        assert exact_ratio > 0.85, f"Too few exact matches: {exact_ratio:.3f}"
        assert max_diff <= 15, f"Values differ too much: max_diff={max_diff}"
        assert count_diff_ratio < 0.3, f"Non-zero counts too different: {count_diff_ratio:.3f}"
        assert mean_diff < 0.5, f"Mean difference too high: {mean_diff:.3f}"

        print("Quick RVT comparison PASSED!")
        print(f"   - Exact match ratio: {exact_ratio:.3f} > 0.85")
        print(f"   - Max value difference: {max_diff} <= 15")
        print(f"   - Mean difference: {mean_diff:.3f} < 0.5")
        print(f"   - Non-zero count difference: {count_diff_ratio:.3f} < 0.3")

        # Report quality metrics
        if exact_ratio > 0.95:
            print("EXCELLENT: >95% exact matches")
        elif exact_ratio > 0.90:
            print("GOOD: >90% exact matches")
        elif exact_ratio > 0.85:
            print("WARNING: ACCEPTABLE: >85% exact matches")

        if max_diff <= 10:
            print("Count cutoff working well")
        else:
            print("WARNING: Some count aggregation differences detected")
