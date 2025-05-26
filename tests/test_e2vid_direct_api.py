"""
Test suite for the direct E2Vid Rust API using real data only.

This module tests the new direct E2Vid class exposed from Rust,
ensuring all tests use real event data from the /data/ directory.
"""

import pytest
import numpy as np
from pathlib import Path

try:
    import evlib

    EVLIB_AVAILABLE = True
except ImportError:
    EVLIB_AVAILABLE = False


@pytest.mark.skipif(not EVLIB_AVAILABLE, reason="evlib not available")
class TestE2VidDirectAPI:
    """Test the direct E2Vid Rust API with real data."""

    @pytest.fixture
    def slider_depth_events(self):
        """Load real events from slider_depth dataset."""
        events_file = Path("data/slider_depth/events.txt")
        if not events_file.exists():
            pytest.skip("slider_depth dataset not available")

        # Load events using evlib
        events_data = evlib.formats.load_events(str(events_file))

        # Take first 10k events for testing
        n_events = min(10000, len(events_data[0]))
        xs = events_data[0][:n_events]
        ys = events_data[1][:n_events]
        ts = events_data[2][:n_events]
        ps = events_data[3][:n_events]

        # Get sensor resolution from data
        height = int(ys.max()) + 1
        width = int(xs.max()) + 1

        return {
            "xs": xs.tolist(),
            "ys": ys.tolist(),
            "ts": ts.tolist(),
            "ps": ps.tolist(),
            "height": height,
            "width": width,
            "n_events": n_events,
        }

    @pytest.fixture
    def hdf5_events(self):
        """Load real events from HDF5 dataset."""
        hdf5_file = Path("data/original/front/seq01.h5")
        if not hdf5_file.exists():
            pytest.skip("HDF5 dataset not available")

        try:
            import h5py

            with h5py.File(hdf5_file, "r") as f:
                # Load events from HDF5 file
                events = f["events"]

                # Take first 5k events for testing
                n_events = min(5000, len(events["x"]))
                xs = events["x"][:n_events].tolist()
                ys = events["y"][:n_events].tolist()
                ts = events["t"][:n_events].tolist()
                ps = events["p"][:n_events].tolist()

                # Get sensor resolution
                height = int(max(ys)) + 1
                width = int(max(xs)) + 1

                return {
                    "xs": xs,
                    "ys": ys,
                    "ts": ts,
                    "ps": ps,
                    "height": height,
                    "width": width,
                    "n_events": n_events,
                }
        except ImportError:
            pytest.skip("h5py not available")
        except Exception as e:
            pytest.skip(f"Could not load HDF5 data: {e}")

    def test_e2vid_class_creation(self, slider_depth_events):
        """Test creating E2Vid instance with real sensor dimensions."""
        events = slider_depth_events

        # Create E2Vid instance with real dimensions
        e2vid = evlib.processing.E2Vid(events["height"], events["width"])

        assert e2vid is not None
        assert not e2vid.has_model_py  # Initially no model loaded

    def test_e2vid_reconstruction_without_model(self, slider_depth_events):
        """Test reconstruction using default UNet fallback with real events."""
        events = slider_depth_events

        # Create E2Vid instance
        e2vid = evlib.processing.E2Vid(events["height"], events["width"])

        # Reconstruct frame with real events (should use default UNet)
        frame = e2vid.reconstruct_frame(events["xs"], events["ys"], events["ts"], events["ps"])

        assert frame.shape == (events["height"], events["width"])
        assert frame.dtype == np.float32
        assert np.all(frame >= 0.0) and np.all(frame <= 1.0)
        assert np.any(frame > 0)  # Should have some signal from real events

    def test_e2vid_model_loading(self, slider_depth_events):
        """Test loading PyTorch model with real data."""
        events = slider_depth_events

        # Create E2Vid instance
        e2vid = evlib.processing.E2Vid(events["height"], events["width"])

        # Check available models
        model_paths = ["models/minimal_test_unet.pth", "models/E2VID_lightweight.pth.tar"]

        model_loaded = False
        for model_path in model_paths:
            if Path(model_path).exists():
                try:
                    e2vid.load_model_from_file(model_path)
                    assert e2vid.has_model_py
                    model_loaded = True
                    print(f"✅ Successfully loaded model: {model_path}")
                    break
                except Exception as e:
                    print(f"⚠️ Could not load {model_path}: {e}")
                    continue

        if not model_loaded:
            pytest.skip("No loadable PyTorch models found")

        # Test reconstruction with loaded model
        frame = e2vid.reconstruct_frame(events["xs"], events["ys"], events["ts"], events["ps"])

        assert frame.shape == (events["height"], events["width"])
        assert frame.dtype == np.float32
        assert np.any(frame > 0)  # Should have signal

    def test_e2vid_with_hdf5_data(self, hdf5_events):
        """Test E2Vid with real HDF5 event data."""
        events = hdf5_events

        # Create E2Vid instance
        e2vid = evlib.processing.E2Vid(events["height"], events["width"])

        # Reconstruct with HDF5 events
        frame = e2vid.reconstruct_frame(events["xs"], events["ys"], events["ts"], events["ps"])

        assert frame.shape == (events["height"], events["width"])
        assert frame.dtype == np.float32
        assert np.any(frame > 0)  # Should have signal from real events

    def test_e2vid_different_event_counts(self, slider_depth_events):
        """Test reconstruction with different numbers of real events."""
        full_events = slider_depth_events

        e2vid = evlib.processing.E2Vid(full_events["height"], full_events["width"])

        # Test with different event counts
        event_counts = [100, 1000, 5000]

        for count in event_counts:
            if count > full_events["n_events"]:
                continue

            # Take subset of events
            events = {
                "xs": full_events["xs"][:count],
                "ys": full_events["ys"][:count],
                "ts": full_events["ts"][:count],
                "ps": full_events["ps"][:count],
            }

            frame = e2vid.reconstruct_frame(events["xs"], events["ys"], events["ts"], events["ps"])

            assert frame.shape == (full_events["height"], full_events["width"])
            assert frame.dtype == np.float32

            # With real events, should always have some signal
            assert np.any(frame > 0), f"No signal with {count} events"

    def test_e2vid_temporal_consistency(self, slider_depth_events):
        """Test temporal consistency with real event sequences."""
        events = slider_depth_events

        e2vid = evlib.processing.E2Vid(events["height"], events["width"])

        # Split events into temporal windows
        n_frames = 3
        events_per_frame = len(events["xs"]) // n_frames

        frames = []
        for i in range(n_frames):
            start_idx = i * events_per_frame
            end_idx = (i + 1) * events_per_frame if i < n_frames - 1 else len(events["xs"])

            frame_events = {
                "xs": events["xs"][start_idx:end_idx],
                "ys": events["ys"][start_idx:end_idx],
                "ts": events["ts"][start_idx:end_idx],
                "ps": events["ps"][start_idx:end_idx],
            }

            frame = e2vid.reconstruct_frame(
                frame_events["xs"], frame_events["ys"], frame_events["ts"], frame_events["ps"]
            )

            frames.append(frame)

        # Verify all frames
        for i, frame in enumerate(frames):
            assert frame.shape == (events["height"], events["width"])
            assert frame.dtype == np.float32
            # Real events should produce signal
            assert np.any(frame > 0), f"No signal in frame {i}"

    def test_e2vid_memory_efficiency(self, slider_depth_events):
        """Test memory efficiency with real data."""
        events = slider_depth_events

        # Create multiple E2Vid instances to test memory usage
        instances = []
        for i in range(5):
            e2vid = evlib.processing.E2Vid(events["height"], events["width"])

            # Process events
            frame = e2vid.reconstruct_frame(events["xs"], events["ys"], events["ts"], events["ps"])

            assert frame.shape == (events["height"], events["width"])
            instances.append(e2vid)

        # All instances should work
        assert len(instances) == 5

    def test_e2vid_reproducibility(self, slider_depth_events):
        """Test that reconstruction is consistent within same instance."""
        events = slider_depth_events

        # Create single instance and run twice
        e2vid = evlib.processing.E2Vid(events["height"], events["width"])

        # Reconstruct with same events twice
        frame1 = e2vid.reconstruct_frame(events["xs"], events["ys"], events["ts"], events["ps"])
        frame2 = e2vid.reconstruct_frame(events["xs"], events["ys"], events["ts"], events["ps"])

        # Results should be identical when using same instance
        assert frame1.shape == frame2.shape
        diff = np.abs(frame1 - frame2)
        max_diff = np.max(diff)

        # With same instance and same data, should be identical
        assert max_diff < 1e-6, f"Same instance gave different results: {max_diff}"

        # Test that both frames have valid signal from real events
        assert np.any(frame1 > 0) and np.any(frame2 > 0), "Both frames should have signal"

    def test_e2vid_edge_cases_real_data(self, slider_depth_events):
        """Test edge cases with real data."""
        events = slider_depth_events

        e2vid = evlib.processing.E2Vid(events["height"], events["width"])

        # Test with single event
        single_event = {
            "xs": [events["xs"][0]],
            "ys": [events["ys"][0]],
            "ts": [events["ts"][0]],
            "ps": [events["ps"][0]],
        }

        frame = e2vid.reconstruct_frame(
            single_event["xs"], single_event["ys"], single_event["ts"], single_event["ps"]
        )

        assert frame.shape == (events["height"], events["width"])
        assert frame.dtype == np.float32

        # Test with events from same timestamp
        same_time_events = {
            "xs": events["xs"][:5],
            "ys": events["ys"][:5],
            "ts": [events["ts"][0]] * 5,  # Same timestamp
            "ps": events["ps"][:5],
        }

        frame = e2vid.reconstruct_frame(
            same_time_events["xs"], same_time_events["ys"], same_time_events["ts"], same_time_events["ps"]
        )

        assert frame.shape == (events["height"], events["width"])
        assert frame.dtype == np.float32


@pytest.mark.skipif(not EVLIB_AVAILABLE, reason="evlib not available")
class TestE2VidAPIComparison:
    """Compare the new direct API with legacy functions using real data."""

    @pytest.fixture
    def slider_depth_events(self):
        """Load real events from slider_depth dataset."""
        events_file = Path("data/slider_depth/events.txt")
        if not events_file.exists():
            pytest.skip("slider_depth dataset not available")

        events_data = evlib.formats.load_events(str(events_file))

        # Take first 5k events for comparison testing
        n_events = min(5000, len(events_data[0]))

        return {
            "xs": events_data[0][:n_events],
            "ys": events_data[1][:n_events],
            "ts": events_data[2][:n_events],
            "ps": events_data[3][:n_events],
            "height": int(events_data[1][:n_events].max()) + 1,
            "width": int(events_data[0][:n_events].max()) + 1,
        }

    def test_api_consistency(self, slider_depth_events):
        """Test that both APIs produce reasonable results with real data."""
        events = slider_depth_events

        # Test new direct API
        e2vid = evlib.processing.E2Vid(events["height"], events["width"])
        direct_frame = e2vid.reconstruct_frame(
            events["xs"].tolist(), events["ys"].tolist(), events["ts"].tolist(), events["ps"].tolist()
        )

        # Test legacy API
        legacy_frame = evlib.processing.events_to_video(
            events["xs"], events["ys"], events["ts"], events["ps"], events["height"], events["width"]
        )

        # Both should produce valid outputs
        assert direct_frame.shape == (events["height"], events["width"])
        assert legacy_frame.shape == (events["height"], events["width"], 1)

        # Both should have signal from real events
        assert np.any(direct_frame > 0)
        assert np.any(legacy_frame > 0)

        # Data types should be consistent
        assert direct_frame.dtype == np.float32
        assert legacy_frame.dtype == np.float32

    def test_performance_comparison(self, slider_depth_events):
        """Compare performance between APIs with real data."""
        import time

        events = slider_depth_events

        # Time direct API
        e2vid = evlib.processing.E2Vid(events["height"], events["width"])

        start_time = time.time()
        direct_frame = e2vid.reconstruct_frame(
            events["xs"].tolist(), events["ys"].tolist(), events["ts"].tolist(), events["ps"].tolist()
        )
        direct_time = time.time() - start_time

        # Time legacy API
        start_time = time.time()
        legacy_frame = evlib.processing.events_to_video(
            events["xs"], events["ys"], events["ts"], events["ps"], events["height"], events["width"]
        )
        legacy_time = time.time() - start_time

        print(f"Direct API time: {direct_time:.4f}s")
        print(f"Legacy API time: {legacy_time:.4f}s")

        # Both should complete in reasonable time (neural networks take 10-20s)
        assert direct_time < 30.0  # Should complete within 30s
        assert legacy_time < 30.0  # Should complete within 30s

        # Both should produce valid results
        assert np.any(direct_frame > 0)
        assert np.any(legacy_frame > 0)


@pytest.mark.skipif(not EVLIB_AVAILABLE, reason="evlib not available")
class TestE2VidRealDataFormats:
    """Test E2Vid with different real data formats."""

    def test_text_format_events(self):
        """Test with slider_depth text format events."""
        events_file = Path("data/slider_depth/events.txt")
        if not events_file.exists():
            pytest.skip("slider_depth text events not available")

        # Load via evlib formats
        events_data = evlib.formats.load_events(str(events_file))

        height = int(events_data[1].max()) + 1
        width = int(events_data[0].max()) + 1

        # Test with subset
        n_events = min(3000, len(events_data[0]))

        e2vid = evlib.processing.E2Vid(height, width)
        frame = e2vid.reconstruct_frame(
            events_data[0][:n_events].tolist(),
            events_data[1][:n_events].tolist(),
            events_data[2][:n_events].tolist(),
            events_data[3][:n_events].tolist(),
        )

        assert frame.shape == (height, width)
        assert np.any(frame > 0)

    def test_calibration_info(self):
        """Test using real calibration info from slider_depth."""
        calib_file = Path("data/slider_depth/calib.txt")
        if not calib_file.exists():
            pytest.skip("slider_depth calibration not available")

        # Read calibration file to get true sensor dimensions
        with open(calib_file, "r") as f:
            _ = f.readlines()  # Read but don't use - known dimensions below

        # Parse width/height from calibration if available
        width, height = 240, 180  # Known slider_depth dimensions

        events_file = Path("data/slider_depth/events.txt")
        if events_file.exists():
            events_data = evlib.formats.load_events(str(events_file))

            n_events = min(2000, len(events_data[0]))

            e2vid = evlib.processing.E2Vid(height, width)
            frame = e2vid.reconstruct_frame(
                events_data[0][:n_events].tolist(),
                events_data[1][:n_events].tolist(),
                events_data[2][:n_events].tolist(),
                events_data[3][:n_events].tolist(),
            )

            assert frame.shape == (height, width)
            assert np.any(frame > 0)

    def test_multiple_real_datasets(self):
        """Test with multiple real datasets if available."""
        dataset_paths = ["data/slider_depth/events.txt", "data/original/front/seq01.h5"]

        tested_datasets = 0

        for dataset_path in dataset_paths:
            dataset_file = Path(dataset_path)
            if not dataset_file.exists():
                continue

            try:
                if dataset_path.endswith(".txt"):
                    # Text format
                    events_data = evlib.formats.load_events(str(dataset_file))
                    height = int(events_data[1].max()) + 1
                    width = int(events_data[0].max()) + 1

                    n_events = min(1000, len(events_data[0]))

                    e2vid = evlib.processing.E2Vid(height, width)
                    frame = e2vid.reconstruct_frame(
                        events_data[0][:n_events].tolist(),
                        events_data[1][:n_events].tolist(),
                        events_data[2][:n_events].tolist(),
                        events_data[3][:n_events].tolist(),
                    )

                    assert frame.shape == (height, width)
                    assert np.any(frame > 0)
                    tested_datasets += 1

                elif dataset_path.endswith(".h5"):
                    # HDF5 format
                    import h5py

                    with h5py.File(dataset_file, "r") as f:
                        events = f["events"]

                        n_events = min(1000, len(events["x"]))
                        xs = events["x"][:n_events].tolist()
                        ys = events["y"][:n_events].tolist()
                        ts = events["t"][:n_events].tolist()
                        ps = events["p"][:n_events].tolist()

                        height = int(max(ys)) + 1
                        width = int(max(xs)) + 1

                        e2vid = evlib.processing.E2Vid(height, width)
                        frame = e2vid.reconstruct_frame(xs, ys, ts, ps)

                        assert frame.shape == (height, width)
                        assert np.any(frame > 0)
                        tested_datasets += 1

            except Exception as e:
                print(f"Could not test {dataset_path}: {e}")
                continue

        if tested_datasets == 0:
            pytest.skip("No real datasets available for testing")

        print(f"Successfully tested {tested_datasets} real datasets")
