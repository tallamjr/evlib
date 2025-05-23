"""Test E2VID Candle architectures (UNet and FireNet)"""

import numpy as np
import pytest
import evlib


class TestE2VidArchitectures:
    """Test the new Candle-based E2VID architectures"""

    def generate_test_events(self, n_events=1000, width=256, height=256):
        """Generate synthetic events for testing"""
        xs = np.random.randint(0, width, n_events, dtype=np.int64)
        ys = np.random.randint(0, height, n_events, dtype=np.int64)
        ts = np.sort(np.random.uniform(0, 1.0, n_events))
        ps = np.random.choice([-1, 1], n_events).astype(np.int64)
        return xs, ys, ts, ps

    def test_unet_architecture(self):
        """Test E2VID with UNet architecture"""
        xs, ys, ts, ps = self.generate_test_events()
        height, width = 256, 256

        # Process with UNet
        frame = evlib.processing.events_to_video_advanced(
            xs, ys, ts, ps, height, width, num_bins=5, model_type="unet"
        )

        assert frame.shape == (height, width, 1)
        assert frame.dtype == np.float32
        assert np.all(frame >= 0.0) and np.all(frame <= 1.0)

    def test_firenet_architecture(self):
        """Test E2VID with FireNet architecture (lightweight)"""
        xs, ys, ts, ps = self.generate_test_events()
        height, width = 256, 256

        # Process with FireNet
        frame = evlib.processing.events_to_video_advanced(
            xs, ys, ts, ps, height, width, num_bins=5, model_type="firenet"
        )

        assert frame.shape == (height, width, 1)
        assert frame.dtype == np.float32
        assert np.all(frame >= 0.0) and np.all(frame <= 1.0)

    def test_simple_accumulation(self):
        """Test simple accumulation mode"""
        xs, ys, ts, ps = self.generate_test_events()
        height, width = 256, 256

        # Process with simple accumulation
        frame = evlib.processing.events_to_video_advanced(
            xs, ys, ts, ps, height, width, num_bins=5, model_type="simple"
        )

        assert frame.shape == (height, width, 1)
        assert frame.dtype == np.float32
        assert np.all(frame >= 0.0) and np.all(frame <= 1.0)

    def test_default_model(self):
        """Test default model selection (should be UNet)"""
        xs, ys, ts, ps = self.generate_test_events()
        height, width = 256, 256

        # Process with default (no model_type specified)
        frame = evlib.processing.events_to_video_advanced(xs, ys, ts, ps, height, width, num_bins=5)

        assert frame.shape == (height, width, 1)
        assert frame.dtype == np.float32

    def test_varying_bin_counts(self):
        """Test different voxel grid bin counts"""
        xs, ys, ts, ps = self.generate_test_events()
        height, width = 256, 256

        for num_bins in [3, 5, 10]:
            frame = evlib.processing.events_to_video_advanced(
                xs, ys, ts, ps, height, width, num_bins=num_bins, model_type="unet"
            )
            assert frame.shape == (height, width, 1)

    @pytest.mark.benchmark(group="architectures")
    def test_unet_performance(self, benchmark):
        """Benchmark UNet architecture"""
        xs, ys, ts, ps = self.generate_test_events(n_events=10000)
        height, width = 256, 256

        def run_unet():
            return evlib.processing.events_to_video_advanced(
                xs, ys, ts, ps, height, width, num_bins=5, model_type="unet"
            )

        frame = benchmark(run_unet)
        assert frame.shape == (height, width, 1)

    @pytest.mark.benchmark(group="architectures")
    def test_firenet_performance(self, benchmark):
        """Benchmark FireNet architecture (should be faster)"""
        xs, ys, ts, ps = self.generate_test_events(n_events=10000)
        height, width = 256, 256

        def run_firenet():
            return evlib.processing.events_to_video_advanced(
                xs, ys, ts, ps, height, width, num_bins=5, model_type="firenet"
            )

        frame = benchmark(run_firenet)
        assert frame.shape == (height, width, 1)

    def test_edge_cases(self):
        """Test edge cases for architectures"""
        height, width = 256, 256

        # Empty events
        xs = np.array([], dtype=np.int64)
        ys = np.array([], dtype=np.int64)
        ts = np.array([], dtype=np.float64)
        ps = np.array([], dtype=np.int64)

        for model_type in ["unet", "firenet", "simple"]:
            frame = evlib.processing.events_to_video_advanced(
                xs, ys, ts, ps, height, width, model_type=model_type
            )
            assert frame.shape == (height, width, 1)
            # With no events, output should be zeros (or uniform)
            assert np.allclose(frame, frame.flat[0])

    def test_different_resolutions(self):
        """Test various image resolutions"""
        resolutions = [(128, 128), (256, 256), (512, 512), (256, 512)]

        for height, width in resolutions:
            xs, ys, ts, ps = self.generate_test_events(n_events=1000, width=width, height=height)

            for model_type in ["unet", "firenet"]:
                frame = evlib.processing.events_to_video_advanced(
                    xs, ys, ts, ps, height, width, model_type=model_type
                )
                assert frame.shape == (height, width, 1)
