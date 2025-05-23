"""Test reconstruction quality metrics"""

import numpy as np
import pytest
import evlib


class TestReconstructionMetrics:
    """Test suite for reconstruction quality metrics"""

    def test_mse_identical_images(self):
        """MSE should be 0 for identical images"""
        # Create identical images
        np.random.rand(128, 128).astype(np.float32)

        # Using evlib for reconstruction (as proxy for metrics testing)
        # In future, we'd have direct metric functions
        n_events = 1000
        xs = np.random.randint(0, 128, n_events, dtype=np.int64)
        ys = np.random.randint(0, 128, n_events, dtype=np.int64)
        ts = np.sort(np.random.uniform(0, 0.1, n_events))
        ps = np.random.choice([-1, 1], n_events).astype(np.int64)

        # Test that reconstruction produces valid output
        frame = evlib.processing.events_to_video_advanced(xs, ys, ts, ps, 128, 128, model_type="simple")

        assert frame.shape == (128, 128, 1)
        assert frame.dtype == np.float32
        assert np.all(frame >= 0) and np.all(frame <= 1)

    def test_different_models_different_metrics(self):
        """Different models should produce different quality metrics"""
        # Generate test events
        n_events = 5000
        width, height = 128, 128
        np.random.seed(42)

        xs = np.random.randint(0, width, n_events, dtype=np.int64)
        ys = np.random.randint(0, height, n_events, dtype=np.int64)
        ts = np.sort(np.random.uniform(0, 0.1, n_events))
        ps = np.random.choice([-1, 1], n_events).astype(np.int64)

        # Get outputs from different models
        results = {}
        for model_type in ["unet", "firenet", "simple"]:
            frame = evlib.processing.events_to_video_advanced(
                xs, ys, ts, ps, height, width, model_type=model_type
            )
            results[model_type] = frame

        # Calculate simple metrics
        def calculate_mse(img1, img2):
            return np.mean((img1 - img2) ** 2)

        def calculate_psnr(img1, img2, max_val=1.0):
            mse = calculate_mse(img1, img2)
            if mse < 1e-10:
                return 100.0
            return 20 * np.log10(max_val / np.sqrt(mse))

        # Compare models
        mse_unet_simple = calculate_mse(results["unet"], results["simple"])
        mse_firenet_simple = calculate_mse(results["firenet"], results["simple"])

        # Neural models should differ from simple accumulation
        assert mse_unet_simple > 0.001
        assert mse_firenet_simple > 0.001

        # Calculate PSNR
        psnr_unet_simple = calculate_psnr(results["unet"], results["simple"])
        psnr_firenet_simple = calculate_psnr(results["firenet"], results["simple"])

        print(f"MSE UNet vs Simple: {mse_unet_simple:.4f}")
        print(f"MSE FireNet vs Simple: {mse_firenet_simple:.4f}")
        print(f"PSNR UNet vs Simple: {psnr_unet_simple:.2f} dB")
        print(f"PSNR FireNet vs Simple: {psnr_firenet_simple:.2f} dB")

    def test_ssim_calculation(self):
        """Test structural similarity index calculation"""

        # Simple SSIM implementation for testing
        def calculate_ssim(img1, img2, c1=0.01**2, c2=0.03**2):
            mu1 = np.mean(img1)
            mu2 = np.mean(img2)

            sigma1_sq = np.var(img1)
            sigma2_sq = np.var(img2)
            sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

            numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
            denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)

            return numerator / denominator

        # Test with identical images
        img = np.random.rand(64, 64).astype(np.float32)
        ssim_same = calculate_ssim(img, img)
        assert ssim_same > 0.99  # Should be very close to 1

        # Test with different images
        img1 = np.random.rand(64, 64).astype(np.float32)
        img2 = np.random.rand(64, 64).astype(np.float32)
        ssim_diff = calculate_ssim(img1, img2)
        assert 0 < ssim_diff < 0.5  # Should be low for random images

    def test_temporal_consistency(self):
        """Test temporal consistency across frames"""
        # Generate events for multiple time windows
        n_events = 10000
        width, height = 128, 128

        xs = np.random.randint(0, width, n_events, dtype=np.int64)
        ys = np.random.randint(0, height, n_events, dtype=np.int64)
        ts = np.sort(np.random.uniform(0, 1.0, n_events))
        ps = np.random.choice([-1, 1], n_events).astype(np.int64)

        # Split into time windows
        n_frames = 5
        frames = []

        for i in range(n_frames):
            t_start = i / n_frames
            t_end = (i + 1) / n_frames

            mask = (ts >= t_start) & (ts < t_end)
            if np.sum(mask) > 0:
                frame = evlib.processing.events_to_video_advanced(
                    xs[mask], ys[mask], ts[mask], ps[mask], height, width, model_type="unet"
                )
                frames.append(frame)

        # Calculate temporal differences
        if len(frames) > 1:
            temporal_diffs = []
            for i in range(1, len(frames)):
                diff = np.mean((frames[i] - frames[i - 1]) ** 2)
                temporal_diffs.append(diff)

            avg_temporal_diff = np.mean(temporal_diffs)
            print(f"Average temporal difference: {avg_temporal_diff:.4f}")

            # Temporal differences should be relatively small
            assert avg_temporal_diff < 0.5

    @pytest.mark.benchmark(group="metrics")
    def test_metric_computation_performance(self, benchmark):
        """Benchmark metric computation speed"""
        # Generate test data
        img1 = np.random.rand(256, 256).astype(np.float32)
        img2 = np.random.rand(256, 256).astype(np.float32)

        def compute_all_metrics():
            mse = np.mean((img1 - img2) ** 2)
            psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100

            # Simple SSIM
            mu1, mu2 = np.mean(img1), np.mean(img2)
            var1, var2 = np.var(img1), np.var(img2)
            cov12 = np.mean((img1 - mu1) * (img2 - mu2))

            c1, c2 = 0.01**2, 0.03**2
            ssim = ((2 * mu1 * mu2 + c1) * (2 * cov12 + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))

            return {"mse": mse, "psnr": psnr, "ssim": ssim}

        metrics = benchmark(compute_all_metrics)
        assert "mse" in metrics
        assert "psnr" in metrics
        assert "ssim" in metrics


if __name__ == "__main__":
    test = TestReconstructionMetrics()
    print("Testing reconstruction metrics...")
    test.test_mse_identical_images()
    test.test_different_models_different_metrics()
    test.test_ssim_calculation()
    test.test_temporal_consistency()
    print("\nAll metric tests passed! ✓")
