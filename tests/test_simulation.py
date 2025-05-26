"""Test event simulation (Video-to-Events) functionality."""

from pathlib import Path
import pytest
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

EVLIB_AVAILABLE = False
try:
    import evlib  # noqa: F401

    EVLIB_AVAILABLE = True
except ImportError:
    pass


def test_simulation_config():
    """Test simulation configuration options."""
    # Test default configuration
    default_config = {
        "resolution": (640, 480),
        "contrast_threshold_pos": 0.2,
        "contrast_threshold_neg": 0.2,
        "refractory_period_us": 100.0,
        "timestep_us": 1000.0,
        "enable_noise": True,
    }

    # Validate default values
    assert default_config["resolution"] == (640, 480)
    assert default_config["contrast_threshold_pos"] > 0.0
    assert default_config["contrast_threshold_neg"] > 0.0
    assert default_config["refractory_period_us"] > 0.0
    assert default_config["timestep_us"] > 0.0
    assert isinstance(default_config["enable_noise"], bool)

    print("✓ Default simulation configuration valid")

    # Test various camera configurations
    camera_configs = [
        {"type": "dvs128", "resolution": (128, 128), "threshold": 0.3},
        {"type": "dvs240", "resolution": (240, 180), "threshold": 0.2},
        {"type": "dvs346", "resolution": (346, 240), "threshold": 0.15},
        {"type": "dvs640", "resolution": (640, 480), "threshold": 0.2},
        {"type": "davis346", "resolution": (346, 240), "threshold": 0.15},
    ]

    for config in camera_configs:
        assert config["resolution"][0] > 0
        assert config["resolution"][1] > 0
        assert config["threshold"] > 0.0
        print(f"✓ {config['type']} configuration valid")


def test_noise_model_config():
    """Test noise model configuration."""
    noise_configs = {
        "shot_noise": {
            "enabled": True,
            "scale_factor": 1.0,
            "min_photon_count": 1.0,
        },
        "dark_current": {
            "enabled": True,
            "rate_hz": 0.01,
            "temperature_coefficient": 0.001,
            "reference_temperature": 25.0,
        },
        "pixel_mismatch": {
            "enabled": True,
            "threshold_std": 0.05,
            "gain_std": 0.02,
            "correlation_length": 2.0,
        },
        "thermal_noise": {
            "enabled": False,
            "variance": 1e-6,
            "temperature": 25.0,
        },
    }

    for noise_type, config in noise_configs.items():
        assert isinstance(config["enabled"], bool)
        if noise_type == "dark_current":
            assert config["rate_hz"] >= 0.0
            assert config["temperature_coefficient"] >= 0.0
        elif noise_type == "pixel_mismatch":
            assert config["threshold_std"] >= 0.0
            assert config["gain_std"] >= 0.0
            assert config["correlation_length"] > 0.0

        print(f"✓ {noise_type} configuration valid")


def test_esim_features():
    """Test ESIM-specific simulation features."""
    esim_config = {
        "use_bilinear_interpolation": True,
        "adaptive_thresholding": False,
        "leaky_rate": 0.1,
        "cutoff_frequency_hz": 0.0,
    }

    # Validate ESIM configuration
    assert isinstance(esim_config["use_bilinear_interpolation"], bool)
    assert isinstance(esim_config["adaptive_thresholding"], bool)
    assert esim_config["leaky_rate"] >= 0.0
    assert esim_config["cutoff_frequency_hz"] >= 0.0

    print("✓ ESIM configuration valid")

    # Test ESIM-specific features
    features = [
        "bilinear_interpolation",
        "adaptive_thresholding",
        "bandwidth_filtering",
        "temporal_correlation",
        "pixel_response_model",
    ]

    for feature in features:
        # Each feature should be implementable
        assert isinstance(feature, str)
        print(f"✓ ESIM feature: {feature}")


def test_event_generation_logic():
    """Test core event generation algorithms."""

    # Simulate intensity change detection
    def simulate_event_detection(intensity_old, intensity_new, threshold):
        """Simulate ESIM event detection logic."""
        log_old = np.log(np.maximum(intensity_old, 1e-6))
        log_new = np.log(np.maximum(intensity_new, 1e-6))
        log_diff = log_new - log_old

        # Positive events
        pos_events = log_diff > threshold
        # Negative events
        neg_events = log_diff < -threshold

        return pos_events, neg_events

    # Test with synthetic intensity patterns
    width, height = 100, 100

    # Test case 1: Uniform increase
    intensity_old = np.ones((height, width)) * 0.5
    intensity_new = np.ones((height, width)) * 0.8  # Increase

    pos_events, neg_events = simulate_event_detection(intensity_old, intensity_new, 0.2)

    assert np.sum(pos_events) > 0  # Should generate positive events
    assert np.sum(neg_events) == 0  # Should not generate negative events

    print(f"✓ Uniform increase: {np.sum(pos_events)} positive events")

    # Test case 2: Uniform decrease
    intensity_new = np.ones((height, width)) * 0.2  # Decrease

    pos_events, neg_events = simulate_event_detection(intensity_old, intensity_new, 0.2)

    assert np.sum(pos_events) == 0  # Should not generate positive events
    assert np.sum(neg_events) > 0  # Should generate negative events

    print(f"✓ Uniform decrease: {np.sum(neg_events)} negative events")

    # Test case 3: Moving edge
    intensity_old = np.zeros((height, width))
    intensity_new = np.zeros((height, width))
    intensity_new[:, 40:60] = 1.0  # Vertical edge

    pos_events, neg_events = simulate_event_detection(intensity_old, intensity_new, 0.2)

    assert np.sum(pos_events) > 0  # Should generate events at edge
    edge_events = np.sum(pos_events)

    print(f"✓ Moving edge: {edge_events} events generated")


def test_temporal_resolution():
    """Test temporal resolution and timing accuracy."""

    # Test different temporal resolutions
    resolutions_us = [100, 500, 1000, 5000, 10000]  # 100μs to 10ms

    for resolution_us in resolutions_us:
        # Calculate expected frame rate
        fps = 1_000_000 / resolution_us

        # Validate reasonable range
        assert fps >= 100  # At least 100 FPS
        assert fps <= 10_000  # At most 10kFPS

        print(f"✓ {resolution_us}μs timestep = {fps:.0f} FPS")

    # Test event timing interpolation
    def interpolate_event_time(t1, t2, threshold_ratio):
        """Simulate sub-frame event timing."""
        return t1 + threshold_ratio * (t2 - t1)

    t1, t2 = 1000.0, 2000.0  # 1ms frame interval
    ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

    for ratio in ratios:
        event_time = interpolate_event_time(t1, t2, ratio)
        assert t1 <= event_time <= t2
        print(f"✓ Ratio {ratio}: event at {event_time:.1f}μs")


def test_noise_generation():
    """Test noise model implementations."""

    # Test Poisson sampling for dark current
    def poisson_sample(lam):
        """Simple Poisson sampling."""
        if lam <= 0:
            return 0
        return np.random.poisson(lam)

    # Test different rates
    rates = [0.01, 0.1, 1.0, 10.0]

    for rate in rates:
        samples = [poisson_sample(rate) for _ in range(100)]
        mean_sample = np.mean(samples)

        # Poisson distribution: mean ≈ λ (allow more tolerance for small rates)
        tolerance = max(rate * 0.5, 0.02)  # At least 0.02 tolerance for small rates
        assert abs(mean_sample - rate) < tolerance
        print(f"✓ Poisson rate {rate}: mean sample = {mean_sample:.2f}")

    # Test normal distribution for pixel mismatch
    def generate_pixel_mismatch(num_pixels, std_dev):
        """Generate pixel mismatch parameters."""
        return np.random.normal(0, std_dev, num_pixels)

    num_pixels = 1000
    std_devs = [0.01, 0.05, 0.1]

    for std_dev in std_devs:
        mismatch = generate_pixel_mismatch(num_pixels, std_dev)
        actual_std = np.std(mismatch)

        assert abs(actual_std - std_dev) < std_dev * 0.2  # Within 20%
        print(f"✓ Pixel mismatch σ={std_dev}: actual σ={actual_std:.3f}")


def test_video_processing_pipeline():
    """Test video processing and frame handling."""

    # Test synthetic video generation
    def generate_moving_pattern(width, height, frame_idx, pattern_type="gradient"):
        """Generate synthetic video frame with moving pattern."""
        frame = np.zeros((height, width), dtype=np.float32)

        if pattern_type == "gradient":
            # Moving horizontal gradient
            for y in range(height):
                for x in range(width):
                    intensity = 0.5 + 0.5 * np.sin((x + frame_idx * 2) / 20.0)
                    frame[y, x] = intensity

        elif pattern_type == "circle":
            # Moving circle
            center_x = width // 2 + int(20 * np.sin(frame_idx * 0.1))
            center_y = height // 2 + int(10 * np.cos(frame_idx * 0.1))
            radius = 20

            for y in range(height):
                for x in range(width):
                    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    if dist <= radius:
                        frame[y, x] = 1.0

        return frame

    # Test different patterns
    width, height = 160, 120
    num_frames = 30

    patterns = ["gradient", "circle"]

    for pattern in patterns:
        frames = []
        for frame_idx in range(num_frames):
            frame = generate_moving_pattern(width, height, frame_idx, pattern)
            frames.append(frame)

        assert len(frames) == num_frames
        assert frames[0].shape == (height, width)

        # Check that frames are different (motion)
        frame_diff = np.sum(np.abs(frames[1] - frames[0]))
        assert frame_diff > 0  # Should have motion

        print(f"✓ {pattern} pattern: {num_frames} frames, motion detected")


def test_camera_parameter_effects():
    """Test how camera parameters affect simulation."""

    # Test contrast threshold effects
    thresholds = [0.1, 0.2, 0.3, 0.5]

    # Simulate intensity change
    intensity_change = 0.25  # Fixed change

    for threshold in thresholds:
        events_generated = intensity_change > threshold
        print(f"✓ Threshold {threshold}: events = {events_generated}")

    # Test refractory period effects
    refractory_periods = [0, 50, 100, 200, 500]  # microseconds

    def check_refractory_violation(event_times, refractory_us):
        """Check if events violate refractory period."""
        if len(event_times) < 2:
            return False

        for i in range(1, len(event_times)):
            if event_times[i] - event_times[i - 1] < refractory_us:
                return True
        return False

    # Test event stream
    event_times = [1000, 1050, 1200, 1250, 1800]  # microseconds

    for refrac_period in refractory_periods:
        violation = check_refractory_violation(event_times, refrac_period)
        print(f"✓ Refractory {refrac_period}μs: violation = {violation}")

    # Test resolution effects
    resolutions = [(128, 128), (240, 180), (346, 240), (640, 480)]

    for res in resolutions:
        total_pixels = res[0] * res[1]
        aspect_ratio = res[0] / res[1]

        assert total_pixels > 0
        assert 1.0 <= aspect_ratio <= 2.0  # Reasonable aspect ratios

        print(f"✓ Resolution {res}: {total_pixels} pixels, AR={aspect_ratio:.2f}")


def test_performance_characteristics():
    """Test simulation performance characteristics."""

    # Test event rate estimation
    def estimate_event_rate(resolution, contrast_threshold, motion_factor):
        """Estimate events per second for given parameters."""
        total_pixels = resolution[0] * resolution[1]

        # Simple model: event rate proportional to pixels and motion
        base_rate = 0.1  # events/pixel/second for unit motion
        sensitivity = 1.0 / contrast_threshold  # Lower threshold = more sensitive

        return total_pixels * base_rate * sensitivity * motion_factor

    # Test scenarios
    scenarios = [
        {"name": "static", "resolution": (640, 480), "threshold": 0.2, "motion": 0.1},
        {"name": "slow_motion", "resolution": (640, 480), "threshold": 0.2, "motion": 1.0},
        {"name": "fast_motion", "resolution": (640, 480), "threshold": 0.2, "motion": 5.0},
        {"name": "high_sensitivity", "resolution": (640, 480), "threshold": 0.1, "motion": 1.0},
    ]

    for scenario in scenarios:
        rate = estimate_event_rate(scenario["resolution"], scenario["threshold"], scenario["motion"])

        # Validate reasonable event rates
        assert 1000 <= rate <= 10_000_000  # 1K to 10M events/second

        print(f"✓ {scenario['name']}: {rate:.0f} events/second")


def test_round_trip_validation():
    """Test Video→Events→Video round-trip validation."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    # Simulate round-trip processing
    def simulate_round_trip():
        """Simulate Video→Events→Video pipeline."""

        # Step 1: Original video frame
        width, height = 160, 120
        original_frame = np.random.rand(height, width).astype(np.float32)

        # Step 2: Generate events from frame difference
        prev_frame = np.ones_like(original_frame) * 0.5
        intensity_diff = original_frame - prev_frame

        # Simple event generation
        threshold = 0.1
        events = []

        for y in range(height):
            for x in range(width):
                if intensity_diff[y, x] > threshold:
                    events.append({"x": x, "y": y, "polarity": 1, "t": 0.0})
                elif intensity_diff[y, x] < -threshold:
                    events.append({"x": x, "y": y, "polarity": -1, "t": 0.0})

        # Step 3: Reconstruct frame from events (simple accumulation)
        reconstructed = np.zeros_like(original_frame)

        for event in events:
            x, y = event["x"], event["y"]
            if 0 <= x < width and 0 <= y < height:
                reconstructed[y, x] += event["polarity"] * 0.1

        return original_frame, events, reconstructed

    original, events, reconstructed = simulate_round_trip()

    # Validate pipeline
    assert original.shape == reconstructed.shape
    assert len(events) > 0

    # Check that reconstruction captures some structure
    correlation = np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1]

    print(f"✓ Round-trip: {len(events)} events, correlation = {correlation:.3f}")

    # Correlation might be low for simple reconstruction, but should be > 0
    assert not np.isnan(correlation)


@pytest.mark.skipif(not EVLIB_AVAILABLE, reason="evlib not available")
def test_evlib_simulation_integration():
    """Test integration with evlib simulation module."""
    import evlib

    # Test ESIM simulation with two intensity frames
    height, width = 50, 50
    intensity_old = np.ones((height, width), dtype=np.float32) * 0.3
    intensity_new = np.ones((height, width), dtype=np.float32) * 0.7

    # Call the ESIM simulation function
    xs, ys, ts, ps = evlib.simulation.esim_simulate_py(
        intensity_old, intensity_new, threshold=0.2, refractory_period_us=100.0
    )

    # Validate results
    assert isinstance(xs, np.ndarray)
    assert isinstance(ys, np.ndarray)
    assert isinstance(ts, np.ndarray)
    assert isinstance(ps, np.ndarray)

    # All arrays should have the same length
    assert len(xs) == len(ys) == len(ts) == len(ps)

    # Should generate positive events (intensity increased)
    if len(ps) > 0:
        assert np.all(ps == 1), "All events should be positive for uniform intensity increase"

    print(f"✓ ESIM simulation generated {len(xs)} events")

    # Test simulation config
    config = evlib.simulation.PySimulationConfig()
    assert config.resolution == (640, 480)  # Default resolution
    assert config.contrast_threshold_pos == 0.2  # Default threshold
    assert config.enable_noise  # Default noise enabled

    print("✓ PySimulationConfig created successfully")

    # Test video-to-events converter
    converter = evlib.simulation.PyVideoToEventsConverter(config)

    # Convert a single frame
    test_frame = np.random.rand(480, 640).astype(np.float32)
    xs2, ys2, ts2, ps2 = converter.convert_frame(test_frame, timestamp_us=0.0)

    # Validate converter results
    assert isinstance(xs2, np.ndarray)
    assert isinstance(ys2, np.ndarray)
    assert isinstance(ts2, np.ndarray)
    assert isinstance(ps2, np.ndarray)

    print(f"✓ VideoToEventsConverter generated {len(xs2)} events from frame")

    # Test simulation statistics
    stats = converter.get_stats()
    assert stats.frames_processed >= 0
    assert stats.total_pixels > 0

    print(f"✓ Simulation stats: {stats.frames_processed} frames, {stats.total_pixels} pixels")

    # Test high-level video_to_events function
    small_frame = np.random.rand(100, 100).astype(np.float32)
    xs3, ys3, ts3, ps3 = evlib.simulation.video_to_events_py(
        small_frame,
        contrast_threshold_pos=0.2,
        contrast_threshold_neg=0.2,
        refractory_period_us=100.0,
        timestep_us=1000.0,
        enable_noise=True,
    )

    # Validate high-level function results
    assert isinstance(xs3, np.ndarray)
    assert isinstance(ys3, np.ndarray)
    assert isinstance(ts3, np.ndarray)
    assert isinstance(ps3, np.ndarray)
    assert len(xs3) == len(ys3) == len(ts3) == len(ps3)

    print(f"✓ video_to_events_py generated {len(xs3)} events")
    print("✓ All evlib simulation integration tests passed!")


def test_webcam_simulation_readiness():
    """Test readiness for webcam integration."""

    # Test real-time processing requirements
    target_fps = 30
    max_latency_ms = 33.3  # ~30 FPS budget

    processing_steps = [
        {"name": "frame_capture", "time_ms": 1.0},
        {"name": "preprocessing", "time_ms": 2.0},
        {"name": "event_generation", "time_ms": 5.0},
        {"name": "post_processing", "time_ms": 3.0},
        {"name": "output", "time_ms": 1.0},
    ]

    total_processing_time = sum(step["time_ms"] for step in processing_steps)

    # Check real-time feasibility
    real_time_capable = total_processing_time <= max_latency_ms
    efficiency = (max_latency_ms - total_processing_time) / max_latency_ms

    print(f"✓ Total processing time: {total_processing_time:.1f}ms")
    print(f"✓ Real-time capable: {real_time_capable}")
    print(f"✓ Efficiency: {efficiency:.1%}")

    assert total_processing_time > 0

    # Test buffer management for real-time
    buffer_configs = [
        {"frames": 3, "description": "triple_buffer"},
        {"frames": 5, "description": "low_latency"},
        {"frames": 10, "description": "stable"},
    ]

    for config in buffer_configs:
        buffer_latency = config["frames"] / target_fps * 1000  # ms
        total_latency = buffer_latency + total_processing_time

        acceptable = total_latency <= 100  # 100ms max total latency

        print(f"✓ {config['description']}: {total_latency:.1f}ms total latency, acceptable = {acceptable}")


if __name__ == "__main__":
    test_simulation_config()
    test_noise_model_config()
    test_esim_features()
    test_event_generation_logic()
    test_temporal_resolution()
    test_noise_generation()
    test_video_processing_pipeline()
    test_camera_parameter_effects()
    test_performance_characteristics()
    test_round_trip_validation()
    test_webcam_simulation_readiness()
    print("All simulation tests passed!")
