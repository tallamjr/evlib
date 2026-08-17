"""Video decode (OpenCV) feeding the Rust ESIM kernel. Output: canonical Polars event frame."""

from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import polars as pl

try:
    import cv2

    _opencv_available = True
except ImportError:
    _opencv_available = False
    cv2 = None

from .config import ESIMConfig, VideoConfig
from .esim import ESIMSimulator, _to_dataframe, simulate_frames

# Above this many frame bytes the whole clip is not stacked in memory.
_BATCH_BYTES_LIMIT = 2 * 1024**3


class VideoToEvents:
    """Decode a video file with OpenCV and simulate events with the ESIM kernel."""

    def __init__(self, esim_config: ESIMConfig, video_config: VideoConfig):
        if not _opencv_available:
            raise ImportError(
                "OpenCV is required for video processing. Install with: pip install opencv-python"
            )
        self.esim_config = esim_config
        self.video_config = video_config
        self._cap = None
        self._video_fps: Optional[float] = None
        self._total_frames: Optional[int] = None
        self._current_frame: int = 0
        self._start_frame: int = 0

    def process_video(self, video_path: Union[str, Path]) -> pl.DataFrame:
        """Simulate the whole clip; batch when it fits in memory, else chunked."""
        with self._opened(video_path):
            frame_bytes = self._frame_bytes()
            fits = 0 < self._total_frames * frame_bytes < _BATCH_BYTES_LIMIT
            if not fits:
                return _concat(list(self._stream_chunks(64)))
            frames, t_ns = self._decode_chunk(None)
            if not frames:
                return _empty_frame()
            return simulate_frames(np.stack(frames), np.asarray(t_ns), self.esim_config)

    def process_frames_streaming(
        self, video_path: Union[str, Path], chunk_frames: int = 64
    ) -> Iterator[pl.DataFrame]:
        """Yield one DataFrame per `chunk_frames` decoded frames; state persists across chunks."""
        with self._opened(video_path):
            yield from self._stream_chunks(chunk_frames)

    def get_video_info(self, video_path: Union[str, Path]) -> dict:
        """Source properties plus the effective decode settings."""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info = {
                "path": str(video_path),
                "fps": fps,
                "frame_count": frame_count,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration_seconds": frame_count / fps if fps > 0 else 0.0,
            }
        finally:
            cap.release()
        target_fps = self.video_config.fps or info["fps"]
        info["processing"] = {
            "target_width": self.video_config.width or info["width"],
            "target_height": self.video_config.height or info["height"],
            "target_fps": target_fps,
            "frame_skip": self.video_config.frame_skip,
            "grayscale": self.video_config.grayscale,
            "effective_fps": target_fps / (self.video_config.frame_skip + 1),
        }
        return info

    # Internals

    def _opened(self, video_path: Union[str, Path]):
        return _Capture(self, Path(video_path))

    def _setup_video_properties(self) -> None:
        original_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._video_fps = self.video_config.fps or original_fps
        if self._video_fps <= 0:
            raise ValueError("video reports no frame rate; set VideoConfig.fps")
        self._start_frame = 0
        if self.video_config.start_time is not None:
            self._start_frame = int(self.video_config.start_time * original_fps)
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)
        self._current_frame = self._start_frame
        if self.video_config.end_time is not None:
            end_frame = int(self.video_config.end_time * original_fps)
            self._total_frames = min(self._total_frames, end_frame)

    def _kept_frame_count(self) -> int:
        """Frames the decode loop keeps after start_time, end_time and frame_skip."""
        step = self.video_config.frame_skip + 1
        end = max(self._total_frames, self._start_frame)
        return -(-end // step) - -(-self._start_frame // step)

    def _frame_bytes(self) -> int:
        width = self.video_config.width or int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = self.video_config.height or int(
            self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        return width * height

    def _next_frame(self) -> Optional[Tuple[np.ndarray, int]]:
        """Decode the next kept frame; None at end of clip or end_time."""
        step = self.video_config.frame_skip + 1
        while True:
            if (
                self.video_config.end_time is not None
                and self._current_frame >= self._total_frames
            ):
                return None
            ok, frame = self._cap.read()
            if not ok:
                return None
            index = self._current_frame
            self._current_frame += 1
            if index % step != 0:
                continue
            t_ns = int(round(index / self._video_fps * 1e9))
            return self._preprocess_frame(frame), t_ns

    def _decode_chunk(
        self, chunk_frames: Optional[int]
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Decode up to `chunk_frames` kept frames; None means until the clip ends."""
        frames: List[np.ndarray] = []
        t_ns: List[int] = []
        while chunk_frames is None or len(frames) < chunk_frames:
            item = self._next_frame()
            if item is None:
                break
            frames.append(item[0])
            t_ns.append(item[1])
        return frames, t_ns

    def _stream_chunks(self, chunk_frames: int) -> Iterator[pl.DataFrame]:
        if chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        simulator: Optional[ESIMSimulator] = None
        while True:
            frames, t_ns = self._decode_chunk(chunk_frames)
            if not frames:
                return
            if simulator is None:
                height, width = frames[0].shape
                simulator = ESIMSimulator(self.esim_config, width=width, height=height)
            parts = [simulator.step_ns(frame, t) for frame, t in zip(frames, t_ns)]
            x = np.concatenate([p[0] for p in parts])
            y = np.concatenate([p[1] for p in parts])
            t = np.concatenate([p[2] for p in parts])
            pol = np.concatenate([p[3] for p in parts])
            order = np.argsort(t, kind="stable")
            yield _to_dataframe(x[order], y[order], t[order], pol[order])

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.video_config.width is not None or self.video_config.height is not None:
            width = self.video_config.width or frame.shape[1]
            height = self.video_config.height or frame.shape[0]
            if (frame.shape[1], frame.shape[0]) != (width, height):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(frame, dtype=np.uint8)


class _Capture:
    """Context manager: open the capture on the processor, release on exit."""

    def __init__(self, owner: VideoToEvents, video_path: Path):
        self.owner = owner
        self.video_path = video_path

    def __enter__(self):
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        self.owner._cap = cap
        try:
            self.owner._setup_video_properties()
        except Exception:
            cap.release()
            self.owner._cap = None
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        self.owner._cap.release()
        self.owner._cap = None
        return False


def _empty_frame() -> pl.DataFrame:
    return _to_dataframe(
        np.empty(0, np.int16),
        np.empty(0, np.int16),
        np.empty(0, np.int64),
        np.empty(0, np.int8),
    )


def _concat(chunks: List[pl.DataFrame]) -> pl.DataFrame:
    if not chunks:
        return _empty_frame()
    return pl.concat(chunks)


def estimate_event_count(
    video_path: Union[str, Path],
    esim_config: Optional[ESIMConfig] = None,
    video_config: Optional[VideoConfig] = None,
    sample_frames: int = 100,
) -> dict:
    """Simulate the first `sample_frames` kept frames and scale to the kept-frame count."""
    esim_config = esim_config or ESIMConfig()
    video_config = video_config or VideoConfig()
    processor = VideoToEvents(esim_config, video_config)
    with processor._opened(video_path):
        frames, t_ns = processor._decode_chunk(sample_frames)
        kept_frames = processor._kept_frame_count()
    if not frames:
        return {
            "estimated_total_events": 0,
            "estimated": True,
            "sample_frames": 0,
            "sample_events": 0,
            "events_per_frame": 0.0,
            "total_frames": kept_frames,
        }
    df = simulate_frames(np.stack(frames), np.asarray(t_ns), esim_config)
    events_per_frame = df.height / len(frames)
    return {
        "estimated_total_events": int(round(events_per_frame * kept_frames)),
        "estimated": len(frames) < kept_frames,
        "sample_frames": len(frames),
        "sample_events": df.height,
        "events_per_frame": events_per_frame,
        "total_frames": kept_frames,
    }
