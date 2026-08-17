"""Generate the rpg_vid2e esim_torch reference digest for slider_depth.

Runs on a CUDA host with esim_torch installed (arg1: ~/vid2e-bench/.venv-torch).
It uses only the public API: esim_torch.ESIM(c_neg, c_pos, refractory_ns) and
.forward(log_image, timestamp_ns), fed one frame at a time. Nothing else from
rpg_vid2e is imported. Output is read by tests/test_vid2e_conformance.py.

Run on arg1:
  ~/vid2e-bench/.venv-torch/bin/python vid2e_reference.py \
      --images ~/vid2e-bench/slider_depth --out vid2e_slider_depth_ct0.2.json
Then scp the JSON to tests/conformance/reference/.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import esim_torch

LOG_EXPRESSION = "ln(I/255 + 1e-3) as float32"
BLOCK = 10


def load_frames(images_root):
    """Read images.txt (seconds, relative path) into a uint8 (T, H, W) stack and int64 ns."""
    frames, t_ns = [], []
    for line in (images_root / "images.txt").read_text().splitlines():
        secs, rel = line.split()
        frames.append(
            np.asarray(Image.open(images_root / rel).convert("L"), dtype=np.uint8)
        )
        t_ns.append(round(float(secs) * 1e9))
    return np.stack(frames), np.asarray(t_ns, dtype=np.int64)


def esim_torch_commit(repo):
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def run_esim(frames, t_ns, c_pos, c_neg, refractory_ns):
    """Feed frames one at a time on CUDA; return (x, y, t_ns, p) int64 arrays with p in {-1, 1}."""
    esim = esim_torch.ESIM(c_neg, c_pos, refractory_ns)
    log_frames = np.log(frames.astype(np.float32) / 255 + 1e-3).astype(np.float32)
    parts = []
    dtypes = {}
    raw_polarities = set()
    for log_frame, t in zip(log_frames, t_ns):
        out = esim.forward(
            torch.from_numpy(log_frame).cuda(),
            torch.tensor(int(t), dtype=torch.int64).cuda(),
        )
        if out is None:
            continue
        for key in ("x", "y", "t", "p"):
            dtypes[key] = str(out[key].dtype)
        p_raw = out["p"].cpu().numpy().astype(np.int64)
        raw_polarities.update(np.unique(p_raw).tolist())
        parts.append(
            np.stack(
                [
                    out["x"].cpu().numpy().astype(np.int64),
                    out["y"].cpu().numpy().astype(np.int64),
                    out["t"].cpu().numpy().astype(np.int64),
                    p_raw,
                ],
                axis=1,
            )
        )
    events = np.concatenate(parts, axis=0)
    # Normalise polarity to -1/1 whatever encoding esim_torch returns.
    if raw_polarities <= {0, 1}:
        events[:, 3] = np.where(events[:, 3] == 1, 1, -1)
    elif not raw_polarities <= {-1, 1}:
        raise ValueError(
            f"unexpected polarity values from esim_torch: {sorted(raw_polarities)}"
        )
    return events, dtypes, sorted(raw_polarities)


def build_digest(events, frames, t_ns, args):
    x, y, t, p = events.T
    height, width = frames.shape[1:]
    # np.histogram closes the last bin, so events at t_ns[-1] are counted once.
    per_frame = np.histogram(t, bins=t_ns)[0]
    per_pixel = np.zeros((height, width), dtype=np.int64)
    np.add.at(per_pixel, (y, x), 1)
    # 10x10 pixel block sums: coarse spatial digest that tolerates per-pixel tie flips.
    blocks = per_pixel.reshape(height // BLOCK, BLOCK, width // BLOCK, BLOCK).sum(
        axis=(1, 3)
    )
    order = np.lexsort((p, x, y, t))[:200]
    return {
        "source": "data/slider_depth/images",
        "frames": int(frames.shape[0]),
        "width": int(width),
        "height": int(height),
        "log": LOG_EXPRESSION,
        "c_pos": args.c_pos,
        "c_neg": args.c_neg,
        "refractory_ns": args.refractory_ns,
        "total_events": int(len(t)),
        "pos_events": int((p == 1).sum()),
        "per_frame_counts": per_frame.tolist(),
        # Recorded for provenance only; the test asserts block_counts (tie flips break exact pixels).
        "per_pixel_counts_sha256": hashlib.sha256(per_pixel.tobytes()).hexdigest(),
        "block_px": BLOCK,
        "block_counts": blocks.tolist(),
        "sample": events[order].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--images", type=Path, required=True, help="dir with images.txt and images/"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--repo", type=Path, default=Path.home() / "vid2e-bench" / "rpg_vid2e"
    )
    parser.add_argument("--c-pos", type=float, default=0.2)
    parser.add_argument("--c-neg", type=float, default=0.2)
    parser.add_argument("--refractory-ns", type=int, default=0)
    args = parser.parse_args()

    frames, t_ns = load_frames(args.images)
    events, dtypes, raw_polarities = run_esim(
        frames, t_ns, args.c_pos, args.c_neg, args.refractory_ns
    )
    digest = build_digest(events, frames, t_ns, args)
    digest["esim_torch_provenance"] = {
        "esim_torch_commit": esim_torch_commit(args.repo),
        "torch_version": torch.__version__,
        "output_dtypes": dtypes,
        "raw_polarity_values": raw_polarities,
        "generated_by": "tests/conformance/vid2e_reference.py",
    }
    args.out.write_text(json.dumps(digest, indent=1) + "\n")
    print(
        f"{args.out}: {digest['total_events']} events, {digest['pos_events']} positive; dtypes {dtypes}; raw polarity {raw_polarities}"
    )


if __name__ == "__main__":
    main()
