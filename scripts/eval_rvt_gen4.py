"""Evaluate the committed gen4 RVT-tiny checkpoint on the gen4 val set and report mAP.

Eval-only (no training): reuses evlib's RVT detector + the ported Prophesee
evaluator. Run on a machine with the gen4 preprocessed data and a GPU.

Example (arg1):
    .venv/bin/python scripts/eval_rvt_gen4.py \
        --val-root /home/tarek/datasets/gen4_1mpx_processed_RVT/gen4/val \
        --device cuda
Add --limit N for a quick smoke run over the first N sequences.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from evlib.data import PreprocessedH5Source, RVT_REPR_DIR_NAME
from evlib.eval.rvt_eval import evaluate_rvt_gen4
from evlib.models.rvt import RVT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="0 = all sequences")
    args = parser.parse_args()

    seq_dirs = sorted(p for p in args.val_root.iterdir() if p.is_dir())
    if args.limit:
        seq_dirs = seq_dirs[: args.limit]
    if not seq_dirs:
        raise SystemExit(f"no sequence directories under {args.val_root}")
    print(f"evaluating {len(seq_dirs)} gen4 val sequences from {args.val_root}")

    sources = [
        PreprocessedH5Source(d, repr_name=RVT_REPR_DIR_NAME, downsample_by_2=True)
        for d in seq_dirs
    ]

    model = RVT(variant="tiny", num_classes=3, pretrained=True).eval()

    t0 = time.time()
    metrics = evaluate_rvt_gen4(
        model,
        sources,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        device=args.device,
    )
    dt = time.time() - t0

    print(f"\n=== gen4 RVT-tiny eval ({len(seq_dirs)} seqs, {dt:.1f}s) ===")
    if metrics is None:
        raise SystemExit("evaluator returned no metrics (no scored frames?)")
    for key in ("AP", "AP_50", "AP_75", "AP_S", "AP_M", "AP_L"):
        if key in metrics:
            print(f"{key:6s} = {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
