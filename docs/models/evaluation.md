# Evaluation

`evlib.eval` scores a trained `evlib.models.RVT` detector against a validation split using the same Prophesee-style mAP metric (COCO-style AP averaged over IoU thresholds, plus size-bucketed AP_S/AP_M/AP_L) that the original RVT paper reports. The evaluator is ported near-verbatim from RVT's own reference implementation, so scores are directly comparable to published RVT numbers.

## `evaluate_rvt_gen4`: the eval loop

`evlib.eval.rvt_eval.evaluate_rvt_gen4(model, sources, ...)` wires together a streaming loader over the val sequences, the RVT forward pass, YOLOX postprocessing, and the Prophesee evaluator into one function:

```python notest
from evlib.data import PreprocessedH5Source, RVT_REPR_DIR_NAME
from evlib.eval.rvt_eval import evaluate_rvt_gen4
from evlib.models.rvt import RVT
from evlib.models.rvt_backbone import partition_size_from_hw

sources = [
    PreprocessedH5Source(seq_dir, repr_name=RVT_REPR_DIR_NAME, downsample_by_2=True)
    for seq_dir in val_root.iterdir()
]

padded_hw = (384, 640)  # gen4 ds2 backbone-multiple padded HW
model = RVT(
    variant="tiny",
    num_classes=3,
    pretrained=True,
    partition_size=partition_size_from_hw(padded_hw),
).eval()

metrics = evaluate_rvt_gen4(
    model, sources, sequence_length=10, batch_size=8, padded_hw=padded_hw, device="cuda"
)
print(metrics["AP"], metrics["AP_50"], metrics["AP_75"])
```

What the function does, matching the reference validation step:

- streams the val sequences with a batched, order-preserving sampler, so every GT-bearing frame is scored, not just the last frame of each chunk;
- resets recurrent state on `is_first_sample` and carries it (detached) across streaming steps;
- pads the input bottom/right to the backbone's required stride multiple (`padded_hw`) so detected box coordinates keep their top-left origin and stay aligned with the unpadded ground truth;
- only scores frames that actually carry ground truth, mirroring the reference's `len(labels) > 0` gate;
- runs the YOLOX `postprocess` (`conf_thre`/`nms_thre`/`num_classes`, defaulting to the gen4 eval settings `0.001`/`0.45`/`3`) on the selected frames, converts both predictions and GT to Prophesee's `BBOX_DTYPE`, and calls the buffered evaluator's `evaluate_buffer` with the padded height/width.

It takes an already-constructed model (weights loaded by the caller) so a small random model can drive the same loop in tests; `sources` must be `ReprSource`s that also expose `read_window_gt` (which `PreprocessedH5Source` does). It returns the Prophesee metrics dict directly, or `None` if no frames were scored.

## Converting to Prophesee's box format

`evlib.eval.convert` bridges evlib's two native box representations to the `BBOX_DTYPE` structured array the Prophesee evaluator consumes (top-left `x, y, w, h` plus `class_id`, `class_confidence`, `track_id`, `t`):

- `preds_to_prophesee(yolox_pred, frame_t)`: converts one frame's `[N, 7]` YOLOX `postprocess` output (`x1, y1, x2, y2, obj_conf, class_conf, class_pred`) to `BBOX_DTYPE` rows. Corner coordinates become top-left `x, y` plus `w, h`; `class_confidence` is the class confidence alone (not `obj_conf * class_conf`), matching the RVT reference's own `to_prophesee` conversion.
- `gt_rows_to_prophesee(structured_rows, frame_t=None)`: converts on-disk `labels.npz` ground-truth rows (already top-left, already carrying `t`) to `BBOX_DTYPE` rows field-by-field. `track_id` defaults to 0 when the on-disk schema omits it (as the tracked `mini_seq` fixture does). If `frame_t` is given, it must agree with the rows' own on-disk timestamp or a `ValueError` is raised.

## The Prophesee evaluator

`evlib.eval.prophesee` is the ported scoring backend itself:

- `PropheseeEvaluator(dataset, downsample_by_2)`: a buffered evaluator. Call `add_predictions`/`add_labels` per frame (each a list of `BBOX_DTYPE` arrays), then `evaluate_buffer(img_height, img_width)` once the val pass is done to get the metrics dict; `reset_buffer()` clears it for the next epoch. `dataset` must be `"gen1"` or `"gen4"`.
- `BBOX_DTYPE`: the structured dtype described above.
- `GEN1_CLASSES` (`car`, `pedestrian`) and `GEN4_CLASSES` (`pedestrian`, `two-wheeler`, `car`): the per-camera class tuples used for scoring.
- `evaluate_list`, `evaluate_detection`, `filter_boxes`: the lower-level functions `evaluate_buffer` calls internally, exposed for callers that already have full prediction/label lists in memory rather than a streaming buffer.

`evaluate_detection` (and therefore the evaluator as a whole) needs `pycocotools`, pulled in by the `eval` extra: `pip install evlib[eval]`.

## Running it from the command line

`scripts/eval_rvt_gen4.py` is the reference CLI wrapper: it loads a gen4 val split from a directory of preprocessed sequence folders, constructs an `RVT` tiny model with the committed pretrained checkpoint and the gen4-derived MaxViT partition size, runs `evaluate_rvt_gen4`, and prints the headline metrics.

```bash
python scripts/eval_rvt_gen4.py \
    --val-root /path/to/gen4_1mpx_processed_RVT/gen4/val \
    --device cuda
```

Flags: `--val-root` (required, one subdirectory per sequence), `--device` (default `cuda`), `--batch-size` (default 8), `--sequence-length` (default 10), and `--limit N` to score only the first `N` sequences for a quick smoke run. Output is one line per metric key present in the returned dict (`AP`, `AP_50`, `AP_75`, `AP_S`, `AP_M`, `AP_L`).

This script needs the full preprocessed gen4 dataset and, in practice, a GPU; the dataset is not part of evlib's tracked fixtures, so running it is a local-only workflow rather than something the docs build or CI can exercise directly.

## Where to go next

- [Models](index.md): how `RVT` is constructed and how `detect` produces the predictions this page scores.
- [Datasets & Training Data](datasets.md): `PreprocessedH5Source` and the preprocessed sequence layout `evaluate_rvt_gen4` reads from.
