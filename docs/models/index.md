# Deep Learning Models

`evlib.models` ships two PyTorch model families for event camera data: E2VID (event-to-video reconstruction) and RVT (event-based object detection). Both are implemented as plain PyTorch modules with pretrained-weight loading and GPU inference, and both share a common base interface for turning raw events into model input.

## Installing PyTorch support

`evlib.models` requires PyTorch, which is not part of the default install:

```bash
pip install evlib[torch]
# or, an identical alias:
pip install evlib[pytorch]
```

Both extras resolve to the same pin (`torch >= 2.0.0`, `torchvision >= 0.15.0`). If PyTorch is not installed, importing a name such as `evlib.models.E2VID` raises an `ImportError` that names the missing dependency and the install command, rather than a bare `AttributeError`.

## The common `BaseModel` interface

`E2VID`, `E2VIDRecurrent` and `RVT` all subclass `evlib.models.base.BaseModel`. The base class fixes three things every model shares: how a model is constructed, how raw events are turned into a dense input tensor, and the shape of the abstract `reconstruct` method each subclass must implement.

```python notest
class BaseModel(ABC):
    def __init__(self, config: Optional[ModelConfig] = None, pretrained: bool = False): ...

    def preprocess_events(self, events, height=None, width=None) -> tuple:
        """Accepts a structured array, an (xs, ys, ts, ps) tuple, or a Polars
        LazyFrame/DataFrame; returns (xs, ys, ts, ps, height, width)."""

    def events_to_voxel_grid(self, xs, ys, ts, ps, height, width) -> np.ndarray:
        """Builds a (num_bins, height, width) voxel grid via
        evlib.representations.create_voxel_grid."""

    @abstractmethod
    def reconstruct(self, events, height=None, width=None) -> np.ndarray: ...
```

`preprocess_events` is the common entry point: pass it a structured numpy array with `x`/`y`/`t`/`p` fields, a `(xs, ys, ts, ps)` tuple, or a Polars `LazyFrame`/`DataFrame` (the kind `evlib.load_events` returns), and it normalises all three shapes to `(xs, ys, ts, ps, height, width)` numpy arrays, inferring `height`/`width` from the max coordinates when not given. `events_to_voxel_grid` builds on that to produce a dense `(num_bins, height, width)` tensor using evlib's own voxel-grid representation; `E2VID` overrides it with a simpler binning scheme better suited to its architecture (see below).

Every model also implements `_build_model()` and `_load_pretrained_weights()` (both abstract on `BaseModel`), and exposes `config` (a `ModelConfig`) and `pretrained` (bool) as public attributes.

### `ModelConfig`

`evlib.models.config.ModelConfig` is a dataclass with the fields `in_channels` (default 5), `out_channels` (1), `base_channels` (32), `num_layers` (4), `num_bins` (5), `use_gpu` (True), and a free-form `extra_params` dict for architecture-specific extensions. A small set of named presets ship in `CONFIGS` and are reachable through `get_config`:

```python
from evlib.models.config import get_config

cfg = get_config("lite")   # matches the bundled e2vid-lite.pth architecture
print(cfg)
```

The preset names are `default`, `lite`, `high_res`, `fast`, `temporal`, `spade`, and `ssl`; `get_config` raises `ValueError` for anything else.

## E2VID: single-frame reconstruction

`E2VID` is a UNet (encoder/residual-bottleneck/decoder with skip connections, `skip_type="sum"` or `"concat"`) based on the official RPG E2VID model (Rebecq et al., CVPR 2019). `reconstruct(events, height=None, width=None)` preprocesses the events, bins them into a voxel grid, and runs one forward pass, returning a single `(height, width)` float32 frame:

```python
import evlib
from evlib.models import E2VID

events = evlib.load_events("data/slider_depth/events.txt")

# pretrained=False builds the architecture with random weights; useful for
# checking the API and tensor shapes without a checkpoint on disk.
model = E2VID(pretrained=False)
frame = model.reconstruct(events)
print(frame.shape, frame.dtype)  # (180, 240) float32
```

`E2VID`'s constructor also takes `skip_type`, `num_encoders` (default 4), `num_residual_blocks` (default 2), and `norm` (`"BN"`, `"IN"`, or `None`), matching the original RPG defaults. `E2VID` overrides `events_to_voxel_grid` with a simple temporal-binning accumulation (no bilinear interpolation), which is what the original E2VID architecture expects as input, rather than evlib's more general voxel-grid representation.

### `E2VIDRecurrent`: stateful reconstruction

`E2VIDRecurrent` is the recurrent variant matching the bundled `e2vid-lite.pth` checkpoint: every encoder is followed by a `ConvLSTM`, so `reconstruct` takes and returns a state list, letting you reconstruct a video sequence chunk by chunk:

```python notest
import evlib
from evlib.models import E2VIDRecurrent

events = evlib.load_events("data/slider_depth/events.txt")

model = E2VIDRecurrent(pretrained=False)
frame, state = model.reconstruct(events)
print(frame.shape)             # (180, 240)
# Feed `state` back in on the next chunk of the same sequence:
next_frame, state = model.reconstruct(next_events_chunk, state=state)
```

Unlike `E2VID`, its `pretrained=True` path always loads the exact bundled `e2vid-lite.pth` checkpoint and derives the architecture (`num_bins`, `base_num_channels`, `num_encoders`, `num_residual_blocks`, `norm`) from the checkpoint's own tensor shapes before loading it `strict=True`.

## RVT: event-based object detection

`RVT` combines a recurrent MaxViT-style backbone (`RVTBackbone`), a PAFPN neck (`PAFPN`), and a YOLOX detection head (`YOLOXHead`) into a single detector, following "Recurrent Vision Transformers for Object Detection with Event Cameras" (Gehrig & Scaramuzza, CVPR 2023). Configuration lives in `RVTModelConfig`, which extends `ModelConfig` with detection-specific fields: `model_variant` (`"tiny"`/`"small"`/`"base"`), `temporal_bins` (default 10), `num_classes`, `confidence_threshold`, `nms_threshold`, `max_detections`, `input_height`/`input_width`, and `fpn_depth_multiplier`. `RVTModelConfig.tiny()`, `.small()`, and `.base()` are ready-made presets that set `base_channels` and `fpn_depth_multiplier` for each variant.

```python
from evlib.models import RVT

model = RVT(variant="tiny", pretrained=False, num_classes=2)
print(model.variant, model.temporal_bins)  # tiny 10
print(model.config.confidence_threshold, model.config.nms_threshold)  # 0.1 0.45
```

The detection entry point is `detect(events, height=None, width=None, confidence_threshold=None, nms_threshold=None, reset_states=False)`, which preprocesses events into a stacked-histogram tensor (`2 * temporal_bins` channels), runs the backbone/FPN/head, and returns a list of dicts each shaped `{"bbox": [x1, y1, x2, y2], "score": float, "class": int, "class_name": str}`. `RVT` also implements `BaseModel.reconstruct` for interface compatibility, but since RVT is a detector rather than a reconstruction model, that method just draws the detected boxes onto a blank `(height, width)` image; call `detect` directly for real use.

```python notest
# The MaxViT partition size must match the padded input resolution the
# checkpoint was trained with (see partition_size_from_hw in
# evlib.models.rvt_backbone; the RVT pipeline guide covers gen4's (6, 10)
# partition for 384x640 padded input in detail).
detections = model.detect(events, height=720, width=1280, reset_states=True)
for det in detections:
    print(det["class_name"], det["score"], det["bbox"])
```

Because RVT is recurrent, `detect` carries LSTM state across calls by default; `reset_states()` clears it explicitly, and `set_worker_id(worker_id)` lets a multi-worker evaluation loop keep one state per worker (see [Evaluation](evaluation.md), which uses exactly this to stream gen4 validation sequences).

## GPU inference

Every model picks a device at construction time: `self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`. There is no separate `.cuda()` call needed for the common path; `RVT.to(device)` overrides `nn.Module.to` to keep `self._device` in sync if you move the model explicitly.

## Loading pretrained weights

Weight files are not distributed with the `pip install evlib[torch]` wheel; `python/evlib/models/weights/` is gitignored, so you provide checkpoints yourself before passing `pretrained=True`:

- `E2VID` looks for any `*.pth*` file in `weights/`, picks the first one alphabetically, remaps the reference RPG E2VID checkpoint keys onto its own module names, and adapts its architecture (base channels, encoder count, norm, transposed-conv vs. upsample-conv) to whatever the checkpoint's tensor shapes imply.
- `E2VIDRecurrent` looks for exactly `weights/e2vid-lite.pth` and loads it `strict=True`.
- `RVT` looks for `weights/rvt-{t,s,b}.ckpt`, `weights/rvt_{tiny,small,base}.ckpt`, or `weights/rvt.ckpt` (checked in that order for the requested variant), remaps the PyTorch Lightning checkpoint's parameter names, and loads with `strict=False`.

If `pretrained=True` and no matching file is found, or loading otherwise fails, all three raise (`FileNotFoundError` or `RuntimeError`) rather than silently continuing with the freshly-initialised random weights: a model that looks pretrained but scores a meaningless mAP is worse than one that fails loudly at construction time.

## Where to go next

- [Datasets & Training Data](datasets.md): `evlib.data`'s PyTorch datasets, augmentation, and the `evlib-rvt-preprocess` console script for turning raw sequences into RVT-compatible training data.
- [Evaluation](evaluation.md): Prophesee-compatible mAP scoring for a trained RVT model.
- [RVT Preprocessing Pipeline](../rvt/index.md): the four scatter-add backends behind the stacked-histogram representation RVT consumes.
