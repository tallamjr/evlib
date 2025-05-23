I ultimately want evlib to be the one stop shop for event based vision utilities
as well as a place where many of the state of the art algorithms have been
implemented and reside.

Events to Video algorithms (vid2e) I would like to implement are:

- [E2VID](https://github.com/uzh-rpg/rpg_e2vid)
- [FireNet](https://github.com/cedric-scheerlinck/rpg_e2vid/tree/cedric/firenet)
- [E2VID+](https://github.com/TimoStoff/event_cnn_minimal)
- [FireNet+](https://github.com/TimoStoff/event_cnn_minimal)
- [SPADE-E2VID](https://github.com/RodrigoGantier/SPADE_E2VID)
- [SSL-E2VID](https://github.com/tudelft/ssl_e2vid)
- [ET-Net](https://github.com/WarranWeng/ET-Net)
- [HyperE2VID](https://github.com/ercanburak/HyperE2VID)

They all should use Candle framework to re-implement the architecture and use
existing pytorch model files where possible. We can leverage either the existing
PyTorch loader (pytorch_loader.rs) or ONNX Runtime (ort) for model inference.

https://github.com/ercanburak/EVREAL does a good job of comparing the different
algorithms and points to many good resources. I would like all of these
algorithms to have their implementation at: `evlib/src/ev_processing/reconstruction/`

After that my plan is to implement the functionality found here that does the
reverse, i.e Video to Events and works as a simulator. The main repo for this
can be found here: https://github.com/uzh-rpg/rpg_e2vid

A useful set of open-source tooling can be found here:
https://gitlab.com/inivation/dv/dv-processing. I would like to eventually
incorporate the functionality there inside evlib too.

A project that is aiming to do similar things is
https://github.com/ucsd-hdsi-dvs/V2CE-Toolbox, maybe you can draw inspiration
from how they do things.

Of course there is also https://github.com/prophesee-ai/openeb too and
https://github.com/shiba24/event-vision-library and actually there is a good
amount of tooling found here too: https://github.com/tub-rip/ETAP

## Implementation Status

### Recent Progress (January 2025)
- ✅ Implemented E2VID UNet and FireNet architectures in Candle
- ✅ Integrated ONNX Runtime (ort v2.0.0-rc.9) for model inference
- ✅ Added Python API with model selection (unet/firenet/onnx/simple)
- ✅ Comprehensive test suite for both Rust and Python APIs
- ✅ Successfully tested with slider_depth dataset
- ✅ PyTorch to ONNX model converter with validation
- ✅ GPU optimization utilities (CUDA/Metal support)
- ✅ EVREAL benchmarking metrics (MSE, PSNR, SSIM, MS-SSIM)
- ✅ Temporal consistency metrics for video sequences

## Implementation Plan

### Phase 1: Foundation (4-6 weeks) - ✅ COMPLETED

**Priority: HIGH**

1. **Core Infrastructure** ✅ COMPLETED
   - ✅ ONNX Runtime (ort) integration
   - ✅ Enhanced voxel grid representations
   - ✅ Candle-based E2VID UNet and FireNet architectures
   - ✅ PyTorch to ONNX model converter
   - ✅ GPU optimization pipeline (MPS on Mac, CUDA support)
   - ✅ Benchmarking framework (EVREAL metrics)

2. **Base Algorithms** ✅ COMPLETED
   - ✅ **E2VID**: CNN-based UNet architecture implemented
   - ✅ **FireNet**: Lightweight speed-optimized variant implemented
   - ✅ Model conversion utilities and download scripts

### Phase 2: Enhanced Variants (4-6 weeks) 🚧 IN PROGRESS

**Priority: MEDIUM**

3. **E2VID+**: Enhanced features and training (2-3 weeks) ✅ COMPLETED
   - ✅ ConvLSTM implementation for temporal processing
   - ✅ E2VID+ architecture with temporal memory
   - ✅ Simplified temporal attention mechanism
   - 🔲 Python bindings for E2VID+
   - 🔲 Integration tests with real event data

4. **FireNet+**: Enhanced FireNet with additional features (2-3 weeks) ✅ COMPLETED
   - ✅ FireNet+ lightweight variant with temporal gating
   - ✅ FireModulePlus with temporal processing
   - 🔲 Python bindings for FireNet+
   - 🔲 Performance benchmarking vs base FireNet

### Phase 3: Advanced Architectures (6-8 weeks)

**Priority: MEDIUM**

5. **SPADE-E2VID**: Requires custom SPADE layers (3-4 weeks)
6. **SSL-E2VID**: Self-supervised approach (3-4 weeks)

### Phase 4: Cutting-Edge Research (8-12 weeks)

**Priority: LOW - Long-term**

7. **ET-Net**: Transformer-based (4-6 weeks)
8. **HyperE2VID**: Dynamic convolutions + hypernetworks (4-6 weeks)

### Algorithm Complexity Analysis

**Low Effort (1-2 weeks):** ✅ COMPLETED

- ✅ E2VID basic implementation
- ✅ FireNet implementation
- ⏳ Model loading from PyTorch checkpoints

**Medium Effort (3-4 weeks):**

- E2VID+ with enhanced features
- SPADE-E2VID (requires custom SPADE layers)
- ConvLSTM implementation for temporal processing

**High Effort (1-2 months):**

- ET-Net (transformer architecture)
- SSL-E2VID (self-supervised training)
- HyperE2VID (dynamic convolutions + hypernetworks)

### Implementation Notes

**Candle Framework Capabilities:**

- CNN layers available (conv2d, batch norm, layer norm)
- Limited transformer and ConvLSTM support (will need custom implementation)

**Completed Components:**

- ✅ E2VID UNet architecture in Candle (e2vid_arch.rs)
- ✅ FireNet architecture in Candle (e2vid_arch.rs)
- ✅ ONNX Runtime integration (onnx_loader_simple.rs)
- ✅ Python API with model selection (events_to_video_advanced)
- ✅ Basic PyTorch loader infrastructure (pytorch_loader.rs)

**Remaining Components:**

- ⏳ Loading actual pre-trained weights (.pth files)
- ⏳ GPU optimization (CUDA/Metal providers)
- ✅ ConvLSTM layers for temporal processing
- ⏳ SPADE normalization layers
- ⏳ Benchmark metrics implementation

**Benchmark Framework (EVREAL):**

- MSE, SSIM, LPIPS (full-reference metrics)
- BRISQUE, NIQE, MANIQA (no-reference metrics)

**Dataset Support:**

- 🔲 ECD (Event Camera Dataset)
- 🔲 MVSEC (Multi Vehicle Stereo Event Camera)
- 🔲 HQF (High Quality Frames)
- 🔲 BS-ERGB (Beam Splitter Event-RGB)
- 🔲 HDR (High Dynamic Range)
