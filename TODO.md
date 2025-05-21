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
existing pytorch model files where possible.

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

## Implementation Plan

### Phase 1: Foundation (4-6 weeks) - START HERE
**Priority: HIGH**

1. **Core Infrastructure**
   - PyTorch model loading in Candle
   - Enhanced voxel grid representations
   - Benchmarking framework (EVREAL metrics)
   - GPU optimization pipeline

2. **Base Algorithms**
   - **E2VID**: CNN-based reconstruction (1-2 weeks)
   - **FireNet**: Speed-optimized variant (1-2 weeks)

### Phase 2: Enhanced Variants (4-6 weeks)
**Priority: MEDIUM**

3. **E2VID+**: Enhanced features and training (2-3 weeks)
4. **FireNet+**: Enhanced FireNet with additional features (2-3 weeks)

### Phase 3: Advanced Architectures (6-8 weeks)
**Priority: MEDIUM**

5. **SPADE-E2VID**: Requires custom SPADE layers (3-4 weeks)
6. **SSL-E2VID**: Self-supervised approach (3-4 weeks)

### Phase 4: Cutting-Edge Research (8-12 weeks)
**Priority: LOW - Long-term**

7. **ET-Net**: Transformer-based (4-6 weeks)
8. **HyperE2VID**: Dynamic convolutions + hypernetworks (4-6 weeks)

### Algorithm Complexity Analysis

**Low Effort (1-2 weeks):**
- E2VID basic implementation
- FireNet implementation
- Model loading from PyTorch checkpoints

**Medium Effort (3-4 weeks):**
- E2VID+ with enhanced features
- SPADE-E2VID (requires custom SPADE layers)
- ConvLSTM implementation for temporal processing

**High Effort (1-2 months):**
- ET-Net (transformer architecture)
- SSL-E2VID (self-supervised training)
- HyperE2VID (dynamic convolutions + hypernetworks)

### Recommended Git Worktree Strategy

For parallel development:
```bash
git worktree add ../evlib-e2vid feature/e2vid-implementation
git worktree add ../evlib-firenet feature/firenet-implementation
git worktree add ../evlib-infra feature/pytorch-model-loading
```

### Implementation Notes

**Candle Framework Capabilities:**
- CNN layers available (conv2d, batch norm, layer norm)
- PyTorch model loading support (.pth files)
- GPU acceleration via CUDA
- Limited transformer and ConvLSTM support (will need custom implementation)

**Missing Components for Full Implementation:**
- Actual neural network models (currently using simple accumulation in e2vid.rs)
- PyTorch model loading and conversion
- GPU optimization
- ConvLSTM layers for temporal processing
- SPADE normalization layers

**Benchmark Framework (EVREAL):**
- MSE, SSIM, LPIPS (full-reference metrics)
- BRISQUE, NIQE, MANIQA (no-reference metrics)
- Support for ECD, MVSEC, HQF, BS-ERGB, HDR datasets
