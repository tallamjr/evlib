# evlib Development Roadmap

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

## 🚨 CRITICAL: Audit Results & Immediate Actions Required

**Date**: January 2025
**Status**: 📋 **AUDIT COMPLETE - Major Issues Identified**

A comprehensive audit has revealed significant discrepancies between claimed functionality and actual implementation. See `REPORT.md` for full details.

### 🔴 Critical Issues Found

1. **❌ 6 of 8 "implemented" models are placeholders** that fallback to basic UNet
2. **❌ Examples are broken** due to API mismatches (wrong function names)
3. **❌ PyTorch weight loading is non-functional** (downloads but doesn't load weights)
4. **❌ Performance claims are unverified** (benchmark example broken)
5. **❌ Many test files are empty placeholders**

### ⚡ Immediate Actions (Next 2-3 weeks)

#### Phase 0: Foundation Cleanup - 🔴 URGENT
**Priority: CRITICAL - Restore credibility**

1. **Fix Documentation (1-2 days)** ⏱️ HIGH PRIORITY
   - ❌ Remove "8 reconstruction algorithms" claims from README
   - ❌ Remove "Research-ready library complete" statements
   - ❌ Remove unverified performance claims (5x-47x speedup)
   - ✅ Add honest description of current capabilities
   - ✅ Add clear warnings about placeholder models

2. **Fix Broken Examples (3-5 days)** ⏱️ HIGH PRIORITY
   - ❌ Fix `benchmark.py`: `events_to_block_py` → `events_to_block`
   - ❌ Fix `event_augmentation.py`: `add_random_events_py` → `add_random_events`
   - ❌ Fix all API mismatches in examples/
   - ✅ Test all examples before claiming they work
   - ✅ Remove broken examples that cannot be quickly fixed

3. **Fix Weight Loading (1 week)** ⏱️ HIGH PRIORITY
   - ❌ Investigate why downloaded weights aren't being used
   - ❌ Make model outputs deterministic when using real weights
   - ✅ Add verification that weights are actually loaded
   - ✅ Fix non-deterministic model behavior

4. **Clean Up Placeholder Models (3-5 days)** ⏱️ MEDIUM PRIORITY
   - ❌ Add clear warnings to all placeholder model classes
   - ❌ Mark as "experimental/placeholder" in documentation
   - ✅ Option: Remove placeholders entirely until properly implemented
   - ✅ Be honest about what actually works

## Current Implementation Status - 🔍 AUDITED

### ✅ ACTUALLY WORKING (Verified January 2025)

**Core Functionality** - ✅ SOLID FOUNDATION
- ✅ **Event data structures** and basic manipulation
- ✅ **Voxel grid representations** (standard and smooth)
- ✅ **Event simulation** (ESIM) - recently implemented and working
- ✅ **Data I/O** - loading/saving events in multiple formats
- ✅ **Basic transformations** - flipping, rotation, clipping
- ✅ **Visualization** - event-to-image conversion

**Neural Network Models** - ⚠️ LIMITED
- ✅ **E2VID UNet** - basic event-to-video reconstruction (working but non-deterministic)
- ✅ **FireNet** - lightweight variant (working but non-deterministic)

**Infrastructure** - ⚠️ PARTIAL
- ✅ **Model downloading** - successfully downloads model files
- ✅ **Python API structure** - proper namespace organization
- ❌ **Weight loading** - downloads work but weights aren't used
- ❌ **Benchmarking** - broken due to API mismatches

### ❌ PLACEHOLDER/BROKEN (Need Implementation)

**"Advanced" Models** - ❌ ALL PLACEHOLDERS
- ❌ **E2VID+** - placeholder that falls back to basic UNet
- ❌ **FireNet+** - placeholder that falls back to basic UNet
- ❌ **SPADE-E2VID** - placeholder that falls back to basic UNet
- ❌ **SSL-E2VID** - placeholder that falls back to basic UNet
- ❌ **ET-Net** - placeholder that falls back to basic UNet
- ❌ **HyperE2VID** - placeholder that falls back to basic UNet

**Examples & Documentation** - ❌ MOSTLY BROKEN
- ❌ **benchmark.py** - API mismatches, cannot run
- ❌ **event_augmentation.py** - API mismatches, cannot run
- ❌ **Multiple examples** - use non-existent function names
- ✅ **basic_usage.py** - only working example

**Testing** - ❌ INSUFFICIENT
- ❌ **12 of 29 test files** are empty placeholders
- ❌ **Model tests** don't verify actual functionality
- ❌ **Integration tests** for advanced features missing

## Realistic Development Plan

### Phase 1: Foundation Cleanup (2-3 weeks) - 🔴 CRITICAL
**Status**: 🚀 STARTING IMMEDIATELY
**Goal**: Restore honesty and fix broken basics

1. **Documentation Cleanup** (2-3 days) - ⏱️ URGENT
   - Fix README to reflect actual capabilities
   - Remove false claims and misleading statements
   - Add clear status indicators for all features
   - Create honest feature matrix

2. **Example Fixes** (1 week) - ⏱️ HIGH PRIORITY
   - Fix all API mismatches in examples/
   - Test every example to ensure it works
   - Remove or clearly mark broken examples
   - Add automated testing for examples

3. **Weight Loading Fix** (1 week) - ⏱️ HIGH PRIORITY
   - Debug why weights aren't being loaded despite downloads
   - Make model outputs deterministic
   - Add verification of weight loading
   - Fix PyTorch-to-Candle integration

4. **Test Suite Cleanup** (3-5 days) - ⏱️ MEDIUM PRIORITY
   - Remove empty placeholder tests
   - Add real tests for claimed functionality
   - Achieve >80% coverage for working features
   - Add automated testing in CI

### Phase 2: Model Implementation (2-3 months) - 🔧 DEVELOPMENT
**Status**: 🔲 PENDING Phase 1 completion
**Goal**: Actually implement the claimed algorithms

**Choice Point**: Remove placeholders vs. implement them

**Option A: Remove Placeholders (Recommended for immediate honesty)**
- Remove all 6 placeholder model classes
- Update documentation to reflect only 2 working models
- Focus on making E2VID and FireNet excellent
- Timeline: 2-3 days

**Option B: Implement Real Models (Long-term goal)**
- E2VID+ with temporal features (3-4 weeks)
- FireNet+ lightweight variant (2-3 weeks)
- SPADE-E2VID with spatial normalization (4-5 weeks)
- SSL-E2VID with self-supervised learning (4-5 weeks)
- ET-Net transformer architecture (5-6 weeks)
- HyperE2VID with dynamic convolutions (4-5 weeks)
- Timeline: 4-6 months for all

**Hybrid Approach**: Remove placeholders now, implement 1-2 models properly per month

### Phase 3: Advanced Features (3-6 months) - 🚀 FUTURE
**Status**: 🔲 PENDING Phase 2 completion

1. **Performance Optimization** (2-3 weeks)
   - Real benchmarking with verified performance claims
   - Memory optimization and SIMD acceleration
   - GPU acceleration improvements
   - Multi-threading optimization

2. **Production Features** (4-6 weeks)
   - REST API for model inference
   - Docker containers and cloud deployment
   - Model quantization and edge optimization
   - Monitoring and logging infrastructure

3. **Ecosystem Integration** (4-8 weeks)
   - OpenEB format support
   - ROS2 integration
   - Prophesee SDK compatibility
   - DV Processing compatibility layer

## Success Metrics (Realistic)

### Short-term (1 month)
- [ ] All examples work without errors
- [ ] Documentation accurately reflects capabilities
- [ ] Models produce deterministic outputs when using weights
- [ ] Test coverage >80% for claimed functionality
- [ ] No false claims in documentation

### Medium-term (3 months)
- [ ] At least 4 model variants actually implemented
- [ ] Performance benchmarks verified and reproducible
- [ ] Complete API documentation with working examples
- [ ] Real weight loading for all implemented models
- [ ] Community feedback indicates restored credibility

### Long-term (6 months)
- [ ] 6-8 model variants fully implemented
- [ ] Production-ready deployment tools
- [ ] Research papers citing evlib
- [ ] Active community contributions
- [ ] Industry adoption for real applications

## Technical Debt to Address

### Critical Technical Debt
1. **Placeholder Models**: 6 models claiming to work but don't
2. **Broken Examples**: Most examples don't run due to API mismatches
3. **Non-functional Weight Loading**: Downloads but doesn't use weights
4. **Empty Tests**: 40% of test files are placeholders
5. **Inconsistent API**: Function names don't match between docs and code

### Development Standards Going Forward
1. **No "Fake it Till You Make It"**: Only mark things complete when they actually work
2. **Test-Driven Development**: Write tests before claiming functionality works
3. **API Consistency**: Ensure examples use actual API function names
4. **Documentation Accuracy**: All claims must be verifiable
5. **Incremental Honesty**: Better to under-promise and over-deliver

## Current Focus: Phase 1 Foundation Cleanup

**Immediate Next Steps (This Week)**:
1. ✅ Create REPORT.md with audit findings
2. ✅ Update TODO.md with realistic roadmap
3. 🔲 Fix README.md to remove false claims
4. 🔲 Fix broken examples (benchmark.py, event_augmentation.py)
5. 🔲 Add warnings to placeholder model classes
6. 🔲 Debug and fix weight loading issue

**Goal**: By end of month, evlib should honestly represent its capabilities and all claimed functionality should actually work.

---

## Notes

- All future implementations will prioritize honesty over impressive claims
- Focus on making working features excellent rather than adding broken features
- Community trust is more valuable than inflated feature lists
- Every "✅ COMPLETED" must be verified by working tests and examples

**Philosophy**: "Better to be a working library with 2 great models than a broken library claiming 8 models"
