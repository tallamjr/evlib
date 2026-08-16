#!/usr/bin/env bash
# Build the CUDA event simulator kernel into libevsim.so.
# Usage: scripts/build_cuda_kernels.sh [outdir] [arch]
# NVCC overrides the compiler (default nvcc on PATH); arch defaults to native.
set -euo pipefail
out="${1:-target/cuda}"
arch="${2:-native}"
nvcc_bin="${NVCC:-nvcc}"
root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$out"
"$nvcc_bin" -O3 -std=c++17 -arch="$arch" -cudart static -shared -Xcompiler -fPIC \
  -o "$out/libevsim.so" "$root/src/ev_simulation/cuda/esim_kernel.cu"
echo "built $out/libevsim.so; export EVLIB_CUDA_SIM_LIB=$(cd "$out" && pwd)/libevsim.so"
