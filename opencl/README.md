# OpenCL-Agent-Hub
Portable GPU/CPU agents using OpenCL for parallel consciousness & compute experiments.

Runs inside the GARVIS project — hardware-agnostic compute layer for the lattice engine.

## Why OpenCL?
- Runs everywhere: NVIDIA / AMD / Intel / Qualcomm / FPGA / plain CPU
- No vendor lock — unlike CUDA
- Fast kernels for AI, sims, lattice math
- SVM (Shared Virtual Memory) for zero-copy data sharing
- SPIR-V: compile once, deploy anywhere

## Core Features
| # | Feature | File | Status |
|---|---------|------|--------|
| 1 | **Lattice Kernel** — φ-wave propagation on GPU | `kernels/phi_propagate.cl` | ✓ |
| 2 | **Agent Loop** — Observer/Actor/Bridge in parallel threads | `agent_loop.c` | ✓ |
| 3 | **QASM Bridge** — OpenCL wrapper for quantum sim fallback | *(planned)* | — |
| 4 | **Pulse Monitor** — real-time C value → stdout / voice alert | *(planned)* | — |

## Setup

### Linux
```bash
apt install ocl-icd-opencl-dev opencl-headers
# vendor runtime (pick one):
apt install intel-opencl-icd          # Intel CPU/GPU
apt install amdgpu-pro-opencl          # AMD
# NVIDIA: install CUDA toolkit (bundles OpenCL)
```

### macOS
OpenCL ships with Xcode — no extra install needed.

### Windows
Install your GPU vendor's OpenCL SDK (NVIDIA CUDA, AMD ROCm, or Intel oneAPI).

## Build
```bash
cd opencl/
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/garvis_opencl
```

## Kernel overview

### `phi_propagate` — the core law
```c
C[i] = clamp(O[i] * A[i] * B[i] * PHI, 0.0f, 1.0f);
```
Each work-item handles one lattice node independently — full SIMD parallelism.
144 kernel launches per run (Fibonacci cycle count).

## Roadmap
- [ ] SVM buffer variant (zero-copy on APUs / mobile SoCs)
- [ ] SPIR-V offline compilation (`clspv` or `clang -x cl`)
- [ ] Python `pyopencl` wrapper to feed results into `lattice_core.py`
- [ ] Adreno / Qualcomm profiling (per llama.cpp OpenCL backend precedent)
