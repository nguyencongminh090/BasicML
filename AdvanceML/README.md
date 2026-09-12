# AdvanceML

A from-scratch, performance-oriented deep learning library in modern C++ (C++20/23), with a real autograd engine — a define-by-run computation graph and `.backward()`, unlike [`BasicML`](../BasicML)'s hand-written per-layer `backward()`. GEMM/conv-heavy ops are backed by Intel oneDNN, which targets this machine's AVX-512 VNNI.

This is a sibling project to `BasicML`, not a replacement: `BasicML` stays a from-scratch, manual-backward, pure-NumPy teaching project. `AdvanceML` exists because that design cannot run CNN/Transformer-scale workloads efficiently on CPU.

See root `CLAUDE.md`'s `## AdvanceML` section for full architecture/style rules, and `ai-audit/instructions/TODO-0030.md` for the reasoning behind the language/backend/build-system choices, including why hand-written kernels aren't expected to beat oneDNN/PyTorch on raw GEMM throughput.

## Build

Requires a C++20/23 compiler, CMake, Ninja, and oneDNN development headers (`libdnnl-dev` on Debian/Ubuntu).

```bash
cmake -B build -S . -G Ninja
cmake --build build
ctest --test-dir build
```

Catch2 (test framework) is fetched automatically via CMake `FetchContent` — nothing to install manually for tests.

## Status

Project scaffold only (build system, directory layout, one placeholder smoke test). The autograd engine itself has not been implemented yet — see the ai-audit backlog for follow-up TODOs.
