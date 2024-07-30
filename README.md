# Requirements:
- NVIDIA® GPU
- NVIDIA® CUDA® toolkit
- Rust
- Cmake

A docker configuration is provided on `.devcontainer/`

# Project structure
- `cuda/`: cuda code of stwo backend
- `stwo_gpu_backend/`: rust wrapper and binding, implements CudaBackend for consumption in stwo prover

# Execution
In `stwo_gpu_backend` directory:
```bash
cargo test
```