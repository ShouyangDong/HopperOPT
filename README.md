# HopperOPT

HopperOPT is a small collection of CUDA kernels and test scripts focused on high-performance GEMM implementations and pipeline/unroll experiments targeting modern NVIDIA architectures (Ampere, Hopper, and similar GPUs).

**Key goals**
- Provide example GEMM kernels optimized for recent NVIDIA GPU architectures.
- Show pipeline and unroll micro-benchmarks and tests.
- Offer a lightweight starting point for experimenting with CUDA optimizations.

**Repository layout**
- `gemm_hopper.cu` — GEMM example tuned for Hopper-like hardware.
- `gemm_ampere.cu` — GEMM example with Ampere-oriented tuning.
- `test_hopper_pipeline.py` — Python test/benchmark for Hopper pipeline logic.
- `test_ampere_pipeline.py` — Python test/benchmark for Ampere pipeline logic.
- `test_gemm.py` — Functional correctness tests for GEMM kernels.
- `test_unroll.py` — Tests demonstrating unroll optimizations.

Requirements
------------
- CUDA Toolkit (nvcc) compatible with your GPU and drivers.
- Python 3.8+ (for the test scripts).
- Python packages: `pytest`, `numpy` (install with `pip`).

Suggested environment setup (macOS / zsh)
--------------------------------------
1. Install Python dependencies in a virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pytest numpy
```

2. Ensure CUDA is installed and `nvcc` is on your `PATH`. Set `CUDA_HOME` if necessary, for example:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
```

Building the CUDA examples
--------------------------
You can compile the CUDA files with `nvcc`. Choose the appropriate compute architecture for your GPU (e.g., `sm_80`, `sm_90`, etc.). Replace `<arch>` in the examples below with the correct value for your card.

Example:

```bash
# Compile a Hopper-oriented binary (adjust -arch as needed)
nvcc -O3 -arch=sm_90 gemm_hopper.cu -o gemm_hopper

# Compile the Ampere example
nvcc -O3 -arch=sm_80 gemm_ampere.cu -o gemm_ampere
```

Notes:
- The repository uses small example kernels and Python test harnesses. You may need to adapt compile flags, include paths, or additional linker flags depending on your CUDA installation.
- For best performance tuning, set the correct `-arch` and consider additional flags such as `-lineinfo` or `-Xptxas` for debug/profiling.

Running tests
-------------
The included Python tests can be run with `pytest`. They are intended as lightweight correctness and micro-benchmark checks.

```bash
# From repository root (activate your virtualenv first)
pytest -q
```

If tests call compiled binaries, ensure you have compiled the required CUDA examples first.

Contributing
------------
- Feel free to open issues or submit PRs to add more kernels, improve tuning, or add CI for automated testing on supported hardware.

License
-------
This repository does not include a license file. If you plan to publish or share this project, add a `LICENSE` file with your preferred license.

Contact
-------
For questions or suggestions, open an issue or contact the repository owner.
