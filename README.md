# Silero VAD C Port

Standalone C port of the 16 kHz [Silero VAD model](https://github.com/snakers4/silero-vad) with embedded weights and no ONNX dependency.

This repository packages the model as a small shared library for: Windows, macOS, and Linux.

The code currently focuses on the 16 kHz model path and supports both chunked inference and full-audio probability extraction.

## Current status

- 16 kHz model path implemented
- Embedded-weight builds supported
- Windows DLL, Linux and macOS shared-library builds supported through CMake
- Several performance optimizations are already implemented:
  - stft-conv replaced by fixed 256-point FFT
  - fused STFT to first conv path
  - shared SIMD abstraction for scalar, SSE, AVX2 and NEON backends
  - AVX2 conv and LSTM kernels for x86_64
  - SSE conv and LSTM kernels for x86 and x86_64
  - NEON conv and LSTM kernels for arm64

## Python example

Load a built shared library and run full-audio inference:

```python
from run_silero_vad_clib import SileroVadClib
from src.silero_vad.utils_vad import read_audio

audio = read_audio("src/silero_vad/test/tests_data_test.wav", 16000)

with SileroVadClib("build-windows-x64-avx2/silero_vad.dll") as model:
    probs = model.forward_audio(audio)
```

## Repository layout

- `silero_vad.c` / `silero_vad.h` / `silero_vad_simd.h`
  - core C implementation, public API, and SIMD abstraction layer
- `export_silero_vad_weights.py`
  - exports weights from safetensors
- `export_silero_vad_jit_weights.py`
  - exports weights from the Torch hub JIT model
- `run_silero_vad_clib.py`
  - Python `ctypes` example / test runner for the shared library
- `package_release.py`
  - creates release folders and zip files from built binaries
<!-- - `.github/workflows/`
  - release workflows -->


## Model weights

Two weight sources are supported:

- `jit`
  - exported from `torch.hub.load(..., model='silero_vad')`
  - best choice if you want parity with the original Silero Torch hub model
- `safetensors`
  - exported from the local safetensors checkpoint

## Building and packaging

Build and release commands are collected in:

- [`building.md`](/building.md)

That file includes: Windows builds, Linux builds, macOS builds, and release packaging commands


## Public API

The public API is declared in `silero_vad.h`.

Main entry points:

- `silero_vad_model_create`
- `silero_vad_model_destroy`
- `silero_vad_model_reset`
- `silero_vad_model_forward`
- `silero_vad_model_forward_audio`

The current full-model path is designed around 16 kHz audio and typically uses:

- `576` samples per chunk
- `64` samples of left context
- `512` new samples

## Notes

- x86 and x86_64 SIMD backends are now build-selectable:
  - scalar baseline
  - SSE with `SILERO_VAD_ENABLE_SSE=ON`
  - AVX2 with `SILERO_VAD_ENABLE_AVX2=ON`
- Use the baseline or SSE build as the compatibility build for older x86 and x86_64 CPUs without AVX2.
- Use the AVX2 build when you want the faster x86_64 path on AVX2-capable CPUs.
- `SILERO_VAD_FAST_MATH` is available, but on current tests it did not improve performance.
- ARM builds support the NEON-optimized path when `SILERO_VAD_ENABLE_NEON=ON`.

## License

This repository is based on the MIT-licensed Silero VAD project.
