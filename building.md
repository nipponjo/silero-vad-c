# Build Guide

This file collects the release-oriented build commands for the standalone C port.

The commands below assume:

- CMake is installed
- Python is available for the weight export step
- you want to build shared libraries

## Common flags

These options appear repeatedly in the commands below:

- `-DSILERO_VAD_WEIGHT_SOURCE=jit`
  - embeds weights exported from the Torch hub JIT model
  - use this when you want parity with the original Silero model
- `-DSILERO_VAD_WEIGHT_SOURCE=safetensors`
  - embeds weights exported from the local safetensors checkpoint
- `-DSILERO_VAD_ENABLE_AVX2=ON`
  - enables the AVX2/FMA-optimized x86 path
- `-DSILERO_VAD_ENABLE_NEON=ON`
  - enables the NEON-optimized ARM path
- `-DSILERO_VAD_ENABLE_LTO=ON`
  - enables link-time optimization
  - this can reduce binary size or improve runtime by optimizing across translation units
- `-DCMAKE_C_FLAGS_RELEASE="... -flto ..."`
  - `-O3` asks the compiler for aggressive release optimization
  - `-flto` enables link-time optimization in the compiler flags
  - `-march=...` controls the minimum CPU ISA the binary may use
  - `-mtune=generic` keeps scheduling conservative for portability

## Windows

### Portable x64

Use this for the broadest Windows x64 compatibility build.

```powershell
cmake -S . -B build-windows-x64-nosimd -G "Visual Studio 18 2026" -A x64 `
  -DSILERO_VAD_WEIGHT_SOURCE=jit

cmake --build build-windows-x64-nosimd --config Release
```

Output:

- `build-windows-x64-nosimd/silero_vad.dll`


### Portable SSE

Use this when you want the faster x86 build and can assume SSE support.

```powershell
cmake -S . -B build-windows-x64-sse -G "Visual Studio 18 2026" -A x64 `
  -DSILERO_VAD_WEIGHT_SOURCE=jit `
  -DSILERO_VAD_ENABLE_SSE=ON

cmake --build build-windows-x64-sse --config Release
```

Output:

- `build-windows-x64-sse/silero_vad.dll`


### Portable x64 AVX2

Use this when you want the faster x86 build and can assume AVX2 support.

```powershell
cmake -S . -B build-windows-x64-avx2 -G "Visual Studio 18 2026" -A x64 `
  -DSILERO_VAD_WEIGHT_SOURCE=jit `
  -DSILERO_VAD_ENABLE_AVX2=ON

cmake --build build-windows-x64-avx2 --config Release
```

Output:

- `build-windows-x64-avx2/silero_vad.dll`


## Linux

Recommended release targets:

- `x86_64`
  - portable baseline x86_64 build
- `x86_64-avx2`
  - faster x86_64 build for AVX2-capable CPUs
- `arm64`
  - portable AArch64 build with NEON enabled
- `armv7`
  - portable 32-bit ARM hard-float build

### Prerequisites

Native `x86_64` build:

```bash
sudo apt update
sudo apt install -y cmake ninja-build python3 python3-pip
```

Cross-build support for `arm64`:

```bash
sudo apt update
sudo apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu ninja-build
```

Cross-build support for `armv7`:

```bash
sudo apt update
sudo apt install -y gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf ninja-build
```

### Portable x86_64

```bash
cmake -S . -B build-linux-x86_64 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSILERO_VAD_WEIGHT_SOURCE=jit \
  -DSILERO_VAD_ENABLE_LTO=ON \
  -DCMAKE_C_FLAGS_RELEASE="-O3 -flto -march=x86-64 -mtune=generic"

cmake --build build-linux-x86_64 --config Release
```

Output:

- `build-linux-x86_64/silero_vad.so`

### Portable x86_64 AVX2

```bash
cmake -S . -B build-linux-x86_64-avx2 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSILERO_VAD_WEIGHT_SOURCE=jit \
  -DSILERO_VAD_ENABLE_LTO=ON \
  -DSILERO_VAD_ENABLE_AVX2=ON \
  -DCMAKE_C_FLAGS_RELEASE="-O3 -flto -march=x86-64 -mtune=generic"

cmake --build build-linux-x86_64-avx2 --config Release
```

Output:

- `build-linux-x86_64-avx2/silero_vad.so`

### Portable arm64 cross-build

```bash
cmake -S . -B build-linux-arm64 -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/linux-aarch64.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DSILERO_VAD_WEIGHT_SOURCE=jit \
  -DSILERO_VAD_ENABLE_LTO=ON \
  -DSILERO_VAD_ENABLE_NEON=ON \
  -DCMAKE_C_FLAGS_RELEASE="-O3 -flto -march=armv8-a"

cmake --build build-linux-arm64 --config Release
```

Output:

- `build-linux-arm64/silero_vad.so`

### Portable armv7 cross-build

```bash
cmake -S . -B build-linux-armv7 -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/linux-armv7.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DSILERO_VAD_WEIGHT_SOURCE=jit \
  -DSILERO_VAD_ENABLE_LTO=ON \
  -DCMAKE_C_FLAGS_RELEASE="-O3 -flto -march=armv7-a -mfloat-abi=hard -mfpu=vfpv3-d16 -mtune=generic-armv7-a"

cmake --build build-linux-armv7 --config Release
```

Output:

- `build-linux-armv7/silero_vad.so`

<!-- ### CI release workflow

Linux release automation lives in:

- `.github/workflows/release-linux.yml`

That workflow builds and packages:

- `x86_64`
- `x86_64-avx2`
- `arm64`
- `armv7` -->

## macOS

### Portable Apple Silicon arm64

Use this for a broadly portable Apple Silicon build.

```bash
cmake -S . -B build-mac-arm64 -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
  -DSILERO_VAD_WEIGHT_SOURCE=jit \
  -DSILERO_VAD_ENABLE_NEON=ON \
  -DSILERO_VAD_ENABLE_LTO=ON

cmake --build build-mac-arm64 --config Release
```

Output:

- `build-mac-arm64/silero_vad.dylib`

### Apple Silicon local speed build

Use this only for local benchmarking when tuning for one specific Apple Silicon CPU.

```bash
cmake -S . -B build-mac-local -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
  -DSILERO_VAD_WEIGHT_SOURCE=jit \
  -DSILERO_VAD_ENABLE_NEON=ON \
  -DSILERO_VAD_ENABLE_LTO=ON \
  -DCMAKE_C_FLAGS_RELEASE="-O3 -flto -mcpu=apple-m4"

cmake --build build-mac-local --config Release
```

Notes:

- Replace `apple-m4` with the actual target CPU if needed, for example `apple-m1`
- this is not a portable release build

### Portable Intel x86_64

Use this for a conservative Intel macOS binary without AVX2 assumptions.

```bash
cmake -S . -B build-mac-x86_64 -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=x86_64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=10.13 \
  -DSILERO_VAD_WEIGHT_SOURCE=jit \
  -DSILERO_VAD_ENABLE_LTO=ON

cmake --build build-mac-x86_64 --config Release
```

Output:

- `build-mac-x86_64/silero_vad.dylib`

### Intel x86_64 AVX2

Use this only when you want a faster Intel-only build and can assume AVX2 support.

```bash
cmake -S . -B build-mac-x86_64-avx2 -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=x86_64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=10.13 \
  -DSILERO_VAD_WEIGHT_SOURCE=jit \
  -DSILERO_VAD_ENABLE_LTO=ON \
  -DSILERO_VAD_ENABLE_AVX2=ON

cmake --build build-mac-x86_64-avx2 --config Release
```

Output:

- `build-mac-x86_64-avx2/silero_vad.dylib`

### Universal macOS build

Use this when you want one binary that supports both Apple Silicon and Intel macOS.

```bash
cmake -S . -B build-mac-universal -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
  -DSILERO_VAD_WEIGHT_SOURCE=jit \
  -DSILERO_VAD_ENABLE_LTO=ON

cmake --build build-mac-universal --config Release
```

Output:

- `build-mac-universal/silero_vad.dylib`

Notes:

- do not enable `SILERO_VAD_ENABLE_AVX2` for a universal build

## Packaging release zips

Use `package_release.py` after building a shared library.

### Windows

```bash
python package_release.py --library build-windows-x64/silero_vad.dll --platform windows --arch x64 --variant default --weight-source jit --output-dir dist
python package_release.py --library build-windows-x64-avx2/silero_vad.dll --platform windows --arch x64 --variant avx2 --weight-source jit --output-dir dist
```

### Linux

```bash
python package_release.py --library build-linux-x86_64/silero_vad.so --platform linux --arch x86_64 --variant default --weight-source jit --output-dir dist
python package_release.py --library build-linux-x86_64-avx2/silero_vad.so --platform linux --arch x86_64 --variant avx2 --weight-source jit --output-dir dist
python package_release.py --library build-linux-arm64/silero_vad.so --platform linux --arch arm64 --variant default --weight-source jit --output-dir dist
python package_release.py --library build-linux-armv7/silero_vad.so --platform linux --arch armv7 --variant default --weight-source jit --output-dir dist
```

### macOS

```bash
python package_release.py --library build-mac-arm64/silero_vad.dylib --platform macos --arch arm64 --variant default --weight-source jit --output-dir dist
python package_release.py --library build-mac-x86_64/silero_vad.dylib --platform macos --arch x86_64 --variant default --weight-source jit --output-dir dist
python package_release.py --library build-mac-x86_64-avx2/silero_vad.dylib --platform macos --arch x86_64 --variant avx2 --weight-source jit --output-dir dist
python package_release.py --library build-mac-universal/silero_vad.dylib --platform macos --arch universal --variant default --weight-source jit --output-dir dist
```

## Quick runtime check

After building, you can test a shared library with:

```bash
python run_silero_vad_clib.py --dll path/to/library --audio LJ001-0001_16k.wav
```
