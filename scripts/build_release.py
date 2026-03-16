from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


TARGETS: dict[str, dict[str, object]] = {
    "windows-x64": {
        "build_dir": "build-windows-x64-sse",
        "library": "silero_vad.dll",
        "platform": "windows",
        "arch": "x64",
        "variant": "default",
        "weight_source": "jit",
        "configure": [
            "-G", "Visual Studio 18 2026",
            "-A", "x64",
            "-DSILERO_VAD_WEIGHT_SOURCE=jit",
            "-DSILERO_VAD_ENABLE_SSE=ON",
        ],
        "build": ["--config", "Release"],
    },
    "windows-x64-avx2": {
        "build_dir": "build-windows-x64-avx2",
        "library": "silero_vad.dll",
        "platform": "windows",
        "arch": "x64",
        "variant": "avx2",
        "weight_source": "jit",
        "configure": [
            "-G", "Visual Studio 18 2026",
            "-A", "x64",
            "-DSILERO_VAD_WEIGHT_SOURCE=jit",
            "-DSILERO_VAD_ENABLE_AVX2=ON",
        ],
        "build": ["--config", "Release"],
    },
    "linux-x86_64": {
        "build_dir": "build-linux-x86_64-sse",
        "library": "silero_vad.so",
        "platform": "linux",
        "arch": "x86_64",
        "variant": "default",
        "weight_source": "jit",
        "configure": [
            "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DSILERO_VAD_WEIGHT_SOURCE=jit",
            "-DSILERO_VAD_ENABLE_LTO=ON",
            "-DSILERO_VAD_ENABLE_SSE=ON",
            '-DCMAKE_C_FLAGS_RELEASE=-O3 -flto -march=x86-64 -mtune=generic',
        ],
        "build": ["--config", "Release"],
    },
    "linux-x86_64-avx2": {
        "build_dir": "build-linux-x86_64-avx2",
        "library": "silero_vad.so",
        "platform": "linux",
        "arch": "x86_64",
        "variant": "avx2",
        "weight_source": "jit",
        "configure": [
            "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DSILERO_VAD_WEIGHT_SOURCE=jit",
            "-DSILERO_VAD_ENABLE_LTO=ON",
            "-DSILERO_VAD_ENABLE_AVX2=ON",
            '-DCMAKE_C_FLAGS_RELEASE=-O3 -flto -march=x86-64 -mtune=generic',
        ],
        "build": ["--config", "Release"],
    },
    "linux-arm64": {
        "build_dir": "build-linux-arm64",
        "library": "silero_vad.so",
        "platform": "linux",
        "arch": "arm64",
        "variant": "default",
        "weight_source": "jit",
        "configure": [
            "-G", "Ninja",
            "-DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/linux-aarch64.cmake",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DSILERO_VAD_WEIGHT_SOURCE=jit",
            "-DSILERO_VAD_ENABLE_LTO=ON",
            "-DSILERO_VAD_ENABLE_NEON=ON",
            '-DCMAKE_C_FLAGS_RELEASE=-O3 -flto -march=armv8-a',
        ],
        "build": ["--config", "Release"],
    },
    "linux-armv7": {
        "build_dir": "build-linux-armv7",
        "library": "silero_vad.so",
        "platform": "linux",
        "arch": "armv7",
        "variant": "default",
        "weight_source": "jit",
        "configure": [
            "-G", "Ninja",
            "-DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/linux-armv7.cmake",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DSILERO_VAD_WEIGHT_SOURCE=jit",
            "-DSILERO_VAD_ENABLE_LTO=ON",
            '-DCMAKE_C_FLAGS_RELEASE=-O3 -flto -march=armv7-a -mfloat-abi=hard -mfpu=vfpv3-d16 -mtune=generic-armv7-a',
        ],
        "build": ["--config", "Release"],
    },
    "mac-arm64": {
        "build_dir": "build-mac-arm64",
        "library": "silero_vad.dylib",
        "platform": "macos",
        "arch": "arm64",
        "variant": "default",
        "weight_source": "jit",
        "configure": [
            "-G", "Unix Makefiles",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_OSX_ARCHITECTURES=arm64",
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0",
            "-DSILERO_VAD_WEIGHT_SOURCE=jit",
            "-DSILERO_VAD_ENABLE_NEON=ON",
            "-DSILERO_VAD_ENABLE_LTO=ON",
        ],
        "build": ["--config", "Release"],
    },
    "mac-local": {
        "build_dir": "build-mac-local",
        "library": "silero_vad.dylib",
        "platform": "macos",
        "arch": "arm64",
        "variant": "local",
        "weight_source": "jit",
        "configure": [
            "-G", "Unix Makefiles",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_OSX_ARCHITECTURES=arm64",
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0",
            "-DSILERO_VAD_WEIGHT_SOURCE=jit",
            "-DSILERO_VAD_ENABLE_NEON=ON",
            "-DSILERO_VAD_ENABLE_LTO=ON",
            '-DCMAKE_C_FLAGS_RELEASE=-O3 -flto -mcpu=apple-m4',
        ],
        "build": ["--config", "Release"],
    },
    "mac-x86_64": {
        "build_dir": "build-mac-x86_64-sse",
        "library": "silero_vad.dylib",
        "platform": "macos",
        "arch": "x86_64",
        "variant": "default",
        "weight_source": "jit",
        "configure": [
            "-G", "Unix Makefiles",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_OSX_ARCHITECTURES=x86_64",
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=10.13",
            "-DSILERO_VAD_WEIGHT_SOURCE=jit",
            "-DSILERO_VAD_ENABLE_LTO=ON",
            "-DSILERO_VAD_ENABLE_SSE=ON",
        ],
        "build": ["--config", "Release"],
    },
    "mac-x86_64-avx2": {
        "build_dir": "build-mac-x86_64-avx2",
        "library": "silero_vad.dylib",
        "platform": "macos",
        "arch": "x86_64",
        "variant": "avx2",
        "weight_source": "jit",
        "configure": [
            "-G", "Unix Makefiles",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_OSX_ARCHITECTURES=x86_64",
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=10.13",
            "-DSILERO_VAD_WEIGHT_SOURCE=jit",
            "-DSILERO_VAD_ENABLE_LTO=ON",
            "-DSILERO_VAD_ENABLE_AVX2=ON",
        ],
        "build": ["--config", "Release"],
    },
    "mac-universal": {
        "build_dir": "build-mac-universal",
        "library": "silero_vad.dylib",
        "platform": "macos",
        "arch": "universal",
        "variant": "default",
        "weight_source": "jit",
        "configure": [
            "-G", "Unix Makefiles",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64",
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0",
            "-DSILERO_VAD_WEIGHT_SOURCE=jit",
            "-DSILERO_VAD_ENABLE_LTO=ON",
        ],
        "build": ["--config", "Release"],
    },
}


def run(cmd: list[str]) -> None:
    print("+", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="*", help="Build target names.")
    parser.add_argument("--list", action="store_true", help="List available targets and exit.")
    parser.add_argument("--cmake", default="cmake", help="CMake executable to use.")
    parser.add_argument("--python", default=sys.executable, help="Python executable passed to CMake.")
    parser.add_argument("--clean", action="store_true", help="Remove the build directory before configuring.")
    parser.add_argument("--package", action="store_true", help="Package each built target into a zip archive.")
    parser.add_argument("--dist-dir", default="dist", help="Output directory used when packaging.")
    args = parser.parse_args()

    if args.list:
        for name in TARGETS:
            print(name)
        return

    if not args.targets:
        parser.error("Provide at least one target name or use --list.")

    for target in args.targets:
        if target not in TARGETS:
            raise SystemExit(f"Unknown target: {target}")

        config = TARGETS[target]
        build_dir = ROOT / str(config["build_dir"])

        if args.clean and build_dir.exists():
            print(f"Removing {build_dir}")
            import shutil
            shutil.rmtree(build_dir)

        configure_cmd = [
            args.cmake,
            "-S", ".",
            "-B", str(config["build_dir"]),
            f"-DPython3_EXECUTABLE={args.python}",
            *config["configure"],
        ]
        build_cmd = [
            args.cmake,
            "--build", str(config["build_dir"]),
            *config["build"],
        ]

        run(configure_cmd)
        run(build_cmd)

        if args.package:
            package_cmd = [
                args.python,
                "package_release.py",
                "--library", f"{config['build_dir']}/{config['library']}",
                "--platform", str(config["platform"]),
                "--arch", str(config["arch"]),
                "--variant", str(config["variant"]),
                "--weight-source", str(config["weight_source"]),
                "--output-dir", args.dist_dir,
            ]
            run(package_cmd)


if __name__ == "__main__":
    main()
