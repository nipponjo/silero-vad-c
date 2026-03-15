from __future__ import annotations

import argparse
import shutil
from pathlib import Path


README_TEMPLATE = """Silero VAD C Binary Package

Package: {package_name}
Platform: {platform}
Architecture: {arch}
Variant: {variant}
Embedded weights: {weight_source}
Library file: {library_name}

Contents:
- {library_name}
- silero_vad.h
- LICENSE

Notes:
- This package contains a prebuilt shared library for the standalone C port of the 16 kHz Silero VAD model.
- The public API is declared in silero_vad.h.
- The library embeds model weights and does not require ONNX.
- Use the AVX2 variant only on CPUs that support AVX2.

Main exported functions:
- silero_vad_get_embedded_weights
- silero_vad_model_create
- silero_vad_model_destroy
- silero_vad_model_reset
- silero_vad_model_forward
- silero_vad_model_forward_audio

Python ctypes note:
- Windows: load silero_vad.dll
- Linux: load silero_vad.so
- macOS: load silero_vad.dylib
"""


def build_package_name(platform: str, arch: str, variant: str) -> str:
    if variant == "default":
        return f"silero-vad-{platform}-{arch}"
    return f"silero-vad-{platform}-{arch}-{variant}"


def write_readme(path: Path,
                 package_name: str,
                 platform: str,
                 arch: str,
                 variant: str,
                 weight_source: str,
                 library_name: str) -> None:
    content = README_TEMPLATE.format(
        package_name=package_name,
        platform=platform,
        arch=arch,
        variant=variant,
        weight_source=weight_source,
        library_name=library_name,
    )
    path.write_text(content, encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True, help="Path to the built shared library.")
    parser.add_argument("--platform", required=True, choices=["windows", "linux", "macos"], help="Target platform label.")
    parser.add_argument("--arch", required=True, help="Target architecture label, for example x64, arm64, x64-avx2.")
    parser.add_argument("--variant", default="default", help="Variant label, for example default or avx2.")
    parser.add_argument("--weight-source", default="jit", choices=["jit", "safetensors"], help="Embedded weight source.")
    parser.add_argument("--output-dir", type=Path, default=Path("dist"), help="Directory where the package folder and zip will be written.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    header = repo_root / "silero_vad.h"
    license_file = repo_root / "LICENSE"

    if not args.library.is_file():
        raise FileNotFoundError(f"Library not found: {args.library}")
    if not header.is_file():
        raise FileNotFoundError(f"Header not found: {header}")
    if not license_file.is_file():
        raise FileNotFoundError(f"License file not found: {license_file}")

    package_name = build_package_name(args.platform, args.arch, args.variant)
    package_dir = args.output_dir / package_name

    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.library, package_dir / args.library.name)
    shutil.copy2(header, package_dir / header.name)
    shutil.copy2(license_file, package_dir / license_file.name)
    write_readme(
        package_dir / "README.txt",
        package_name=package_name,
        platform=args.platform,
        arch=args.arch,
        variant=args.variant,
        weight_source=args.weight_source,
        library_name=args.library.name,
    )

    archive_base = args.output_dir / package_name
    shutil.make_archive(str(archive_base), "zip", root_dir=args.output_dir, base_dir=package_name)

    print(f"Created package folder: {package_dir}")
    print(f"Created zip archive: {archive_base}.zip")


if __name__ == "__main__":
    main()
