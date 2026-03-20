from __future__ import annotations

import argparse
import os
import platform
import re
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable


DEFAULT_REPO = "nipponjo/silero-vad-c"
DEFAULT_TAG = "v6.2-c.0"

ALL_ASSETS = [
    "silero-vad-linux-arm64.zip",
    "silero-vad-linux-armv7.zip",
    "silero-vad-linux-x86_64-avx2.zip",
    "silero-vad-linux-x86_64-sse.zip",
    "silero-vad-linux-x86_64.zip",
    "silero-vad-macos-arm64.zip",
    "silero-vad-macos-x86_64-avx2.zip",
    "silero-vad-macos-x86_64.zip",
    "silero-vad-windows-x64-avx2.zip",
    "silero-vad-windows-x64-sse.zip",
    "silero-vad-windows-x64.zip",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download prebuilt Silero VAD release zip archives from GitHub."
    )
    parser.add_argument(
        "--os",
        choices=["auto", "windows", "macos", "linux", "all"],
        default="auto",
        help="Which platform assets to download. Default: auto.",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"GitHub repo in owner/name form. Default: {DEFAULT_REPO}.",
    )
    parser.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help=f"Release tag to download from. Default: {DEFAULT_TAG}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("downloads"),
        help="Directory where zip files will be saved. Default: downloads.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files that already exist.",
    )
    parser.add_argument(
        "--unzip",
        action="store_true",
        help="Extract downloaded zip files after download.",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("downloads"),
        help="Directory where zip files will be extracted. Default: downloads.",
    )
    return parser.parse_args()


def detect_host_platform() -> str:
    system = platform.system().lower()
    if system.startswith("win"):
        return "windows"
    if system == "darwin":
        return "macos"
    if system == "linux":
        return "linux"
    raise RuntimeError(f"Unsupported host operating system: {platform.system()}")


def detect_host_arch() -> str:
    machine = platform.machine().lower()
    aliases = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "arm64",
        "aarch64": "arm64",
        "armv7l": "armv7",
        "armv7": "armv7",
    }
    return aliases.get(machine, machine)


def print_cpu_features() -> None:
    try:
        import cpuinfo  # type: ignore
    except ImportError:
        print("cpuinfo: not installed, skipping CPU feature report.")
        return

    try:
        info = cpuinfo.get_cpu_info()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"cpuinfo: failed to query CPU info: {exc}")
        return

    flags = {flag.lower() for flag in info.get("flags", [])}
    has_sse = any(flag.startswith("sse") for flag in flags)
    has_avx2 = "avx2" in flags

    print("CPU features:")
    print(f"  SSE active: {'yes' if has_sse else 'no'}")
    print(f"  AVX2 active: {'yes' if has_avx2 else 'no'}")


def assets_for_platform(os_name: str) -> list[str]:
    prefix = f"silero-vad-{os_name}-"
    return [asset for asset in ALL_ASSETS if asset.startswith(prefix)]


def assets_for_host(os_name: str, arch: str) -> list[str]:
    if os_name == "windows":
        if arch == "x86_64":
            return [
                "silero-vad-windows-x64.zip",
                "silero-vad-windows-x64-sse.zip",
                "silero-vad-windows-x64-avx2.zip",
            ]
        return []

    if os_name == "macos":
        if arch == "arm64":
            return ["silero-vad-macos-arm64.zip"]
        if arch == "x86_64":
            return [
                "silero-vad-macos-x86_64.zip",
                "silero-vad-macos-x86_64-avx2.zip",
            ]
        return []

    if os_name == "linux":
        if arch == "x86_64":
            return [
                "silero-vad-linux-x86_64.zip",
                "silero-vad-linux-x86_64-sse.zip",
                "silero-vad-linux-x86_64-avx2.zip",
            ]
        if arch == "arm64":
            return ["silero-vad-linux-arm64.zip"]
        if arch == "armv7":
            return ["silero-vad-linux-armv7.zip"]
        return []

    return []


def release_url(repo: str, tag: str, asset_name: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{asset_name}"


def download_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "silero-vad-c-download-releases/1.0",
            "Accept": "application/octet-stream",
        },
    )
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        output.write(response.read())


def archive_common_root(archive: zipfile.ZipFile) -> str | None:
    roots: set[str] = set()
    for member in archive.infolist():
        parts = Path(member.filename).parts
        if not parts:
            continue
        if parts[0] in {"", ".", ".."}:
            return None
        roots.add(parts[0])
    if len(roots) == 1:
        return next(iter(roots))
    return None


def safe_extract_member(archive: zipfile.ZipFile,
                        member: zipfile.ZipInfo,
                        destination: Path,
                        strip_prefix: str | None) -> None:
    parts = Path(member.filename).parts
    if not parts:
        return

    if strip_prefix is not None:
        if parts[0] != strip_prefix:
            return
        parts = parts[1:]

    if not parts:
        return

    target_path = destination.joinpath(*parts)
    resolved_destination = destination.resolve()
    resolved_target = target_path.resolve()
    if os.path.commonpath([str(resolved_destination), str(resolved_target)]) != str(resolved_destination):
        raise ValueError(f"Refusing to extract outside target dir: {member.filename}")

    if member.is_dir():
        target_path.mkdir(parents=True, exist_ok=True)
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member, "r") as src, target_path.open("wb") as dst:
        dst.write(src.read())


def extract_zip(zip_path: Path, extract_root: Path, force: bool) -> Path:
    target_dir = extract_root / zip_path.stem
    if target_dir.exists():
        if not force:
            print(f"Skipping existing extract dir: {target_dir}")
            return target_dir
    else:
        target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        strip_prefix = None
        common_root = archive_common_root(archive)
        if common_root == zip_path.stem:
            strip_prefix = common_root

        for member in archive.infolist():
            safe_extract_member(archive, member, target_dir, strip_prefix)
    return target_dir


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def selected_assets(os_arg: str) -> list[str]:
    if os_arg == "all":
        return list(ALL_ASSETS)

    if os_arg in {"windows", "macos", "linux"}:
        return assets_for_platform(os_arg)

    host_os = detect_host_platform()
    host_arch = detect_host_arch()
    assets = assets_for_host(host_os, host_arch)
    if assets:
        return assets

    raise RuntimeError(
        f"No release assets configured for auto-detected platform {host_os}/{host_arch}."
    )


def validate_repo(repo: str) -> str:
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", repo):
        raise ValueError(f"Invalid repo value: {repo!r}. Expected owner/name.")
    return repo


def main() -> None:
    args = parse_args()
    repo = validate_repo(args.repo)
    assets = unique_preserve_order(selected_assets(args.os))

    print_cpu_features()
    print(f"Selected release tag: {args.tag}")
    print(f"Selected repo: {repo}")
    print("Assets to download:")
    for asset in assets:
        print(f"  - {asset}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.unzip:
        args.extract_dir.mkdir(parents=True, exist_ok=True)

    failed = False
    for asset in assets:
        destination = args.output_dir / asset
        url = release_url(repo, args.tag, asset)
        downloaded = False

        if destination.exists() and not args.force:
            print(f"Skipping existing file: {destination}")
        else:
            print(f"Downloading: {url}")
            try:
                download_file(url, destination)
            except urllib.error.HTTPError as exc:
                failed = True
                print(f"Failed to download {asset}: HTTP {exc.code}")
                continue
            except urllib.error.URLError as exc:
                failed = True
                print(f"Failed to download {asset}: {exc.reason}")
                continue
            else:
                downloaded = True
                print(f"Saved: {destination}")

        if args.unzip:
            try:
                extracted = extract_zip(destination, args.extract_dir, args.force)
            except zipfile.BadZipFile:
                failed = True
                print(f"Failed to extract {destination}: invalid zip file")
            else:
                if downloaded:
                    print(f"Extracted to: {extracted}")
                else:
                    print(f"Extracted existing zip to: {extracted}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
