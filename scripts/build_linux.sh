#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  set -- linux-x86_64 linux-x86_64-avx2 linux-arm64 linux-armv7
fi

python3 "$(dirname "$0")/build_release.py" --package "$@"
