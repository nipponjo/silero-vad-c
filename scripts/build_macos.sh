#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  set -- mac-arm64 mac-local mac-x86_64 mac-x86_64-avx2 mac-universal
fi

python3 "$(dirname "$0")/build_release.py" --package "$@"
