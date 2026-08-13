#!/usr/bin/env bash
# Fetch upstream CudaSift. Its data/ images are what demo.py and the test
# suites run against, and cudaSiftD.cu is what the port is line-referenced to.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d CudaSift ]; then
  echo "CudaSift/ already present"
else
  # full clone, not shallow: the README points at commit 6464e95 for the
  # reverted test suite the ported one is derived from
  git clone --branch Pascal https://github.com/Celebrandil/CudaSift.git
fi
echo "ok: $(ls CudaSift/data | tr '\n' ' ')"
