#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Device waits have no built-in timeout. Kill all torchrun workers as a group.
exec timeout --signal=TERM --kill-after=15s "${TEST_TIMEOUT:-600}s" \
    torchrun --standalone --nnodes=1 --nproc-per-node=2 test_gemm_allreduce.py "$@"
