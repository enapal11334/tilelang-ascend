#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec timeout --signal=TERM --kill-after=15s "${TEST_TIMEOUT:-600}s" \
    torchrun --standalone --nnodes=1 --nproc-per-node=2 benchmark_gemm_allreduce.py "$@"
