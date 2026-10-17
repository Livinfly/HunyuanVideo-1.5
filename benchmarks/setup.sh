#!/usr/bin/env bash
# Usage: bash benchmarks/setup.sh [--without-ssta] [--check]
# UV, BENCH_VENV, PYTHON_VERSION, CUDA_HOME, MAX_JOBS, FLASH_ATTN_PACKAGE may be overridden.
set -euo pipefail
BENCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$BENCH_DIR/.." && pwd)
UV=${UV:-uv}
BENCH_VENV=${BENCH_VENV:-"$REPO_DIR/.venv"}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
export UV_CACHE_DIR=${UV_CACHE_DIR:-"$REPO_DIR/.cache/uv"}
export MAX_JOBS=${MAX_JOBS:-8}
WITH_SSTA=1
CHECK_ONLY=0
for option in "$@"; do
    case "$option" in
        --without-ssta) WITH_SSTA=0 ;;
        --check) CHECK_ONLY=1 ;;
        *) echo "Unknown argument: $option" >&2; exit 2 ;;
    esac
done
if ! command -v "$UV" >/dev/null 2>&1; then
    echo 'Install uv (https://docs.astral.sh/uv/getting-started/installation/) or set UV=/path/to/uv.' >&2
    exit 1
fi
if [[ "$CHECK_ONLY" == 0 ]]; then
    if [[ ! -x "$BENCH_VENV/bin/python" ]]; then
        "$UV" venv --python "$PYTHON_VERSION" "$BENCH_VENV"
    fi
    "$UV" pip install --python "$BENCH_VENV/bin/python" -r "$BENCH_DIR/requirements.lock"
    export PATH="$BENCH_VENV/bin:$PATH"
    # Source package selects its compatible upstream wheel where available. Override
    # FLASH_ATTN_PACKAGE with a wheel URL/path when a platform-specific wheel is needed.
    "$UV" pip install --python "$BENCH_VENV/bin/python" --no-build-isolation \
        "${FLASH_ATTN_PACKAGE:-flash-attn==2.7.4.post1}"
    if [[ "$WITH_SSTA" == 1 ]]; then
        # CUDA_HOME is honored, otherwise torch's extension builder discovers nvcc.
        if [[ -z "${CUDA_HOME:-}" ]] && ! command -v nvcc >/dev/null 2>&1; then
            echo 'SSTA compilation requires a CUDA toolkit: set CUDA_HOME or put nvcc on PATH.' >&2
            exit 1
        fi
        SOURCE_DIR="$REPO_DIR/.cache/sources/flex-block-attn"
        if [[ ! -d "$SOURCE_DIR/.git" ]]; then
            git clone https://github.com/Tencent-Hunyuan/flex-block-attn.git "$SOURCE_DIR"
        fi
        git -C "$SOURCE_DIR" checkout bc2ec1eefcbff089c9bd4b59cfd8cb4db9e90d59
        git -C "$SOURCE_DIR" submodule update --init --recursive
        # Official flex-block-attn pins its supported architecture internally.
        # The benchmark does not infer or substitute an attention backend by GPU name.
        "$UV" pip install --python "$BENCH_VENV/bin/python" --no-build-isolation "$SOURCE_DIR"
    fi
fi
"$UV" pip check --python "$BENCH_VENV/bin/python"
"$BENCH_VENV/bin/python" - "$WITH_SSTA" <<'PY'
import sys
import torch
import diffusers
import transformers
import flash_attn
if int(sys.argv[1]):
    import flex_block_attn
print(f'python={sys.version.split()[0]} torch={torch.__version__} CUDA={torch.version.cuda}')
print(f'diffusers={diffusers.__version__} transformers={transformers.__version__}')
for index in range(torch.cuda.device_count()):
    device = torch.cuda.get_device_properties(index)
    print(f'GPU {index}: {device.name}, capability={device.major}.{device.minor}, GiB={device.total_memory/2**30:.2f}')
PY
mkdir -p "$REPO_DIR/.cache/benchmark"
"$UV" pip freeze --python "$BENCH_VENV/bin/python" > "$REPO_DIR/.cache/benchmark/environment.freeze.txt"
echo "Environment: $BENCH_VENV"
