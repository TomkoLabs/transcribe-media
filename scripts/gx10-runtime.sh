#!/usr/bin/env bash
# Sourced by install.sh. No system CUDA installation or driver changes.

gx10_preflight() {
    local command_name compiler_version gpu_names
    gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader)" || return 1
    if [[ "$gpu_names" != *GB10* ]]; then
        echo "Automatic ARM CUDA installation currently targets GB10 (GX10/DGX Spark); detected: $gpu_names" >&2
        return 1
    fi
    GX10_CUDA_ROOT="${TRANSCRIBE_CUDA_ROOT:-/usr/local/cuda}"
    for command_name in cmake g++ make git pkg-config; do
        command -v "$command_name" >/dev/null || {
            echo "GX10 needs $command_name; install build-essential cmake git pkg-config libopenblas-dev." >&2
            return 1
        }
    done
    pkg-config --exists openblas || {
        echo "GX10 needs libopenblas-dev for the CTranslate2 source build." >&2
        return 1
    }
    [[ -x "$GX10_CUDA_ROOT/bin/nvcc" ]] || {
        echo "GX10 needs a CUDA toolkit at $GX10_CUDA_ROOT (or set TRANSCRIBE_CUDA_ROOT)." >&2
        return 1
    }
    compiler_version="$("$GX10_CUDA_ROOT/bin/nvcc" --version)"
    if [[ ! "$compiler_version" =~ release[[:space:]]+([0-9]+)\.([0-9]+) ]]; then
        echo "Cannot determine the GX10 CUDA compiler version." >&2
        return 1
    fi
    if (( BASH_REMATCH[1] < 12 || (BASH_REMATCH[1] == 12 && BASH_REMATCH[2] < 8) )); then
        echo "GX10/Blackwell requires CUDA toolkit 12.8 or newer." >&2
        return 1
    fi
}

build_gx10_ctranslate2() {
    local source_dir="$TOOLS_DIR/ctranslate2-4.7.2-source"
    local install_dir="$TOOLS_DIR/ctranslate2-gx10"
    local build_jobs="${TRANSCRIBE_BUILD_JOBS:-8}"
    [[ "$build_jobs" =~ ^[1-9][0-9]*$ ]] || {
        echo "TRANSCRIBE_BUILD_JOBS must be a positive integer." >&2
        return 1
    }
    log "Building pinned CTranslate2 4.7.2 for ARM64/Blackwell (first install takes time)"
    if [[ ! -d "$source_dir" ]]; then
        git clone --branch v4.7.2 --depth 1 --recurse-submodules --shallow-submodules \
            https://github.com/OpenNMT/CTranslate2.git "$source_dir"
    fi
    [[ "$(git -C "$source_dir" describe --tags --exact-match)" == "v4.7.2" ]] || {
        echo "Unexpected CTranslate2 checkout in $source_dir; expected tag v4.7.2." >&2
        return 1
    }
    # sm_120 binaries also execute on GB10's sm_121. This target works with
    # CUDA 12.8/12.9 toolchains as well as CUDA 13, without requiring sm_121 nvcc.
    # Match upstream 4.7.2's CUDA build: its Whisper convolutions do not need
    # WITH_CUDNN. Link to this host's toolkit and retain an explicit runtime path.
    cmake -S "$source_dir" -B "$source_dir/build-gx10" \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_CLI=OFF \
        -DCMAKE_INSTALL_PREFIX="$install_dir" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_INSTALL_RPATH="$install_dir/lib;$GX10_CUDA_ROOT/lib64" \
        -DCUDA_TOOLKIT_ROOT_DIR="$GX10_CUDA_ROOT" \
        -DCUDA_ARCH_LIST=12.0 \
        -DWITH_CUDA=ON -DWITH_CUDNN=OFF \
        -DWITH_MKL=OFF -DWITH_OPENBLAS=ON -DWITH_RUY=ON \
        -DOPENMP_RUNTIME=COMP
    cmake --build "$source_dir/build-gx10" --parallel "$build_jobs"
    cmake --install "$source_dir/build-gx10"
    "$UV" pip install --python "$PYTHON" -r "$source_dir/python/install_requirements.txt"
    CTRANSLATE2_ROOT="$install_dir" \
        CMAKE_BUILD_PARALLEL_LEVEL="$build_jobs" \
        LDFLAGS="-Wl,-rpath,$install_dir/lib -Wl,-rpath,$install_dir/lib64" \
        "$UV" pip install --python "$PYTHON" --no-build-isolation --no-deps \
        --reinstall-package ctranslate2 "$source_dir/python"
    git -C "$source_dir" rev-parse HEAD > "$install_dir/source-revision.txt"
}
