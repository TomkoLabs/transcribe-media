#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$ROOT/.tools"
VENV_DIR="$ROOT/.venv"
PYTHON="$VENV_DIR/bin/python"
LAUNCHER="$ROOT/transcribe-media"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/transcribe-media"
CONFIG_FILE="$CONFIG_DIR/config.env"
source "$ROOT/scripts/gx10-runtime.sh"

FORCE_CPU=0
SKIP_SYSTEM_PACKAGES=0
SKIP_MODEL_DOWNLOADS=0
NON_INTERACTIVE=0
MODEL_PREP_OPTIONS=()
HF_TOKEN_VALUE="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
export PATH="$TOOLS_DIR:$PATH"

usage() {
    cat <<'EOF'
Usage: ./install.sh [options]

Options:
  --cpu                    Install the CPU PyTorch build even if NVIDIA is found
  --skip-system-packages   Do not invoke the OS package manager
  --skip-model-downloads   Defer weights; run --prepare-models later while online
  --with-emotion           Accepted for compatibility (SpeechBrain is now installed by default)
  --hf-token TOKEN         Save a Hugging Face token non-interactively
  --non-interactive        Never prompt for sudo or a Hugging Face token
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu) FORCE_CPU=1 ;;
        --skip-system-packages) SKIP_SYSTEM_PACKAGES=1 ;;
        --skip-model-downloads) SKIP_MODEL_DOWNLOADS=1 ;;
        --with-emotion) ;;
        --hf-token)
            [[ $# -ge 2 ]] || { echo "--hf-token requires a value" >&2; exit 2; }
            HF_TOKEN_VALUE="$2"
            shift
            ;;
        --non-interactive) NON_INTERACTIVE=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

log() { printf '\n==> %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }

run_privileged() {
    if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1 && [[ $NON_INTERACTIVE -eq 0 ]]; then
        sudo "$@"
    else
        return 1
    fi
}

install_system_packages() {
    [[ $SKIP_SYSTEM_PACKAGES -eq 0 ]] || return 0
    log "Installing required system packages when possible"

    if command -v apt-get >/dev/null 2>&1; then
        run_privileged apt-get update || warn "Could not run apt-get update automatically."
        run_privileged apt-get install -y ffmpeg curl ca-certificates git libsndfile1 xz-utils || \
            warn "Some apt packages could not be installed automatically."
        if [[ $FORCE_CPU -eq 0 && "$(uname -m)" == "aarch64" ]]; then
            run_privileged apt-get install -y build-essential cmake pkg-config libopenblas-dev || \
                warn "GX10 source-build prerequisites could not be installed automatically."
        fi
    elif command -v dnf >/dev/null 2>&1; then
        run_privileged dnf install -y ffmpeg curl ca-certificates git libsndfile xz || \
            warn "Some dnf packages could not be installed. FFmpeg may require RPM Fusion."
    elif command -v yum >/dev/null 2>&1; then
        run_privileged yum install -y ffmpeg curl ca-certificates git libsndfile xz || \
            warn "Some yum packages could not be installed. FFmpeg may require RPM Fusion."
    elif command -v pacman >/dev/null 2>&1; then
        run_privileged pacman -Sy --needed --noconfirm ffmpeg curl ca-certificates git libsndfile xz || \
            warn "Some pacman packages could not be installed automatically."
    elif command -v zypper >/dev/null 2>&1; then
        run_privileged zypper --non-interactive install ffmpeg curl ca-certificates git libsndfile1 xz || \
            warn "Some zypper packages could not be installed automatically."
    else
        warn "No supported package manager found; install FFmpeg and curl manually."
    fi
}

install_uv() {
    if command -v uv >/dev/null 2>&1; then
        UV="$(command -v uv)"
        return
    fi
    if [[ -x "$TOOLS_DIR/uv" ]]; then
        UV="$TOOLS_DIR/uv"
        return
    fi
    command -v curl >/dev/null 2>&1 || { echo "curl is required to install uv." >&2; exit 1; }
    log "Installing the uv Python environment manager locally"
    mkdir -p "$TOOLS_DIR"
    curl -LsSf https://astral.sh/uv/install.sh | \
        env UV_INSTALL_DIR="$TOOLS_DIR" UV_NO_MODIFY_PATH=1 sh
    UV="$TOOLS_DIR/uv"
    [[ -x "$UV" ]] || { echo "uv installation failed." >&2; exit 1; }
}

install_local_ffmpeg() {
    command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1 && return 0
    command -v curl >/dev/null 2>&1 || {
        echo "curl is required for the unprivileged FFmpeg fallback." >&2
        exit 1
    }
    tar --help 2>&1 | grep -q -- "-J" || {
        echo "tar with xz support is required for the private FFmpeg fallback." >&2
        exit 1
    }

    local machine archive_name github_archive
    machine="$(uname -m)"
    case "$machine" in
        x86_64|amd64)
            archive_name="ffmpeg-release-amd64-static.tar.xz"
            github_archive="ffmpeg-master-latest-linux64-gpl.tar.xz"
            ;;
        aarch64|arm64)
            archive_name="ffmpeg-release-arm64-static.tar.xz"
            github_archive="ffmpeg-master-latest-linuxarm64-gpl.tar.xz"
            ;;
        *)
            echo "No local FFmpeg fallback is available for architecture: $machine" >&2
            echo "Install FFmpeg and FFprobe with the system package manager." >&2
            exit 1
            ;;
    esac

    log "Installing a private FFmpeg/FFprobe runtime (no administrator access needed)"
    mkdir -p "$TOOLS_DIR"
    local archive extract_dir ffmpeg_path ffprobe_path
    archive="$TOOLS_DIR/$github_archive"
    extract_dir="$(mktemp -d "$TOOLS_DIR/ffmpeg-extract.XXXXXX")"
    if ! curl -fL --retry 3 --retry-delay 2 \
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/$github_archive" \
        -o "$archive"; then
        warn "Primary FFmpeg mirror failed; trying the secondary mirror."
        archive="$TOOLS_DIR/$archive_name"
        curl -fL --retry 3 --retry-delay 2 \
            "https://johnvansickle.com/ffmpeg/releases/$archive_name" -o "$archive"
    fi
    tar -xJf "$archive" -C "$extract_dir"
    ffmpeg_path="$(find "$extract_dir" -type f -name ffmpeg -print -quit)"
    ffprobe_path="$(find "$extract_dir" -type f -name ffprobe -print -quit)"
    [[ -n "$ffmpeg_path" && -n "$ffprobe_path" ]] || {
        echo "The FFmpeg fallback archive did not contain both executables." >&2
        exit 1
    }
    install -m 0755 "$ffmpeg_path" "$TOOLS_DIR/ffmpeg"
    install -m 0755 "$ffprobe_path" "$TOOLS_DIR/ffprobe"
    rm -rf "$extract_dir"
    rm -f "$archive"
    ffmpeg -version >/dev/null
    ffprobe -version >/dev/null
}

install_python_environment() {
    local gpu_candidate=0
    local gx10=0 torch_index="https://download.pytorch.org/whl/cu126"
    local torch_spec="torch==2.8.0" gpu_requirements="$ROOT/requirements-gpu.txt"
    if [[ $FORCE_CPU -eq 0 ]] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        gpu_candidate=1
        MODEL_PREP_OPTIONS=(--device cuda)
    elif [[ $FORCE_CPU -eq 0 ]]; then
        warn "No working NVIDIA driver detected. Installing CPU mode; for an RTX 3080, fix nvidia-smi before running this installer."
    fi
    if [[ $gpu_candidate -eq 1 && "$(uname -m)" == "aarch64" ]]; then
        gx10=1
        torch_index="https://download.pytorch.org/whl/cu129"
        torch_spec="torch==2.8.0+cu129"
        gpu_requirements="$ROOT/requirements-gx10.txt"
        MODEL_PREP_OPTIONS=(--device cuda)
        # Fail before clearing a working environment if the toolkit is missing.
        gx10_preflight
    fi

    log "Creating a managed Python 3.11 environment"
    "$UV" python install 3.11
    "$UV" venv --python 3.11 --clear "$VENV_DIR"

    if [[ $gpu_candidate -eq 1 ]]; then
        log "NVIDIA GPU detected; installing PyTorch 2.8 from $torch_index"
        if ! "$UV" pip install --python "$PYTHON" \
            --index-url "$torch_index" \
            "$torch_spec" torchvision==0.23.0 torchaudio==2.8.0; then
            echo "CUDA wheels failed to install; the GPU installation is incomplete. Fix the error and rerun ./install.sh (or choose --cpu explicitly)." >&2
            return 1
        fi
    fi

    if [[ $gpu_candidate -eq 0 ]]; then
        log "Installing CPU PyTorch runtime"
        "$UV" pip install --python "$PYTHON" \
            --index-url https://download.pytorch.org/whl/cpu \
            torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
    fi

    if [[ $gpu_candidate -eq 1 ]]; then
        log "Installing transcription, speaker, acoustic, tone, and GPU diarization dependencies"
        # Keep the CUDA wheels selected above as hard resolver constraints. NeMo
        # supports newer Torch releases too, but this project validates 2.8 as a
        # matched torch/vision/audio set and must not silently upgrade one member.
        "$UV" pip install --python "$PYTHON" \
            -r "$ROOT/requirements.txt" \
            -r "$gpu_requirements" \
            "$torch_spec" torchvision==0.23.0 torchaudio==2.8.0
        if [[ $gx10 -eq 1 ]]; then
            build_gx10_ctranslate2
        fi
        if ! "$PYTHON" - <<'PYTEST'
import torch
assert torch.cuda.is_available(), "PyTorch reports CUDA unavailable"
value = (torch.ones(1, device="cuda") * 2).item()
assert value == 2, "CUDA arithmetic test failed"
print(torch.cuda.get_device_name(0))
PYTEST
        then
            echo "CUDA execution failed; check the NVIDIA driver/toolkit and rerun ./install.sh. Use --cpu only if CPU operation is intended." >&2
            return 1
        fi
    else
        log "Installing transcription, speaker, acoustic, and tone dependencies"
        "$UV" pip install --python "$PYTHON" -r "$ROOT/requirements.txt" \
            torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
    fi
}

configure_diarization() {
    if [[ -n "$HF_TOKEN_VALUE" ]]; then
        log "Saving the Hugging Face token"
        "$LAUNCHER" --configure --non-interactive --hf-token "$HF_TOKEN_VALUE" || \
            warn "Token storage failed. Token-free speaker clustering remains available."
        return
    fi

    if [[ -f "$CONFIG_FILE" ]]; then
        log "Using existing diarization configuration: $CONFIG_FILE"
        return
    fi

    if [[ $NON_INTERACTIVE -eq 1 || ! -t 0 ]]; then
        warn "No Hugging Face token supplied; the token-free speaker clustering fallback will be used."
        return
    fi

    log "One-time speaker-detection setup (recommended for quality)"
    "$LAUNCHER" --configure || true
}

install_command() {
    mkdir -p "$HOME/.local/bin"
    ln -sfn "$LAUNCHER" "$HOME/.local/bin/transcribe-media"
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        warn "$HOME/.local/bin is not currently in PATH. Add it or invoke: $LAUNCHER"
    fi
}

main() {
    install_system_packages
    install_uv
    install_local_ffmpeg
    chmod +x "$LAUNCHER" "$ROOT/transcribe_media.py"
    install_python_environment
    mkdir -p "$ROOT/Video Source" "$ROOT/Transcribed" "$ROOT/Review"
    if [[ -s "$ROOT/Review/speaker_registry.json" ]]; then
        warn "Preserving the existing persistent voice registry in $ROOT/Review."
        warn "Git pull and reinstall do not reset ignored Review state or restart VOICE numbering."
        warn "To intentionally start over, run: $LAUNCHER --reset-speaker-registry"
    fi
    configure_diarization
    install_command

    log "Validating the installation"
    "$LAUNCHER" --doctor "${MODEL_PREP_OPTIONS[@]}"

    if [[ $SKIP_MODEL_DOWNLOADS -eq 0 ]]; then
        log "Downloading and validating the default models"
        if ! "$LAUNCHER" --prepare-models "${MODEL_PREP_OPTIONS[@]}"; then
            warn "Model preparation failed. Rerun 'transcribe-media --prepare-models' while online."
            exit 1
        fi
    fi

    cat <<EOF

Installation complete.

Put media in:
  $ROOT/Video Source

Then run:
  $LAUNCHER

Quality mode is enabled by default: English, 2-3 speakers, reviewed voice profiles.
After processing, open this file in your browser to confirm the initial voices:
  $ROOT/Review/speaker-reviews/index.html
Choose Download JSON, then move the file into:
  $ROOT/speaker-decisions
Use Copy apply command on the page and run it from the project folder.
One batch apply updates transcripts and refreshes cached voice matches.

Useful checks:
  $LAUNCHER --doctor
  $LAUNCHER --help
EOF
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main
fi
