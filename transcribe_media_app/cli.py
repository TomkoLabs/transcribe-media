from __future__ import annotations

import argparse
import fcntl
import getpass
import importlib.util
import logging
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
import wave
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from . import __version__
from .analysis import (
    AcousticAnalyzer,
    HeuristicToneEstimator,
    build_turns,
    normalize_segments,
)
from .backends import (
    SUPPORTED_CTRANSLATE2_VERSION,
    Emotion2VecToneEstimator,
    PyannoteDiarizer,
    SpeakerIdentityEncoder,
    SpeechBrainEmbeddingDiarizer,
    SpeechBrainToneEstimator,
    WhisperXBackend,
    package_version,
    release_accelerator_memory,
)
from .renderers import write_outputs
from .schema import (
    RESULT_SCHEMA_VERSION,
    SAMPLE_RATE,
    ProcessingResult,
    ProcessingSettings,
    ProjectPaths,
    RuntimeSettings,
)
from .speakers import (
    DEFAULT_ENROLLMENT_SECONDS,
    DEFAULT_LOCAL_MERGE_THRESHOLD,
    DEFAULT_MATCH_MARGIN,
    DEFAULT_MATCH_THRESHOLD,
    VoiceRegistry,
    apply_speaker_identities,
    overlapping_speaker_pairs,
)
from .storage import (
    ManifestStore,
    discover_media,
    expected_outputs,
    ffprobe_duration,
    parse_extensions,
    source_fingerprint,
    stable_hash,
    state_is_complete,
    utc_now,
)

# Disable optional dependency telemetry before WhisperX imports pyannote. Model
# downloads still work, but normal processing emits no usage traces.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / f"transcribe-media-matplotlib-{os.getuid()}"),
)
warnings.filterwarnings(
    "ignore",
    message="(?s).*torchcodec is not installed correctly.*",
    category=UserWarning,
    module=r"pyannote\.audio\.core\.io",
)
warnings.filterwarnings(
    "ignore",
    message=r"torchaudio\._backend\.list_audio_backends has been deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"(?s)TensorFloat-32 \(TF32\) has been disabled.*",
    category=Warning,
    module=r"pyannote\.audio\.utils\.reproducibility",
)
warnings.filterwarnings(
    "ignore",
    message=(
        r"Module 'speechbrain\.lobes\.models\.huggingface_transformers' "
        r"was deprecated.*"
    ),
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=(
        r"Passing `gradient_checkpointing` to a config initialization is "
        r"deprecated.*"
    ),
    category=UserWarning,
)

LOG = logging.getLogger("transcribe-media")
CONFIG_DIR = (
    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    / "transcribe-media"
)
CONFIG_FILE = CONFIG_DIR / "config.env"
VALID_REVIEW_FORMATS = {"json", "srt", "vtt"}


def parse_review_formats(raw: str) -> tuple[str, ...]:
    formats: list[str] = []
    for value in raw.split(","):
        value = value.strip().lower()
        if not value:
            continue
        if value not in VALID_REVIEW_FORMATS:
            raise argparse.ArgumentTypeError(
                f"unsupported review format {value!r}; use json, srt, and/or vtt"
            )
        if value not in formats:
            formats.append(value)
    if "json" not in formats:
        formats.insert(0, "json")
    return tuple(formats)


def project_root() -> Path:
    configured = os.environ.get("TRANSCRIBE_MEDIA_ROOT")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else Path(__file__).resolve().parents[1]
    )


def resolve_paths(args: argparse.Namespace) -> ProjectPaths:
    root = project_root()
    source = (
        Path(args.source_dir).expanduser() if args.source_dir else root / "Video Source"
    )
    transcript = (
        Path(args.transcript_dir).expanduser()
        if args.transcript_dir
        else root / "Transcribed"
    )
    review = Path(args.review_dir).expanduser() if args.review_dir else root / "Review"
    return ProjectPaths(
        root=root,
        source_dir=source.resolve(),
        transcript_dir=transcript.resolve(),
        review_dir=review.resolve(),
    )


def archive_speaker_state(review_dir: Path) -> Path | None:
    """Move identity/manifest state to a private, recoverable backup directory."""
    registry_path = review_dir / "speaker_registry.json"
    manifest_path = review_dir / "transcription_manifest.json"
    state_paths = [path for path in (registry_path, manifest_path) if path.exists()]
    if not state_paths:
        return None

    lock_path = review_dir / ".speaker_registry.json.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        # Recheck after acquiring the same lock used by registry updates.
        state_paths = [path for path in (registry_path, manifest_path) if path.exists()]
        if not state_paths:
            return None
        backup_root = review_dir / "speaker-registry-backups"
        backup_root.mkdir(parents=True, exist_ok=True)
        backup_root.chmod(0o700)
        timestamp = "".join(character for character in utc_now() if character.isalnum())
        backup_dir = backup_root / timestamp
        suffix = 1
        while backup_dir.exists():
            backup_dir = backup_root / f"{timestamp}-{suffix}"
            suffix += 1
        backup_dir.mkdir(mode=0o700)
        for path in state_paths:
            path.replace(backup_dir / path.name)
        return backup_dir
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe-media",
        description=(
            "Private, local-first batch transcription with durable rerun state."
        ),
    )
    parser.add_argument(
        "source_dir", nargs="?", help="media directory (default: ./Video Source)"
    )
    parser.add_argument(
        "--transcript-dir", help="primary TXT directory (default: ./Transcribed)"
    )
    parser.add_argument(
        "--review-dir", help="JSON/subtitle/manifest directory (default: ./Review)"
    )
    parser.add_argument(
        "--recursive", action="store_true", help="scan source subdirectories"
    )
    parser.add_argument("--extensions", help="comma-separated extension allow-list")

    parser.add_argument("--model", default="large-v3", help="Whisper model name")
    parser.add_argument(
        "--language", help="spoken language code; default is automatic detection"
    )
    parser.add_argument(
        "--task",
        choices=("transcribe", "translate"),
        default="transcribe",
        help="preserve the spoken language or translate speech to English",
    )
    parser.add_argument(
        "--no-align", dest="align", action="store_false", help="disable word alignment"
    )
    parser.set_defaults(align=True)

    parser.add_argument(
        "--diarization-backend",
        choices=("auto", "pyannote", "speechbrain", "off"),
        default="auto",
        help="anonymous speaker-label backend",
    )
    parser.add_argument(
        "--diarize", action="store_true", help="enable automatic speaker labeling"
    )
    parser.add_argument(
        "--no-diarize", action="store_true", help="disable speaker labeling"
    )
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument(
        "--speakers",
        type=int,
        help="exact number of speakers expected in each recording",
    )
    parser.add_argument("--hf-token", help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-speaker-identity",
        dest="speaker_identity",
        action="store_false",
        help="disable automatic anonymous voice matching across recordings",
    )
    parser.set_defaults(speaker_identity=True)
    parser.add_argument(
        "--no-speaker-refinement",
        dest="speaker_refinement",
        action="store_false",
        help="disable conservative acoustic correction of short speaker-label flips",
    )
    parser.set_defaults(speaker_refinement=True)
    parser.add_argument(
        "--reset-speaker-registry",
        action="store_true",
        help=(
            "archive persistent voice/manifest state and rebuild identities from "
            "all current source media"
        ),
    )
    parser.add_argument(
        "--speaker-match-threshold",
        type=float,
        default=DEFAULT_MATCH_THRESHOLD,
        help="advanced: minimum cosine similarity for a persistent voice match",
    )
    parser.add_argument(
        "--speaker-match-margin",
        type=float,
        default=DEFAULT_MATCH_MARGIN,
        help="advanced: minimum lead over the next persistent voice candidate",
    )
    parser.add_argument(
        "--speaker-enrollment-seconds",
        type=float,
        default=DEFAULT_ENROLLMENT_SECONDS,
        help="clean speech required before automatically creating a voice profile",
    )
    parser.add_argument(
        "--speaker-merge-threshold",
        type=float,
        default=DEFAULT_LOCAL_MERGE_THRESHOLD,
        help="advanced: similarity required to reconcile split local clusters",
    )

    parser.add_argument(
        "--no-acoustic",
        dest="acoustic",
        action="store_false",
        help="disable measured acoustic observations",
    )
    parser.set_defaults(acoustic=True)
    parser.add_argument(
        "--tone-backend",
        choices=("auto", "emotion2vec", "speechbrain", "acoustic", "off"),
        default="auto",
        help="approximate turn-level tone estimator",
    )
    parser.add_argument(
        "--no-tone", action="store_true", help="disable approximate tone estimates"
    )
    parser.add_argument(
        "--review-formats",
        type=parse_review_formats,
        default=("json",),
        help="comma-separated JSON/SRT/VTT review artifacts (JSON is always included)",
    )

    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--compute-type",
        help="CTranslate2 compute type; chosen automatically by default",
    )
    parser.add_argument(
        "--batch-size", type=int, help="ASR batch size; chosen automatically by default"
    )
    parser.add_argument(
        "--analysis-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="speaker/tone device; isolated from ASR automatically on smaller GPUs",
    )
    parser.add_argument(
        "--diarization-batch-size",
        type=int,
        help="pyannote internal batch size; chosen conservatively by default",
    )
    parser.add_argument("--threads", type=int, help="CPU inference threads")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="reprocess even complete unchanged files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show planned work without loading models",
    )
    parser.add_argument("--verbose", action="store_true")

    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--doctor", action="store_true", help="run installation self-checks"
    )
    modes.add_argument(
        "--prepare-models",
        action="store_true",
        help="download and exercise configured models",
    )
    modes.add_argument(
        "--configure",
        action="store_true",
        help="save optional Hugging Face access token",
    )
    parser.add_argument(
        "--non-interactive", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for option in (
        "min_speakers",
        "max_speakers",
        "speakers",
        "batch_size",
        "diarization_batch_size",
        "threads",
    ):
        value = getattr(args, option)
        if value is not None and value < 1:
            parser.error(f"--{option.replace('_', '-')} must be at least 1")
    if args.speakers is not None:
        if args.min_speakers is not None or args.max_speakers is not None:
            parser.error(
                "--speakers cannot be combined with --min-speakers or --max-speakers"
            )
        args.min_speakers = args.speakers
        args.max_speakers = args.speakers
    if (
        args.min_speakers is not None
        and args.max_speakers is not None
        and args.min_speakers > args.max_speakers
    ):
        parser.error("--min-speakers cannot exceed --max-speakers")
    if args.diarize and args.no_diarize:
        parser.error("--diarize and --no-diarize cannot be combined")
    if args.no_diarize:
        args.diarization_backend = "off"
    elif args.diarize and args.diarization_backend == "off":
        parser.error("--diarize cannot be combined with --diarization-backend off")
    elif args.diarize:
        args.diarization_backend = "auto"
    if args.diarization_backend == "off":
        args.speaker_identity = False
    if args.reset_speaker_registry and not args.speaker_identity:
        parser.error("--reset-speaker-registry requires speaker identity")
    if args.reset_speaker_registry and args.dry_run:
        parser.error("--reset-speaker-registry cannot be combined with --dry-run")
    if args.reset_speaker_registry and (
        args.configure or args.doctor or args.prepare_models
    ):
        parser.error(
            "--reset-speaker-registry cannot be combined with a maintenance mode"
        )
    for option in (
        "speaker_match_threshold",
        "speaker_match_margin",
        "speaker_merge_threshold",
    ):
        value = getattr(args, option)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{option.replace('_', '-')} must be between 0 and 1")
    if args.speaker_enrollment_seconds < 1.0:
        parser.error("--speaker-enrollment-seconds must be at least 1")
    if args.no_tone:
        args.tone_backend = "off"
    if not args.acoustic and args.tone_backend == "acoustic":
        parser.error("--tone-backend acoustic requires acoustic analysis")


def save_hf_token(token: str, path: Path = CONFIG_FILE) -> None:
    token = token.strip()
    if not token:
        raise ValueError("token cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(f"HF_TOKEN={shlex.quote(token)}\n", encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def load_saved_hf_token(path: Path = CONFIG_FILE) -> Optional[str]:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                values = shlex.split(line[len("HF_TOKEN=") :])
                return values[0] if values else None
    except (OSError, UnicodeError, ValueError):
        return None
    return None


def resolve_hf_token(explicit: Optional[str] = None) -> Optional[str]:
    return (
        explicit
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or load_saved_hf_token()
    )


def configure(args: argparse.Namespace) -> int:
    token = resolve_hf_token(args.hf_token)
    if not token and not args.non_interactive and sys.stdin.isatty():
        print("A Hugging Face token enables the higher-quality pyannote diarizer.")
        print("Leave blank to keep the token-free SpeechBrain fallback.")
        token = getpass.getpass("Hugging Face read token: ").strip() or None
    if not token:
        print("No token saved. Token-free speaker clustering remains available.")
        return 0
    save_hf_token(token)
    print(f"Token saved privately to {CONFIG_FILE} (mode 0600).")
    return 0


def resolve_runtime(args: argparse.Namespace) -> RuntimeSettings:
    threads = args.threads or max(1, min(os.cpu_count() or 1, 8))
    requested = args.device
    device = requested
    if requested == "auto":
        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
        except ImportError:
            pass
    compute_type = args.compute_type or ("float16" if device == "cuda" else "int8")
    batch_size = args.batch_size or (8 if device == "cuda" else 1)
    analysis_device = "cpu"
    gpu_memory_gib: Optional[float] = None
    if device == "cuda":
        if args.analysis_device == "cuda":
            analysis_device = "cuda"
        elif args.analysis_device == "auto":
            try:
                import torch

                gpu_memory_gib = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                # large-v3, alignment, pyannote, and emotion2vec do not reliably
                # coexist on 8-12 GiB cards. Keep CUDA for ASR/alignment and run
                # the smaller analysis stages on CPU; this does not reduce model
                # accuracy and follows the product's accuracy-over-speed priority.
                analysis_device = "cuda" if gpu_memory_gib >= 16.0 else "cpu"
            except Exception:
                analysis_device = "cpu"
    diarization_batch_size = args.diarization_batch_size or 4
    analysis_detail = f"speaker/tone {analysis_device.upper()}"
    if gpu_memory_gib is not None and analysis_device == "cpu":
        analysis_detail += f" ({gpu_memory_gib:.1f} GiB GPU safeguard)"
    return RuntimeSettings(
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        threads=threads,
        device_was_auto=requested == "auto",
        description=(
            f"{device.upper()} / {compute_type}, batch {batch_size}, "
            f"{threads} thread(s); {analysis_detail}"
        ),
        analysis_device=analysis_device,
        diarization_batch_size=diarization_batch_size,
    )


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def resolve_analysis_backends(
    args: argparse.Namespace,
    token: Optional[str],
) -> tuple[str, str, list[str]]:
    degraded: list[str] = []
    diarization = args.diarization_backend
    if diarization == "auto":
        if token:
            diarization = "pyannote"
        elif _available("speechbrain"):
            diarization = "speechbrain"
            degraded.append(
                "diarization: using token-free windowed voice clustering; overlapping "
                "speech "
                "is less accurately separated than pyannote"
            )
        else:
            diarization = "off"
            degraded.append(
                "diarization: unavailable (no token and SpeechBrain is not installed)"
            )

    tone = args.tone_backend
    if tone == "auto":
        if _available("funasr"):
            tone = "emotion2vec"
        elif _available("speechbrain"):
            tone = "speechbrain"
            degraded.append(
                "tone: emotion2vec unavailable; using the smaller four-label "
                "SpeechBrain IEMOCAP model"
            )
        elif args.acoustic:
            tone = "acoustic"
            degraded.append(
                "tone: SpeechBrain unavailable; using low-confidence acoustic heuristic"
            )
        else:
            tone = "off"
            degraded.append("tone: unavailable because no estimator is installed")
    return diarization, tone, degraded


def processing_settings(
    args: argparse.Namespace,
    diarization_backend: str,
    tone_backend: str,
) -> ProcessingSettings:
    return ProcessingSettings(
        model=args.model,
        language=args.language,
        task=args.task,
        align=args.align,
        diarization_backend=diarization_backend,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        speaker_identity=args.speaker_identity and diarization_backend != "off",
        speaker_refinement=args.speaker_refinement and diarization_backend != "off",
        speaker_match_threshold=args.speaker_match_threshold,
        speaker_match_margin=args.speaker_match_margin,
        speaker_merge_threshold=args.speaker_merge_threshold,
        speaker_enrollment_seconds=args.speaker_enrollment_seconds,
        acoustic_analysis=args.acoustic,
        tone_backend=tone_backend,
        review_formats=tuple(args.review_formats),
    )


def settings_identity(
    settings: ProcessingSettings,
    runtime: RuntimeSettings,
) -> dict[str, Any]:
    return {
        "program_version": __version__,
        "processing": settings.fingerprint_payload(),
        "runtime": asdict(runtime),
        "dependency_versions": {
            "whisperx": package_version("whisperx"),
            "faster_whisper": package_version("faster-whisper"),
            "ctranslate2": package_version("ctranslate2"),
            "pyannote_audio": package_version("pyannote-audio"),
            "speechbrain": package_version("speechbrain"),
            "funasr": package_version("funasr"),
            "torch": package_version("torch"),
        },
    }


def create_diarizer(
    name: str,
    token: Optional[str],
    device: str,
    batch_size: int = 4,
) -> Any:
    if name == "off":
        return None
    if name == "pyannote":
        if not token:
            raise RuntimeError("pyannote diarization requires a Hugging Face token")
        return PyannoteDiarizer(token, device, batch_size)
    if name == "speechbrain":
        return SpeechBrainEmbeddingDiarizer(device)
    raise ValueError(f"unsupported diarization backend: {name}")


def create_tone_estimator(name: str, device: str) -> Any:
    if name == "off":
        return None
    if name == "emotion2vec":
        return Emotion2VecToneEstimator(device)
    if name == "speechbrain":
        return SpeechBrainToneEstimator(device)
    if name == "acoustic":
        return HeuristicToneEstimator()
    raise ValueError(f"unsupported tone backend: {name}")


def _doctor_check(label: str, function: Any) -> bool:
    try:
        detail = function()
        print(f"[OK]   {label}: {detail}")
        return True
    except Exception as exc:
        print(f"[FAIL] {label}: {exc}")
        return False


def _ctranslate2_check() -> str:
    version = package_version("ctranslate2")
    if not version:
        raise RuntimeError("not installed")

    import torch

    if not torch.cuda.is_available():
        return f"{version}; CPU runtime"
    if version != SUPPORTED_CTRANSLATE2_VERSION:
        raise RuntimeError(
            f"{version} is installed; CUDA requires the project-pinned "
            f"{SUPPORTED_CTRANSLATE2_VERSION} release. Run ./install.sh."
        )

    import ctranslate2

    device_count = ctranslate2.get_cuda_device_count()
    if device_count < 1:
        raise RuntimeError(
            "PyTorch sees CUDA, but CTranslate2 sees no CUDA device. "
            "Run ./install.sh to repair the GPU runtime."
        )
    compute_types = ctranslate2.get_supported_compute_types("cuda", 0)
    if "float16" not in compute_types:
        raise RuntimeError("CUDA device 0 does not report float16 support")
    return (
        f"{version}; {device_count} visible CUDA device(s); device 0 supports float16"
    )


def doctor() -> int:
    print(f"transcribe-media {__version__} installation check")
    checks = [
        _doctor_check("Python", lambda: platform.python_version()),
        _doctor_check(
            "FFmpeg",
            lambda: subprocess.check_output(
                ["ffmpeg", "-version"], text=True
            ).splitlines()[0],
        ),
        _doctor_check(
            "FFprobe",
            lambda: subprocess.check_output(
                ["ffprobe", "-version"], text=True
            ).splitlines()[0],
        ),
        _doctor_check(
            "WhisperX",
            lambda: (
                package_version("whisperx")
                or (_ for _ in ()).throw(RuntimeError("not installed"))
            ),
        ),
        _doctor_check("CTranslate2", _ctranslate2_check),
        _doctor_check(
            "SpeechBrain",
            lambda: (
                package_version("speechbrain")
                or (_ for _ in ()).throw(RuntimeError("not installed"))
            ),
        ),
        _doctor_check(
            "FunASR / emotion2vec runtime",
            lambda: (
                package_version("funasr")
                or (_ for _ in ()).throw(RuntimeError("not installed"))
            ),
        ),
    ]

    def torch_check() -> str:
        import torch

        value = (torch.ones(2) + 1).sum().item()
        if value != 4:
            raise RuntimeError("tensor arithmetic failed")
        return f"{torch.__version__}; CUDA available={torch.cuda.is_available()}"

    checks.append(_doctor_check("PyTorch arithmetic", torch_check))
    token = resolve_hf_token()
    token_status = "configured" if token else "not configured"
    print(f"[INFO] Higher-quality pyannote token: {token_status}")
    runtime = resolve_runtime(build_parser().parse_args([]))
    print(f"[INFO] Selected runtime: {runtime.description}")
    print("[INFO] Media is processed locally; no runtime upload service is used.")
    return 0 if all(checks) else 1


def _write_silence(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(b"\0\0" * SAMPLE_RATE)


def _exercise_models(
    args: argparse.Namespace,
    token: Optional[str],
    runtime: RuntimeSettings,
    diarization_name: str,
    tone_name: str,
    backend: WhisperXBackend,
) -> None:
    with tempfile.TemporaryDirectory(prefix="transcribe-media-selftest-") as directory:
        sample = Path(directory) / "one-second-silence.wav"
        _write_silence(sample)
        result, audio, provenance, degraded = backend.transcribe(sample, False, False)
        if not isinstance(result.get("segments"), list):
            raise RuntimeError("ASR self-test returned an invalid result")
        print(f"[OK] ASR decoded a local WAV ({provenance['transcription_engine']}).")
        for message in degraded:
            print(f"[INFO] {message}")
        if args.align and args.task != "translate":
            backend._alignment_model(args.language or result.get("language") or "en")
            print("[OK] Alignment model loaded.")
        synthetic_result = {
            "language": args.language or "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "self test",
                    "words": [
                        {"word": "self", "start": 0.0, "end": 0.4, "score": 0.5},
                        {"word": "test", "start": 0.5, "end": 0.9, "score": 0.5},
                    ],
                }
            ],
        }
        turns = build_turns(normalize_segments(synthetic_result))
        AcousticAnalyzer().analyze(audio, turns)
        print("[OK] Acoustic observation stage exercised.")
        diarizer = None
        if diarization_name != "off":
            diarizer = create_diarizer(
                diarization_name,
                token,
                runtime.analysis_device,
                runtime.diarization_batch_size,
            )
            diarizer.assign(sample, audio, synthetic_result, 1, 1)
            print(f"[OK] Diarization model exercised ({diarization_name}).")
        if (args.speaker_identity or args.speaker_refinement) and diarizer is not None:
            import numpy as np
            import torch

            identity_encoder = SpeakerIdentityEncoder(
                runtime.analysis_device,
                classifier=getattr(diarizer, "classifier", None),
            )
            samples = np.sin(
                np.linspace(0.0, 100.0, SAMPLE_RATE, dtype=np.float32)
            ).astype(np.float32)
            waveform = (
                torch.from_numpy(samples).unsqueeze(0).to(runtime.analysis_device)
            )
            with torch.inference_mode():
                embedding = identity_encoder.classifier.encode_batch(waveform)
            if not embedding.numel():
                raise RuntimeError("speaker identity self-test returned no embedding")
            print("[OK] Speaker refinement/identity encoder exercised.")
        if tone_name != "off":
            estimator = create_tone_estimator(tone_name, runtime.analysis_device)
            estimator.estimate(audio, turns)
            tone_results = [turn.get("tone") for turn in turns if turn.get("tone")]
            if not tone_results:
                raise RuntimeError("tone self-test produced no turn result")
            unavailable = [
                item for item in tone_results if item.get("kind") == "unavailable"
            ]
            if unavailable:
                raise RuntimeError(
                    "tone inference failed: "
                    f"{unavailable[0].get('error', 'unknown error')}"
                )
            if not all(item.get("scores") for item in tone_results):
                raise RuntimeError("tone self-test produced no confidence scores")
            print(f"[OK] Tone estimator exercised ({tone_name}).")


def prepare_models(args: argparse.Namespace) -> int:
    token = resolve_hf_token(args.hf_token)
    runtime = resolve_runtime(args)
    diarization_name, tone_name, messages = resolve_analysis_backends(args, token)
    for message in messages:
        print(f"[INFO] {message}")
    print(f"Loading ASR model {args.model!r} on {runtime.description} ...")
    backend, runtime = WhisperXBackend.create(
        args.model, args.language, args.task, runtime, token
    )
    try:
        _exercise_models(args, token, runtime, diarization_name, tone_name, backend)
    finally:
        backend.release_audio()
    print("Model preparation and local inference self-test completed.")
    return 0


def _safe_error(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {message}"[:2000]


def _validate_speaker_result(
    result: dict[str, Any],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> dict[str, Any]:
    segments = result.get("segments") or []
    if not segments:
        return {"detected_speakers": 0, "labeled_segments": 0}

    timeline = result.get("speaker_timeline") or []
    labels = {
        str(item.get("speaker"))
        for item in timeline
        if item.get("speaker") not in (None, "", "SPEAKER_UNKNOWN")
    }
    labeled_segments = sum(
        1
        for segment in segments
        if segment.get("speaker") not in (None, "", "SPEAKER_UNKNOWN")
    )
    if not timeline or not labels or not labeled_segments:
        raise RuntimeError(
            "diarization returned no usable speaker timeline or segment labels"
        )
    if min_speakers is not None and len(labels) < min_speakers:
        raise RuntimeError(
            f"diarization detected {len(labels)} speaker(s), below requested "
            f"minimum {min_speakers}"
        )
    if max_speakers is not None and len(labels) > max_speakers:
        raise RuntimeError(
            f"diarization detected {len(labels)} speaker(s), above requested "
            f"maximum {max_speakers}"
        )
    return {
        "detected_speakers": len(labels),
        "labeled_segments": labeled_segments,
        "total_segments": len(segments),
        "timeline_intervals": len(timeline),
    }


def _processing_payload(
    source: Path,
    relative: Path,
    fingerprint: dict[str, Any],
    settings: ProcessingSettings,
    settings_hash: str,
    runtime: RuntimeSettings,
    result: dict[str, Any],
    turns: list[dict[str, Any]],
    provenance: dict[str, Any],
    degraded: list[str],
    started_utc: str,
    elapsed: float,
) -> dict[str, Any]:
    duration = ffprobe_duration(source)
    if duration is None:
        duration = max(
            (float(item.get("end") or 0.0) for item in result.get("segments") or []),
            default=0.0,
        )
    detected = result.get("language") or settings.language
    output_language = "en" if settings.task == "translate" else detected
    segments = normalize_segments(result)
    overlap_events = []
    seen: set[tuple[str, str]] = set()
    for turn in turns:
        for overlap in turn.get("overlaps") or []:
            key = tuple(sorted((turn["id"], overlap["turn_id"])))
            if key in seen:
                continue
            seen.add(key)
            overlap_events.append(
                {
                    "turn_ids": list(key),
                    "start": overlap["start"],
                    "end": overlap["end"],
                    "duration_seconds": overlap["duration_seconds"],
                }
            )
    speaker_timeline = sorted(
        result.get("speaker_timeline") or [],
        key=lambda item: float(item.get("start") or 0.0),
    )
    active: list[dict[str, Any]] = []
    for interval in speaker_timeline:
        start = float(interval.get("start") or 0.0)
        end = max(start, float(interval.get("end") or start))
        active = [item for item in active if float(item.get("end") or 0.0) > start]
        for other in active:
            if other.get("speaker") == interval.get("speaker"):
                continue
            overlap_start = max(start, float(other.get("start") or 0.0))
            overlap_end = min(end, float(other.get("end") or 0.0))
            if overlap_end > overlap_start:
                overlap_events.append(
                    {
                        "speakers": [other.get("speaker"), interval.get("speaker")],
                        "start": overlap_start,
                        "end": overlap_end,
                        "duration_seconds": overlap_end - overlap_start,
                        "source": "diarization_timeline",
                    }
                )
        active.append(interval)
    identity_report = result.get("speaker_identity")
    speaker_limitation = (
        "VOICE labels are anonymous, automatically matched across this project's "
        "recordings, and are not verified real-world identities."
        if identity_report
        else "Speaker labels are anonymous and valid only within this source file."
    )
    limitations = [
        speaker_limitation,
        "ASR, speaker, word timing, acoustic, and tone outputs can be uncertain.",
        "Tone estimates are not facts about emotion, intent, honesty, mental state, "
        "or diagnosis.",
        "No language model correction or therapeutic interpretation is applied.",
    ]
    if settings.diarization_backend == "speechbrain":
        limitations.append(
            "Token-free ECAPA diarization globally clusters recognized-speech windows "
            "and is less reliable for overlap than frame-level pyannote diarization."
        )
    if identity_report:
        limitations.append(
            "Persistent voice matches can abstain or be wrong under short speech, "
            "overlap, noise, illness, aging, or recording-condition changes."
        )
    if result.get("speaker_refinement"):
        limitations.append(
            "Acoustic speaker refinement is confidence-gated but can still preserve "
            "or introduce a wrong short-turn label; raw assignments remain in JSON."
        )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "program_version": __version__,
        "source": {
            "path": str(source),
            "relative_path": relative.as_posix(),
            "size_bytes": fingerprint["size"],
            "mtime_ns": fingerprint["mtime_ns"],
            "fingerprint": fingerprint,
            "duration_seconds": round(duration, 3),
        },
        "language": {
            "requested": settings.language or "auto",
            "detected": detected,
            "task": settings.task,
            "output": output_language,
        },
        "processing": {
            "started_utc": started_utc,
            "completed_utc": utc_now(),
            "elapsed_seconds": round(elapsed, 3),
            "settings": asdict(settings),
            "settings_hash": settings_hash,
            "runtime": asdict(runtime),
            "provenance": provenance,
            "degraded_stages": degraded,
            "local_processing": True,
            "source_was_modified": False,
        },
        "segments": segments,
        "turns": turns,
        "speaker_timeline": speaker_timeline,
        "speaker_assignment_timeline": result.get("speaker_assignment_timeline"),
        "speaker_assignment_timeline_original": result.get(
            "speaker_assignment_timeline_original"
        ),
        "speaker_refinement": result.get("speaker_refinement"),
        "speaker_identity": result.get("speaker_identity"),
        "overlap_events": overlap_events,
        "limitations": limitations,
    }


def process_one(
    source: Path,
    relative: Path,
    outputs: dict[str, Path],
    fingerprint: dict[str, Any],
    settings: ProcessingSettings,
    settings_hash: str,
    runtime: RuntimeSettings,
    backend: Any,
    diarizer: Any,
    identity_encoder: Any,
    voice_registry: Any,
    acoustic_analyzer: Any,
    tone_estimator: Any,
    initial_degraded: list[str],
    args: argparse.Namespace,
) -> tuple[ProcessingResult, dict[str, Any]]:
    started = time.monotonic()
    started_utc = utc_now()
    degraded = list(initial_degraded)
    LOG.info("[%s] stage 1/7: decode and transcribe", relative)
    result, audio, provenance, backend_degraded = backend.transcribe(
        source, settings.align, args.verbose
    )
    degraded.extend(backend_degraded)

    if diarizer is not None:
        LOG.info("[%s] stage 2/7: anonymous speaker labeling", relative)
        try:
            result = diarizer.assign(
                source, audio, result, settings.min_speakers, settings.max_speakers
            )
            provenance["diarization_backend"] = diarizer.name
            provenance["diarization_model"] = getattr(diarizer, "model_name", None)
            validation = _validate_speaker_result(
                result, settings.min_speakers, settings.max_speakers
            )
            provenance["diarization_validation"] = validation
            provenance["diarization_runtime"] = getattr(
                diarizer, "last_run", {"device": runtime.analysis_device}
            )
        except Exception as exc:
            raise RuntimeError(
                "required speaker diarization failed; no transcript was marked "
                f"complete: {_safe_error(exc)}"
            ) from exc
    elif settings.diarization_backend != "off":
        raise RuntimeError(
            "required speaker diarization backend could not be initialized; "
            "no transcript was marked complete"
        )
    else:
        LOG.info("[%s] stage 2/7: speaker labeling disabled", relative)

    if settings.speaker_refinement:
        LOG.info(
            "[%s] stage 3/7: confidence-gated acoustic speaker refinement",
            relative,
        )
        if identity_encoder is None:
            degraded.append("speaker refinement: encoder is unavailable")
        else:
            try:
                refinement_report = identity_encoder.refine(audio, result)
                provenance["speaker_refinement"] = {
                    key: value
                    for key, value in refinement_report.items()
                    if key not in {"corrections", "evaluated_candidates"}
                }
                corrected = int(refinement_report.get("corrections_applied") or 0)
                if corrected:
                    LOG.info(
                        "[%s] acoustically corrected %d speaker-label run(s)",
                        relative,
                        corrected,
                    )
            except Exception as exc:
                degraded.append(f"speaker refinement: {_safe_error(exc)}")
    elif diarizer is not None:
        LOG.info("[%s] stage 3/7: acoustic speaker refinement disabled", relative)

    if settings.speaker_identity:
        LOG.info(
            "[%s] stage 4/7: whole-recording voice reconciliation and "
            "persistent matching",
            relative,
        )
        if identity_encoder is None or voice_registry is None:
            degraded.append("speaker identity: encoder or registry is unavailable")
        else:
            try:
                timeline = result.get("speaker_timeline") or []
                local_speakers = sorted(
                    {
                        str(item.get("speaker"))
                        for item in timeline
                        if item.get("speaker") not in (None, "", "SPEAKER_UNKNOWN")
                    }
                )
                evidence = identity_encoder.extract(audio, timeline, result)
                if source_fingerprint(source) != fingerprint:
                    raise RuntimeError(
                        "source file changed during speaker identity extraction"
                    )
                identity_report = voice_registry.identify(
                    source_key=relative.as_posix(),
                    source_fingerprint=stable_hash(
                        {
                            "size": fingerprint["size"],
                            "sample_sha256": fingerprint["sample_sha256"],
                        }
                    ),
                    local_speakers=local_speakers,
                    evidence=evidence,
                    incompatible_pairs=overlapping_speaker_pairs(timeline),
                    minimum_groups=settings.min_speakers,
                )
                result = apply_speaker_identities(result, identity_report)
                if identity_report["merged_local_clusters"]:
                    LOG.info(
                        "[%s] reconciled %d local speaker clusters into %d "
                        "voice candidate(s)",
                        relative,
                        identity_report["local_clusters_detected"],
                        identity_report["speaker_groups_after_reconciliation"],
                    )
                active_speaker_ids = identity_report["active_speaker_ids"]
                if (
                    settings.max_speakers is not None
                    and identity_report["active_speaker_count"] > settings.max_speakers
                ):
                    raise RuntimeError(
                        "persistent identity mapping exceeded the requested maximum "
                        "speaker count"
                    )
                LOG.info(
                    "[%s] active speakers in this recording: %d (%s); "
                    "project registry profiles: %d",
                    relative,
                    identity_report["active_speaker_count"],
                    ", ".join(active_speaker_ids),
                    identity_report["profile_count"],
                )
                matches = identity_report.get("matches") or []
                statuses = [item.get("status") for item in matches]
                enrolled_voices = {
                    item.get("speaker")
                    for item in matches
                    if item.get("status") == "enrolled"
                }
                matched_voices = {
                    item.get("speaker")
                    for item in matches
                    if item.get("status") == "matched"
                }
                provenance["speaker_identity"] = {
                    "embedding_model": identity_encoder.name,
                    "embedding_model_revision": identity_encoder.version,
                    "registry_revision": identity_report["registry_revision"],
                    "registry_state_hash": identity_report["registry_state_hash"],
                    "profile_count": identity_report["profile_count"],
                    "scope": identity_report["scope"],
                    "active_speaker_count": identity_report["active_speaker_count"],
                    "active_speaker_ids": active_speaker_ids,
                    "matched_speakers": len(matched_voices),
                    "enrolled_speakers": len(enrolled_voices),
                    "unresolved_speakers": sum(
                        str(status).startswith("unresolved") for status in statuses
                    ),
                    "local_clusters_detected": identity_report[
                        "local_clusters_detected"
                    ],
                    "speaker_groups_after_reconciliation": identity_report[
                        "speaker_groups_after_reconciliation"
                    ],
                    "merged_local_clusters": identity_report["merged_local_clusters"],
                }
            except Exception as exc:
                degraded.append(f"speaker identity: {_safe_error(exc)}")
    elif diarizer is not None:
        LOG.info("[%s] stage 4/7: persistent voice matching disabled", relative)

    LOG.info("[%s] stage 5/7: structure words, turns, and overlap", relative)
    segments = normalize_segments(result)
    turns = build_turns(segments)

    if acoustic_analyzer is not None:
        LOG.info("[%s] stage 6/7: measured acoustic observations", relative)
        try:
            acoustic_analyzer.analyze(audio, turns)
            provenance["acoustic_analyzer"] = {
                "name": acoustic_analyzer.name,
                "version": acoustic_analyzer.version,
            }
        except Exception as exc:
            degraded.append(f"acoustic observations: {_safe_error(exc)}")

    if tone_estimator is not None:
        LOG.info("[%s] stage 6/7: approximate turn-level tone", relative)
        try:
            tone_estimator.estimate(audio, turns)
            scored_turns = sum(
                bool((turn.get("tone") or {}).get("scores")) for turn in turns
            )
            unavailable_turns = sum(
                (turn.get("tone") or {}).get("kind") == "unavailable" for turn in turns
            )
            unclassified_turns = sum(
                bool((turn.get("tone") or {}).get("scores"))
                and (turn.get("tone") or {})["scores"][0].get("label") == "unknown"
                for turn in turns
            )
            temporally_mixed_turns = sum(
                bool((turn.get("tone") or {}).get("temporal_variation"))
                for turn in turns
            )
            scored_windows = sum(
                len((turn.get("tone") or {}).get("windows") or []) for turn in turns
            )
            if turns and scored_turns == 0:
                raise RuntimeError("tone estimator produced no usable turn scores")
            if unavailable_turns:
                degraded.append(
                    f"tone: unavailable for {unavailable_turns} of {len(turns)} turns"
                )
            provenance["tone_estimator"] = {
                "name": tone_estimator.name,
                "version": tone_estimator.version,
                "device": getattr(tone_estimator, "device", "cpu"),
            }
            provenance["tone_validation"] = {
                "scored_turns": scored_turns,
                "unavailable_turns": unavailable_turns,
                "unclassified_turns": unclassified_turns,
                "temporally_mixed_turns": temporally_mixed_turns,
                "scored_windows": scored_windows,
                "total_turns": len(turns),
            }
        except Exception as exc:
            degraded.append(f"tone: {_safe_error(exc)}")

    if source_fingerprint(source) != fingerprint:
        raise RuntimeError(
            "source file changed while it was being processed; retrying is safe"
        )
    degraded = list(dict.fromkeys(degraded))
    elapsed = time.monotonic() - started
    payload = _processing_payload(
        source,
        relative,
        fingerprint,
        settings,
        settings_hash,
        runtime,
        result,
        turns,
        provenance,
        degraded,
        started_utc,
        elapsed,
    )
    # The canonical result must contain the exact normalized segments used to
    # derive turns, including diarization labels and word confidence values.
    payload["segments"] = segments
    LOG.info("[%s] stage 7/7: atomically write outputs", relative)
    write_outputs(outputs, payload)
    release_accelerator_memory(runtime.device)
    return (
        ProcessingResult(
            source=source,
            relative_source=relative,
            status="complete",
            outputs=list(outputs.values()),
            language=payload["language"]["output"],
            duration_seconds=payload["source"]["duration_seconds"],
            elapsed_seconds=elapsed,
            degraded_stages=degraded,
        ),
        payload,
    )


def run_batch(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    if not paths.source_dir.exists():
        if args.source_dir:
            print(
                f"Source directory does not exist: {paths.source_dir}", file=sys.stderr
            )
            return 2
        paths.source_dir.mkdir(parents=True)
        print(f"Created {paths.source_dir}; add media files and run again.")
        return 0
    if not paths.source_dir.is_dir():
        print(f"Source path is not a directory: {paths.source_dir}", file=sys.stderr)
        return 2

    paths.transcript_dir.mkdir(parents=True, exist_ok=True)
    paths.review_dir.mkdir(parents=True, exist_ok=True)
    if args.reset_speaker_registry:
        try:
            backup_dir = archive_speaker_state(paths.review_dir)
        except OSError as exc:
            print(
                "Could not archive the existing speaker registry safely: "
                f"{_safe_error(exc)}",
                file=sys.stderr,
            )
            return 1
        if backup_dir is None:
            print("Speaker registry reset requested; no existing state was present.")
        else:
            print(f"Archived prior speaker identity state to: {backup_dir}")
            print(
                "Persistent voice IDs will be rebuilt from all currently discovered "
                "source media."
            )
    try:
        extensions = parse_extensions(args.extensions)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    files = discover_media(
        paths.source_dir,
        (paths.transcript_dir, paths.review_dir),
        args.recursive,
        extensions,
    )
    print(f"Source: {paths.source_dir}")
    print(f"Primary transcripts: {paths.transcript_dir}")
    print(f"Review artifacts: {paths.review_dir}")
    print(f"Discovered {len(files)} media file(s).")
    if not files:
        return 0

    token = resolve_hf_token(args.hf_token)
    runtime = resolve_runtime(args)
    diarization_name, tone_name, initial_degraded = resolve_analysis_backends(
        args, token
    )
    if args.task == "translate" and args.align:
        initial_degraded.append("alignment: unavailable for translation task")
    expected_degraded = list(initial_degraded)
    settings = processing_settings(args, diarization_name, tone_name)
    identity = settings_identity(settings, runtime)
    settings_hash = stable_hash(identity)
    manifest = ManifestStore(paths.review_dir / "transcription_manifest.json")
    voice_registry = None
    if settings.speaker_identity:
        registry_path = paths.review_dir / "speaker_registry.json"
        prior_registry = manifest.data.get("speaker_registry") or {}
        if not isinstance(prior_registry, dict):
            print(
                "Speaker registry metadata in the manifest is invalid. Restore the "
                "corresponding Review state before processing.",
                file=sys.stderr,
            )
            return 1
        try:
            prior_profile_count = int(prior_registry.get("profile_count") or 0)
            prior_revision = int(prior_registry.get("revision") or 0)
        except (TypeError, ValueError):
            print(
                "Speaker registry metadata in the manifest is invalid. Restore the "
                "corresponding Review state before processing.",
                file=sys.stderr,
            )
            return 1
        for entry in manifest.data.get("sources", {}).values():
            if not isinstance(entry, dict):
                continue
            provenance = entry.get("provenance") or {}
            if not isinstance(provenance, dict):
                continue
            prior_identity = provenance.get("speaker_identity") or {}
            if not isinstance(prior_identity, dict):
                continue
            try:
                prior_profile_count = max(
                    prior_profile_count,
                    int(prior_identity.get("profile_count") or 0),
                )
            except (TypeError, ValueError):
                print(
                    "Speaker identity metadata in the manifest is invalid. Restore "
                    "the corresponding Review state before processing.",
                    file=sys.stderr,
                )
                return 1
        if prior_profile_count and not registry_path.exists():
            print(
                "Persistent speaker registry is missing: "
                f"{registry_path}. Restore it before processing so existing "
                "VOICE identifiers are not silently reassigned.",
                file=sys.stderr,
            )
            return 1
        voice_registry = VoiceRegistry(
            registry_path,
            match_threshold=settings.speaker_match_threshold,
            match_margin=settings.speaker_match_margin,
            local_merge_threshold=settings.speaker_merge_threshold,
            enrollment_seconds=settings.speaker_enrollment_seconds,
        )
        try:
            registry_summary = voice_registry.validate()
        except Exception as exc:
            print(
                "Persistent speaker registry failed validation: "
                f"{_safe_error(exc)}. Restore or repair the registry before retrying.",
                file=sys.stderr,
            )
            return 1
        current_revision = int(registry_summary.get("revision") or 0)
        current_profile_count = int(registry_summary.get("profile_count") or 0)
        registry_has_manifest_history = bool(prior_registry) or prior_profile_count > 0
        if (
            current_revision > 0
            and current_profile_count > 0
            and not registry_has_manifest_history
        ):
            print(
                "Persistent speaker registry exists without its corresponding "
                "manifest history. Git pull and ./install.sh preserve ignored "
                "Review state, so this is not a fresh identity registry. Restore "
                "the matching transcription_manifest.json or rerun with "
                "--reset-speaker-registry to archive and rebuild the state.",
                file=sys.stderr,
            )
            return 1
        if prior_revision > current_revision or (
            prior_revision == current_revision
            and prior_revision > 0
            and prior_registry.get("state_hash")
            and prior_registry["state_hash"] != registry_summary.get("state_hash")
        ):
            print(
                "Persistent speaker registry does not match the latest manifest "
                "state. Restore the corresponding Review/speaker_registry.json "
                "before processing so VOICE identifiers remain stable.",
                file=sys.stderr,
            )
            return 1
        if current_profile_count:
            print(
                "Speaker identity state: preserving "
                f"{current_profile_count} project-wide profile(s) at registry "
                f"revision {current_revision}. VOICE numbers are durable IDs, not "
                "the active speaker count."
            )
        else:
            print(
                "Speaker identity state: empty registry; active voices will be "
                "allocated after each complete recording is reconciled."
            )
    pending: list[tuple[Path, Path, dict[str, Any], dict[str, Path]]] = []
    skipped = 0
    for source in files:
        relative = source.relative_to(paths.source_dir)
        fingerprint = source_fingerprint(source)
        outputs = expected_outputs(
            source,
            paths.source_dir,
            paths.transcript_dir,
            paths.review_dir,
            settings.review_formats,
        )
        complete, reason = state_is_complete(
            manifest.get(relative.as_posix()), fingerprint, settings_hash, outputs
        )
        if complete and not args.overwrite:
            print(f"SKIP    {relative} ({reason})")
            skipped += 1
        else:
            processing_reason = "overwrite requested" if args.overwrite else reason
            print(f"PROCESS {relative} ({processing_reason})")
            pending.append((source, relative, fingerprint, outputs))

    if args.dry_run:
        print(
            f"Dry run complete: {len(pending)} would process; {skipped} would skip. "
            "No models were loaded."
        )
        return 0
    if not pending:
        manifest.record_run(
            {
                "started_utc": utc_now(),
                "completed_utc": utc_now(),
                "status": "complete",
                "processed": 0,
                "skipped": skipped,
                "failed": 0,
            }
        )
        print(f"Nothing to do: {skipped} complete unchanged file(s) skipped.")
        return 0

    # Installation/model preparation is the only mode allowed to contact model
    # registries. A normal media run consumes the already prepared local cache.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print(f"Runtime: {runtime.description}")
    run_started = utc_now()
    try:
        backend, actual_runtime = WhisperXBackend.create(
            settings.model, settings.language, settings.task, runtime, token
        )
    except Exception as exc:
        print(f"Could not load ASR model: {_safe_error(exc)}", file=sys.stderr)
        print(
            "Run ./transcribe-media --prepare-models while online, then retry.",
            file=sys.stderr,
        )
        return 1
    if actual_runtime != runtime:
        runtime = actual_runtime
        identity = settings_identity(settings, runtime)
        settings_hash = stable_hash(identity)
        print(f"Runtime fallback: {runtime.description}")

    try:
        diarizer = create_diarizer(
            diarization_name,
            token,
            runtime.analysis_device,
            runtime.diarization_batch_size,
        )
    except Exception as exc:
        initial_degraded.append(f"diarization initialization: {_safe_error(exc)}")
        if diarization_name == "pyannote" and _available("speechbrain"):
            try:
                diarizer = create_diarizer("speechbrain", None, runtime.analysis_device)
                initial_degraded.append(
                    "diarization: pyannote unavailable; using token-free "
                    "windowed voice clustering"
                )
            except Exception as fallback_exc:
                initial_degraded.append(
                    f"diarization fallback initialization: {_safe_error(fallback_exc)}"
                )
                diarizer = None
        else:
            diarizer = None
    identity_encoder = None
    if (
        settings.speaker_identity or settings.speaker_refinement
    ) and diarizer is not None:
        try:
            identity_encoder = SpeakerIdentityEncoder(
                runtime.analysis_device,
                classifier=getattr(diarizer, "classifier", None),
            )
        except Exception as exc:
            initial_degraded.append(
                f"speaker refinement/identity initialization: {_safe_error(exc)}"
            )
    acoustic_analyzer = AcousticAnalyzer() if settings.acoustic_analysis else None
    try:
        tone_estimator = create_tone_estimator(tone_name, runtime.analysis_device)
    except Exception as exc:
        initial_degraded.append(f"tone initialization: {_safe_error(exc)}")
        tone_estimator = None
        if tone_name == "emotion2vec" and _available("speechbrain"):
            try:
                tone_estimator = create_tone_estimator(
                    "speechbrain", runtime.analysis_device
                )
                initial_degraded.append(
                    "tone: emotion2vec unavailable; using the smaller four-label "
                    "SpeechBrain IEMOCAP model"
                )
            except Exception as fallback_exc:
                initial_degraded.append(
                    f"tone fallback initialization: {_safe_error(fallback_exc)}"
                )
        if tone_estimator is None and settings.acoustic_analysis:
            tone_estimator = HeuristicToneEstimator()
            initial_degraded.append(
                "tone: learned estimator unavailable; using low-confidence "
                "acoustic heuristic"
            )

    processed = failed = degraded_count = 0
    interrupted = False
    for source, relative, fingerprint, outputs in pending:
        key = relative.as_posix()
        entry_base = {
            "source_path": str(source),
            "relative_path": key,
            "source_fingerprint": fingerprint,
            "settings": settings.fingerprint_payload(),
            "settings_hash": settings_hash,
            "model_versions": identity["dependency_versions"],
            "runtime": asdict(runtime),
            "outputs": {name: str(path) for name, path in outputs.items()},
        }
        manifest.update(
            key,
            {
                **entry_base,
                "status": "processing",
                "completion_state": False,
                "started_utc": utc_now(),
            },
        )
        try:
            result, payload = process_one(
                source,
                relative,
                outputs,
                fingerprint,
                settings,
                settings_hash,
                runtime,
                backend,
                diarizer,
                identity_encoder,
                voice_registry,
                acoustic_analyzer,
                tone_estimator,
                initial_degraded,
                args,
            )
            needs_retry = any(
                item not in expected_degraded for item in (result.degraded_stages or [])
            )
            manifest.update(
                key,
                {
                    **entry_base,
                    "status": "degraded" if needs_retry else "complete",
                    "completion_state": not needs_retry,
                    "completed_utc": utc_now(),
                    "elapsed_seconds": round(result.elapsed_seconds or 0.0, 3),
                    "duration_seconds": result.duration_seconds,
                    "language": result.language,
                    "degraded_stages": result.degraded_stages,
                    "retry_recommended": needs_retry,
                    "provenance": payload["processing"]["provenance"],
                },
            )
            identity_report = payload.get("speaker_identity") or {}
            if identity_report:
                manifest.record_speaker_registry(
                    {
                        "path": str(voice_registry.path),
                        "schema_version": identity_report.get(
                            "registry_schema_version"
                        ),
                        "revision": identity_report.get("registry_revision"),
                        "profile_count": identity_report.get("profile_count"),
                        "state_hash": identity_report.get("registry_state_hash"),
                        "embedding_model": identity_report.get("embedding_model"),
                    }
                )
            processed += 1
            if needs_retry:
                degraded_count += 1
                print(
                    f"DEGRADED {relative}: one or more requested stages failed; "
                    "the file will be retried on the next run",
                    file=sys.stderr,
                )
            else:
                print(f"DONE    {relative} -> {outputs['txt']}")
        except KeyboardInterrupt:
            interrupted = True
            manifest.update(
                key,
                {
                    **entry_base,
                    "status": "interrupted",
                    "completion_state": False,
                    "updated_utc": utc_now(),
                },
            )
            print(
                f"INTERRUPTED {relative}; completed files remain resumable.",
                file=sys.stderr,
            )
            break
        except Exception as exc:
            failed += 1
            error = _safe_error(exc)
            manifest.update(
                key,
                {
                    **entry_base,
                    "status": "failed",
                    "completion_state": False,
                    "failed_utc": utc_now(),
                    "error": error,
                    "traceback": traceback.format_exc() if args.verbose else None,
                },
            )
            print(f"FAILED  {relative}: {error}", file=sys.stderr)
            release_accelerator_memory(runtime.device)
        finally:
            release_audio = getattr(backend, "release_audio", None)
            if release_audio is not None:
                release_audio()

    status = (
        "interrupted"
        if interrupted
        else ("partial" if failed or degraded_count else "complete")
    )
    manifest.record_run(
        {
            "started_utc": run_started,
            "completed_utc": utc_now(),
            "status": status,
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
            "degraded": degraded_count,
            "interrupted": interrupted,
            "settings_hash": settings_hash,
            "runtime": asdict(runtime),
        }
    )
    print(
        f"Run {status}: {processed} processed, {skipped} skipped, {failed} failed, "
        f"{degraded_count} degraded. "
        f"Manifest: {manifest.path}"
    )
    return 130 if interrupted else (1 if failed or degraded_count else 0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.verbose:
        for logger_name in (
            "filelock",
            "huggingface_hub",
            "lightning",
            "lightning.pytorch.utilities.migration.utils",
            "matplotlib",
            "opentelemetry",
            "pytorch_lightning",
            "speechbrain",
            "urllib3",
        ):
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    if args.configure:
        return configure(args)
    if args.doctor:
        return doctor()
    if args.prepare_models:
        try:
            return prepare_models(args)
        except KeyboardInterrupt:
            print("Model preparation interrupted.", file=sys.stderr)
            return 130
        except Exception as exc:
            print(f"Model preparation failed: {_safe_error(exc)}", file=sys.stderr)
            if args.verbose:
                traceback.print_exc()
            return 1
    return run_batch(args)
