from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from . import __version__

SAMPLE_RATE = 16_000
RESULT_SCHEMA_VERSION = "1.6"
MANIFEST_SCHEMA_VERSION = "1.0"
ACOUSTIC_ANALYZER_VERSION = "1.0"
HEURISTIC_TONE_VERSION = "1.0"

# Known media extensions are accepted even when a corrupt file cannot be
# identified by ffprobe. Files with other extensions are also accepted when
# ffprobe finds an audio stream.
COMMON_MEDIA_EXTENSIONS = {
    ".3gp",
    ".aac",
    ".aiff",
    ".alac",
    ".amr",
    ".ape",
    ".asf",
    ".avi",
    ".flac",
    ".m2ts",
    ".m4a",
    ".m4v",
    ".mka",
    ".mkv",
    ".mov",
    ".mp2",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".mts",
    ".oga",
    ".ogg",
    ".ogv",
    ".opus",
    ".ts",
    ".wav",
    ".webm",
    ".wma",
    ".wmv",
}


@dataclass(frozen=True)
class RuntimeSettings:
    device: str
    compute_type: str
    batch_size: int
    threads: int
    device_was_auto: bool
    description: str
    analysis_device: str = "cpu"
    diarization_batch_size: int = 4


@dataclass(frozen=True)
class ProcessingSettings:
    model: str
    language: Optional[str]
    task: str
    align: bool
    diarization_backend: str
    min_speakers: Optional[int]
    max_speakers: Optional[int]
    speaker_identity: bool
    speaker_refinement: bool
    speaker_match_threshold: float
    speaker_match_margin: float
    speaker_merge_threshold: float
    speaker_enrollment_seconds: float
    acoustic_analysis: bool
    tone_backend: str
    review_formats: tuple[str, ...]

    def fingerprint_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["program_version"] = __version__
        payload["result_schema_version"] = RESULT_SCHEMA_VERSION
        payload["acoustic_analyzer_version"] = ACOUSTIC_ANALYZER_VERSION
        payload["heuristic_tone_version"] = HEURISTIC_TONE_VERSION
        return payload


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    source_dir: Path
    transcript_dir: Path
    review_dir: Path


@dataclass
class ProcessingResult:
    source: Path
    relative_source: Path
    status: str
    outputs: list[Path]
    language: Optional[str] = None
    duration_seconds: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    error: Optional[str] = None
    degraded_stages: Optional[list[str]] = None
