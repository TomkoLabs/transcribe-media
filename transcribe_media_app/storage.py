from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from . import __version__
from .schema import (
    COMMON_MEDIA_EXTENSIONS,
    MANIFEST_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
)

FINGERPRINT_CHUNK_BYTES = 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_fingerprint(path: Path) -> dict[str, Any]:
    """Return a durable, bounded-cost fingerprint without loading the file."""
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("ascii"))
    with path.open("rb") as handle:
        first = handle.read(FINGERPRINT_CHUNK_BYTES)
        digest.update(first)
        if stat.st_size > FINGERPRINT_CHUNK_BYTES:
            handle.seek(max(0, stat.st_size - FINGERPRINT_CHUNK_BYTES))
            digest.update(handle.read(FINGERPRINT_CHUNK_BYTES))
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sample_sha256": digest.hexdigest(),
    }


def parse_extensions(raw: Optional[str]) -> Optional[set[str]]:
    if raw is None:
        return None
    extensions: set[str] = set()
    for value in raw.split(","):
        value = value.strip().lower()
        if value:
            extensions.add(value if value.startswith(".") else f".{value}")
    if not extensions:
        raise ValueError("--extensions requires at least one extension")
    return extensions


def ffprobe_has_audio(path: Path) -> bool:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and bool(completed.stdout.strip())


def ffprobe_duration(path: Path) -> Optional[float]:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        value = float(completed.stdout.strip())
        return value if completed.returncode == 0 and value >= 0 else None
    except (FileNotFoundError, subprocess.SubprocessError, TypeError, ValueError):
        return None


def _inside(candidate: Path, directory: Path) -> bool:
    try:
        candidate.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def discover_media(
    source_dir: Path,
    excluded_dirs: Iterable[Path],
    recursive: bool,
    extensions: Optional[set[str]],
) -> list[Path]:
    iterator = source_dir.rglob("*") if recursive else source_dir.iterdir()
    excluded = [path for path in excluded_dirs if path.exists()]
    files: list[Path] = []

    for candidate in iterator:
        if not candidate.is_file() or candidate.name.startswith("."):
            continue
        if any(_inside(candidate, directory) for directory in excluded):
            continue

        suffix = candidate.suffix.lower()
        if extensions is not None:
            if suffix in extensions:
                files.append(candidate)
            continue

        if suffix in COMMON_MEDIA_EXTENSIONS or ffprobe_has_audio(candidate):
            files.append(candidate)

    return sorted(files, key=lambda path: str(path.relative_to(source_dir)).casefold())


def output_base(source: Path, source_dir: Path, destination: Path) -> Path:
    relative = source.relative_to(source_dir)
    return destination / relative.parent / relative.name


def output_path(base: Path, extension: str) -> Path:
    return base.parent / f"{base.name}.{extension.lstrip('.')}"


def expected_outputs(
    source: Path,
    source_dir: Path,
    transcript_dir: Path,
    review_dir: Path,
    review_formats: Iterable[str],
) -> dict[str, Path]:
    transcript_base = output_base(source, source_dir, transcript_dir)
    review_base = output_base(source, source_dir, review_dir)
    outputs = {"txt": output_path(transcript_base, "txt")}
    outputs.update(
        {
            format_name: output_path(review_base, format_name)
            for format_name in review_formats
        }
    )
    return outputs


def output_is_valid(path: Path, format_name: str) -> bool:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        if format_name == "json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            return (
                isinstance(payload, dict)
                and payload.get("schema_version") == RESULT_SCHEMA_VERSION
                and isinstance(payload.get("turns"), list)
                and isinstance(payload.get("processing"), dict)
            )
        return True
    except (OSError, UnicodeError, ValueError):
        return False


class ManifestStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = self._load()

    def _empty(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "program_version": __version__,
            "updated_utc": utc_now(),
            "sources": {},
            "last_run": None,
        }

    def _load(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get(
                "schema_version"
            ) != MANIFEST_SCHEMA_VERSION or not isinstance(
                payload.get("sources"), dict
            ):
                raise ValueError("unsupported manifest schema")
            return payload
        except FileNotFoundError:
            return self._empty()
        except (AttributeError, OSError, TypeError, ValueError):
            # A fresh in-memory state is safer than trusting completeness that
            # cannot be proven. The next atomic save replaces the invalid file.
            return self._empty()

    def get(self, source_key: str) -> Optional[dict[str, Any]]:
        entry = self.data["sources"].get(source_key)
        return entry if isinstance(entry, dict) else None

    def update(self, source_key: str, entry: dict[str, Any]) -> None:
        self.data["sources"][source_key] = entry
        self.save()

    def save(self) -> None:
        self.data["program_version"] = __version__
        self.data["updated_utc"] = utc_now()
        atomic_write_json(self.path, self.data)

    def record_run(self, payload: dict[str, Any]) -> None:
        self.data["last_run"] = payload
        self.save()

    def record_speaker_registry(self, payload: dict[str, Any]) -> None:
        self.data["speaker_registry"] = payload
        self.save()


def state_is_complete(
    state: Optional[dict[str, Any]],
    fingerprint: dict[str, Any],
    settings_hash: str,
    outputs: dict[str, Path],
) -> tuple[bool, str]:
    if not state:
        return False, "not previously processed"
    if state.get("status") != "complete" or not state.get("completion_state"):
        return False, f"previous status is {state.get('status', 'unknown')}"
    if state.get("retry_recommended"):
        return False, "a processing stage requested retry"
    if state.get("source_fingerprint") != fingerprint:
        return False, "source changed"
    if state.get("settings_hash") != settings_hash:
        return False, "processing settings changed"
    for format_name, path in outputs.items():
        if not output_is_valid(path, format_name):
            return False, f"required {format_name.upper()} output is missing or invalid"
    return True, "complete and unchanged"
