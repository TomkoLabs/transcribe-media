"""Publish a current, tool-owned TXT export without network or model calls."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import tempfile
from pathlib import Path

from . import __version__
from .recovery import inspect_packet
from .renderers import render_txt
from .review import packet_path
from .storage import ManifestStore, atomic_write_json, stable_hash, utc_now
import hashlib

MARKER = ".transcribe-media-export.json"
OWNER = "transcribe-media-analysis-ready-v1"


def _exchange(left, right):
    """Linux atomic directory exchange; no remove-then-rename visibility gap."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise RuntimeError(
            "atomic directory exchange is unavailable; existing export was retained"
        )
    if rename(-100, os.fsencode(left), -100, os.fsencode(right), 2):
        raise OSError(ctypes.get_errno(), "atomic export publication failed")


def _owned(destination):
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("export destination must be a real tool-owned directory")
    marker = destination / MARKER
    try:
        state = json.loads(marker.read_text(encoding="utf-8"))
        if state["owner"] != OWNER:
            raise ValueError
        expected = {item["relative_path"]: item["sha256"] for item in state["files"]}
        actual = set()
        for path in destination.rglob("*"):
            if path.is_symlink():
                raise ValueError
            if path.is_file() and path != marker:
                key = path.relative_to(destination).as_posix()
                actual.add(key)
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected.get(key):
                    raise ValueError
        if actual != set(expected):
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            "destination is unowned or contains operator changes; choose a new empty export path, preserving this directory"
        ) from exc


def export_analysis_ready(review_dir, transcript_dir, destination):
    review_dir, transcript_dir, destination = map(
        Path, (review_dir, transcript_dir, destination)
    )
    if any(
        destination.resolve().is_relative_to(root.resolve())
        or root.resolve().is_relative_to(destination.resolve())
        for root in (review_dir, transcript_dir)
    ):
        raise ValueError(
            "export directory must be separate from Review and Transcribed"
        )
    if destination.exists() or destination.is_symlink():
        _owned(destination)
    manifest = ManifestStore(review_dir / "transcription_manifest.json")
    if not manifest.path.is_file() or not manifest.data["sources"]:
        raise ValueError(
            "no saved processing manifest; render current reviewed transcripts first"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix="." + destination.name + ".generation-", dir=destination.parent
        )
    )
    included, excluded = [], []
    published = False
    try:
        for key, entry in sorted(manifest.data["sources"].items()):
            reason = "incomplete_or_invalid"
            try:
                path = packet_path(review_dir, key)
                status = inspect_packet(path, manifest=manifest)
                if not status["decisions_allowed"] or status["pending"]:
                    reason = status["state"]
                    raise ValueError
                if (
                    entry.get("status") != "complete"
                    or not entry.get("completion_state")
                    or entry.get("retry_recommended")
                ):
                    raise ValueError
                source = Path(entry["outputs"]["txt"])
                if source.is_symlink() or not source.resolve().is_relative_to(
                    transcript_dir.resolve()
                ):
                    raise ValueError
                relative = source.resolve().relative_to(transcript_dir.resolve())
                raw = source.read_bytes()
                text = raw.decode("utf-8-sig")
                packet = json.loads(path.read_text(encoding="utf-8"))
                if not text.strip().startswith(
                    "ANALYSIS-READY TRANSCRIPT\n"
                ) or text != render_txt(packet["payload"]):
                    reason = "draft_or_outdated_render; run --render-transcripts"
                    raise ValueError
                output = stage / relative
                output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with output.open("xb") as handle:
                    os.fchmod(handle.fileno(), 0o600)
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                included.append(
                    {
                        "relative_path": relative.as_posix(),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "source": key,
                        "finality": "ANALYSIS-READY TRANSCRIPT",
                    }
                )
            except (OSError, ValueError, KeyError, TypeError):
                excluded.append({"source": key, "reason": reason})
        atomic_write_json(
            stage / MARKER,
            {
                "owner": OWNER,
                "version": __version__,
                "exported_at": utc_now(),
                "files": included,
                "content_hash": stable_hash(included),
            },
        )
        if destination.exists():
            _owned(destination)
            _exchange(stage, destination)
            # The former generation stays private and recoverable, even if an
            # operator wrote into it concurrently. Never recursively delete it.
        else:
            stage.rename(destination)
        published = True
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return {
            "destination": str(destination),
            "included": len(included),
            "excluded": len(excluded),
            "exclusions": excluded,
            "previous_generation": str(stage) if stage.exists() else None,
        }
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)
