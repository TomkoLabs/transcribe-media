"""Reuse expensive ASR/alignment when only voice policy changes."""
from __future__ import annotations

import json

from .storage import atomic_write_json, stable_hash


def transcribe_cached(backend, source, fingerprint, settings, args):
    if not settings.quality:
        return backend.transcribe(source, settings.align, args.verbose)
    from .cli import resolve_paths
    from .backends import package_version
    key = stable_hash({"version": 2, "audio": fingerprint.get("sha256", fingerprint["sample_sha256"]),
                       "model": settings.model, "language": settings.language, "task": settings.task,
                       "align": settings.align, "quality": settings.quality,
                       "versions": {name: package_version(name) for name in
                                    ("whisperx", "faster-whisper", "ctranslate2", "torch")}})
    path = resolve_paths(args).review_dir / "stage-cache" / (key + ".asr.json")
    if not args.overwrite and path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached["key"] == key and isinstance(cached["result"].get("segments"), list):
                audio = backend._decode_audio(source)
                cached["provenance"]["asr_alignment_cache_hit"] = True
                from .timing import annotate_word_timing
                annotate_word_timing(cached['result'])
                return cached["result"], audio, cached["provenance"], []
        except (KeyError, ValueError, TypeError):
            pass
    result, audio, provenance, degraded = backend.transcribe(source, settings.align, args.verbose)
    if not degraded:
        atomic_write_json(path, {"key": key, "result": result, "provenance": provenance})
    return result, audio, provenance, degraded
