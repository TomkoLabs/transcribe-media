"""Rebuild presentation files from canonical results, without inference or learning."""
from __future__ import annotations

import json
from pathlib import Path

from .renderers import write_outputs
from .schema import MANIFEST_SCHEMA_VERSION, RESULT_SCHEMA_VERSION
from .storage import StateTransaction


def render_saved_transcripts(review_dir: Path) -> dict[str, int]:
    """Caller holds the project lock and has recovered any interrupted transaction.

    JSON, review decisions, profile evidence and processing state are read-only.
    Validate every saved result first; any write failure restores all exports.
    Original recordings and model weights are unnecessary for this operation.
    """
    review_dir = Path(review_dir)
    manifest_path = review_dir / "transcription_manifest.json"
    if not manifest_path.exists():
        return {"recordings_rendered": 0, "files_written": 0}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
            or not isinstance(manifest.get("sources"), dict)):
        raise ValueError("invalid processing manifest; restore its backup before rendering transcripts")

    jobs = []
    protected = {manifest_path.resolve(), (review_dir / "speaker_registry.json").resolve()}
    for source, entry in manifest["sources"].items():
        if not isinstance(entry, dict) or not isinstance(entry.get("outputs", {}), dict):
            raise ValueError(f"invalid saved output list for {source}")
        outputs = entry.get("outputs", {})
        if not outputs:
            continue
        if any(not isinstance(path, str) or not path for path in outputs.values()):
            raise ValueError(f"invalid saved output path for {source}")
        canonical = Path(outputs["json"]) if outputs.get("json") else None
        if canonical is None or not canonical.is_file():
            raise ValueError(f"saved JSON is missing for {source}; restore it or process that recording again")
        try:
            payload = json.loads(canonical.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ValueError(f"invalid saved transcript JSON for {source}: {canonical}") from exc
        if (not isinstance(payload, dict) or payload.get("schema_version") != RESULT_SCHEMA_VERSION
                or any(not isinstance(payload.get(key), dict) for key in ("source", "language", "processing"))
                or not isinstance(payload.get("turns"), list)):
            raise ValueError(f"invalid or unsupported saved transcript JSON for {source}")
        files = {key: Path(path) for key, path in outputs.items() if key != "json"}
        if set(files) - {"txt", "detailed_txt", "srt", "vtt"}:
            raise ValueError(f"unsupported saved output format for {source}")
        protected.add(canonical.resolve())
        jobs.append((source, files, payload))

    destinations = [path for _, files, _ in jobs for path in files.values()]
    resolved = [path.resolve() for path in destinations]
    if any(path in protected for path in resolved) or len(set(resolved)) != len(resolved):
        raise ValueError("saved output paths overlap; restore distinct export paths before rendering")
    if not destinations:
        return {"recordings_rendered": 0, "files_written": 0}
    transaction = StateTransaction(review_dir, destinations)
    transaction.begin()
    try:
        for source, files, payload in jobs:
            try:
                write_outputs(files, payload)
            except (KeyError, TypeError, AttributeError) as exc:
                raise ValueError(f"invalid saved transcript fields for {source}; restore its JSON backup") from exc
        transaction.commit()
    except BaseException:
        transaction.rollback()
        raise
    return {"recordings_rendered": sum(bool(files) for _, files, _ in jobs), "files_written": len(destinations)}
