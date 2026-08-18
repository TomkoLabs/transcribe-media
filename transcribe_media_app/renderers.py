from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .storage import atomic_write_json, atomic_write_text


def format_timestamp(seconds: Any, decimal: str = ".") -> str:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        value = 0.0
    if not math.isfinite(value) or value < 0:
        value = 0.0
    total_milliseconds = int(round(value * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}{decimal}{milliseconds:03d}"


def _tone_line(tone: Any) -> str:
    if not isinstance(tone, dict):
        return "not estimated"
    scores = tone.get("scores") or []

    def display_label(item: dict[str, Any]) -> str:
        # emotion2vec's ninth class is an abstention, not a pipeline failure or
        # a claim that the speaker's presentation is literally "unknown".
        if item.get("label") == "unknown":
            return "unclassified"
        return str(item.get("label", "unclassified"))

    rendered = ", ".join(
        f"{display_label(item)} {float(item.get('probability', 0.0)):.0%}"
        for item in scores[:3]
    )
    if tone.get("temporal_variation"):
        rendered = f"varies across {len(tone.get('windows') or [])} windows; {rendered}"
    model = tone.get("model") or "unknown model"
    if tone.get("kind") == "unavailable":
        return f"unavailable ({model})"
    return f"{rendered or 'no estimate'}; model: {model}"


def render_txt(payload: dict[str, Any]) -> str:
    source = payload["source"]
    processing = payload["processing"]
    language = payload["language"]
    identity = payload.get("speaker_identity") or {}
    speaker_scope = (
        "persistent anonymous profiles "
        f"(registry revision {identity.get('registry_revision', 'unknown')})"
        if identity
        else "anonymous labels local to this recording"
    )
    reconciliation = None
    refinement = None
    active_speakers = None
    registry_profiles = None
    refinement_report = payload.get("speaker_refinement") or {}
    if refinement_report:
        corrected = int(refinement_report.get("corrections_applied") or 0)
        refinement = (
            f"Speaker refinement: {corrected} label correction(s); "
            "raw assignments retained in JSON"
        )
    if identity:
        local_count = identity.get("local_clusters_detected")
        group_count = identity.get("speaker_groups_after_reconciliation")
        if local_count is not None and group_count is not None:
            reconciliation = (
                f"Speaker clustering: {local_count} local cluster(s) -> "
                f"{group_count} reconciled voice candidate(s)"
            )
        active_ids = [str(item) for item in identity.get("active_speaker_ids") or []]
        if active_ids:
            active_count = identity.get("active_speaker_count", len(active_ids))
            active_speakers = (
                f"Active speakers in this recording: {active_count} "
                f"({', '.join(active_ids)})"
            )
        if identity.get("profile_count") is not None:
            registry_profiles = (
                "Project voice registry: "
                f"{identity['profile_count']} profile(s) total across recordings"
            )
    lines = [
        "TRANSCRIPT",
        "==========",
        f"Source: {source.get('relative_path') or source.get('path')}",
        f"Language: {language.get('output') or language.get('detected') or 'unknown'}",
        f"Task: {language.get('task', 'transcribe')}",
        f"Created: {processing.get('completed_utc')}",
        "ASR: "
        f"{processing.get('provenance', {}).get('transcription_model', 'unknown')}",
        f"Device: {processing.get('runtime', {}).get('device', 'unknown')}",
        f"Speaker IDs: {speaker_scope}",
        *((reconciliation,) if reconciliation else ()),
        *((refinement,) if refinement else ()),
        *((active_speakers,) if active_speakers else ()),
        *((registry_profiles,) if registry_profiles else ()),
        "",
        "Observed annotations are measured acoustic/timing features. Tone annotations",
        "are approximate model estimates, not facts about emotion, intent, honesty,",
        "mental state, diagnosis, or the meaning of the conversation.",
        "Speaker labels are probabilistic acoustic assignments. Verify attribution",
        "against the recording before concluding who said a statement.",
    ]
    degraded = processing.get("degraded_stages") or []
    if degraded:
        lines.extend(("", "Degraded stages:", *(f"- {item}" for item in degraded)))
    lines.extend(("", "TRANSCRIPT", "----------", ""))

    for turn in payload.get("turns") or []:
        start = format_timestamp(turn.get("start"))
        end = format_timestamp(turn.get("end"))
        speaker = turn.get("speaker") or "SPEAKER_UNKNOWN"
        observations = turn.get("observations") or []
        observed = "; ".join(str(item.get("label")) for item in observations)
        if not observed:
            observed = "no notable acoustic flags"
        lines.extend(
            (
                f"[{start} - {end}] {speaker}:",
                str(turn.get("text") or "").strip(),
                f"[Observed: {observed}]",
                f"[Tone approx: {_tone_line(turn.get('tone'))}]",
                "",
            )
        )
    if not payload.get("turns"):
        lines.append("[No speech was transcribed.]\n")
    return "\n".join(lines).rstrip() + "\n"


def render_srt(segments: list[dict[str, Any]]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        speaker = segment.get("speaker") or "SPEAKER_UNKNOWN"
        text = str(segment.get("text") or "").strip()
        blocks.append(
            "\n".join(
                (
                    str(index),
                    f"{format_timestamp(segment.get('start'), ',')} --> "
                    f"{format_timestamp(segment.get('end'), ',')}",
                    f"[{speaker}] {text}",
                )
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_vtt(segments: list[dict[str, Any]]) -> str:
    blocks = []
    for segment in segments:
        speaker = segment.get("speaker") or "SPEAKER_UNKNOWN"
        text = str(segment.get("text") or "").strip()
        blocks.append(
            "\n".join(
                (
                    f"{format_timestamp(segment.get('start'))} --> "
                    f"{format_timestamp(segment.get('end'))}",
                    f"<{speaker}>{text}",
                )
            )
        )
    body = "\n\n".join(blocks)
    return "WEBVTT\n\n" + body + ("\n" if body else "")


def write_outputs(
    outputs: dict[str, Path],
    payload: dict[str, Any],
) -> None:
    segments = payload.get("segments") or []
    for format_name, path in outputs.items():
        if format_name == "txt":
            atomic_write_text(path, render_txt(payload))
        elif format_name == "json":
            atomic_write_json(path, payload)
        elif format_name == "srt":
            atomic_write_text(path, render_srt(segments))
        elif format_name == "vtt":
            atomic_write_text(path, render_vtt(segments))
        else:
            raise ValueError(f"unsupported output format: {format_name}")
