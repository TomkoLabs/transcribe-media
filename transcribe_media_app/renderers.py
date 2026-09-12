from __future__ import annotations

import math
from html import escape
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
    if tone.get("kind") == "unavailable":
        return "unavailable"
    return rendered or "no estimate"


def _tone_models(payload: dict[str, Any]) -> list[str]:
    return list(
        dict.fromkeys(
            str(tone["model"])
            for turn in payload.get("turns") or []
            if isinstance((tone := turn.get("tone")), dict) and tone.get("model")
        )
    )


def _speaker_is_uncertain(turn: dict[str, Any]) -> bool:
    return (turn.get("speaker_attribution") or {}).get("status") == "uncertain"


def speaker_display(speaker, profiles=()):
    speaker = speaker or "SPEAKER_UNKNOWN"
    if speaker == "SPEAKER_UNKNOWN":
        return "UNKNOWN"
    profile = next((item for item in profiles if item["voice_id"] == speaker), {})
    label = profile.get("label")
    return f"{label} [{speaker}]" if label and label != speaker else speaker


def render_detailed_txt(payload: dict[str, Any]) -> str:
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
    ensemble_summary = None
    active_speakers = None
    registry_profiles = None
    tone_models = _tone_models(payload)
    refinement_report = payload.get("speaker_refinement") or {}
    ensemble = payload.get("diarization_ensemble") or {}
    if ensemble:
        if ensemble.get("secondary_available"):
            annotated = int(ensemble.get("annotated_words") or 0)
            agreement = float(ensemble.get("agreement_fraction") or 0.0)
            ensemble_summary = (
                "Diarization ensemble: pyannote + Sortformer v2.1; "
                f"agreement on {agreement:.0%} of {annotated} compared word(s)"
            )
        else:
            ensemble_summary = "Diarization ensemble: Sortformer unavailable"
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
        *((f"Tone model: {', '.join(tone_models)}",) if tone_models else ()),
        f"Speaker IDs: {speaker_scope}",
        *((ensemble_summary,) if ensemble_summary else ()),
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
        speaker = speaker_display(turn.get("speaker"), payload.get("speaker_profiles", []))
        observations = turn.get("observations") or []
        observed = "; ".join(str(item.get("label")) for item in observations)
        if not observed:
            observed = "no notable acoustic flags"
        block = [
            f"[{start} - {end}] {speaker}:",
            str(turn.get("text") or "").strip(),
        ]
        if _speaker_is_uncertain(turn):
            block.append(
                "[Speaker attribution: uncertain; verify against the recording]"
            )
        block.extend(
            [
                f"[Observed: {observed}]",
                f"[Tone approx: {_tone_line(turn.get('tone'))}]",
                "",
            ]
        )
        lines.extend(block)
    if not payload.get("turns"):
        lines.append("[No speech was transcribed.]\n")
    return "\n".join(lines).rstrip() + "\n"


CONDENSED_MAX_PARAGRAPH_WORDS = 250
CONDENSED_TONE_MIN_PROBABILITY = 0.65
CONDENSED_TONE_MIN_MARGIN = 0.20
CONDENSED_CONTEXT_LABELS = {
    "long pause before": "long pause before",
    "overlap": "overlapping speech",
    "interruption": "interruption",
    "elevated volume": "elevated volume",
}


def _strong_tone_from_scores(scores: Any) -> str | None:
    ranked = [item for item in (scores or []) if isinstance(item, dict)]
    if not ranked:
        return None
    top = ranked[0]
    label = str(top.get("label") or "unknown").lower()
    probability = float(top.get("probability") or 0.0)
    runner_up = (
        float(ranked[1].get("probability") or 0.0) if len(ranked) > 1 else 0.0
    )
    if (
        label in {"unknown", "unclassified", "neutral"}
        or probability < CONDENSED_TONE_MIN_PROBABILITY
        or probability - runner_up < CONDENSED_TONE_MIN_MARGIN
    ):
        return None
    return label


def _condensed_tone_signal(tone: Any) -> str | None:
    if not isinstance(tone, dict) or tone.get("kind") != "approximate_model_estimate":
        return None
    if not tone.get("temporal_variation"):
        return _strong_tone_from_scores(tone.get("scores"))

    labels: list[str] = []
    for window in tone.get("windows") or []:
        label = _strong_tone_from_scores(window.get("scores"))
        if label and (not labels or labels[-1] != label):
            labels.append(label)
    unique = list(dict.fromkeys(labels))
    if len(unique) < 2:
        return None
    return f"varied ({' -> '.join(unique[:3])})"


def _condensed_paragraphs(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paragraphs: list[dict[str, Any]] = []
    for turn in turns:
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        speaker = str(turn.get("speaker") or "SPEAKER_UNKNOWN")
        uncertain = _speaker_is_uncertain(turn)
        words = len(text.split())
        can_merge = bool(
            paragraphs
            and paragraphs[-1]["speaker"] == speaker
            and paragraphs[-1]["uncertain"] == uncertain
            and float(turn.get("pause_before_seconds") or 0.0) < 2.0
            and paragraphs[-1]["word_count"] + words
            <= CONDENSED_MAX_PARAGRAPH_WORDS
        )
        if can_merge:
            paragraph = paragraphs[-1]
            paragraph["text"] = f"{paragraph['text']} {text}".strip()
            paragraph["word_count"] += words
            paragraph["turns"].append(turn)
        else:
            paragraphs.append(
                {
                    "speaker": speaker,
                    "uncertain": uncertain,
                    "text": text,
                    "word_count": words,
                    "turns": [turn],
                }
            )
    return paragraphs


def _paragraph_context(paragraph: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for turn in paragraph["turns"]:
        for observation in turn.get("observations") or []:
            rendered = CONDENSED_CONTEXT_LABELS.get(str(observation.get("label")))
            if rendered and rendered not in labels:
                labels.append(rendered)
    return labels


def _paragraph_tone(paragraph: dict[str, Any]) -> str | None:
    signals: list[str] = []
    for turn in paragraph["turns"]:
        signal = _condensed_tone_signal(turn.get("tone"))
        if signal and (not signals or signals[-1] != signal):
            signals.append(signal)
    if not signals:
        return None
    if len(signals) == 1:
        return signals[0]
    return f"varied ({' -> '.join(signals[:3])})"


def render_txt(payload: dict[str, Any]) -> str:
    """Render the concise, analysis-ready transcript written to Transcribed."""
    source = payload["source"]
    language = payload["language"]
    turns = payload.get("turns") or []
    identity = payload.get("speaker_identity") or {}
    active = [str(item) for item in identity.get("active_speaker_ids") or []]
    if not active:
        active = list(
            dict.fromkeys(
                str(turn.get("speaker") or "SPEAKER_UNKNOWN") for turn in turns
            )
        )
    labels = {profile["voice_id"]: profile for profile in payload.get("speaker_profiles", [])}
    display = []
    for voice in active:
        profile = labels.get(voice, {})
        annotations = [str(profile["label"])] if profile.get("label") and profile["label"] != voice else []
        if profile.get("role") in ("adult", "child"):
            annotations.append(str(profile["role"]))
        display.append(voice + (f" ({'; '.join(annotations)})" if annotations else ""))
    lines = [
        "DRAFT: SPEAKER REVIEW REQUIRED" if (payload.get("speaker_review") or {}).get("pending") else "ANALYSIS-READY TRANSCRIPT",
        "=========================",
        f"Source: {source.get('relative_path') or source.get('path')}",
        f"Language: {language.get('output') or language.get('detected') or 'unknown'}",
        f"Speakers: {', '.join(display) if display else 'none detected'}",
        "ASR wording is preserved and not summarized. Speaker attribution and",
        "selective vocal-tone labels are probabilistic; verify consequential passages",
        "against the recording. Full timestamps, evidence, and scores are in Review.",
        "",
        "TRANSCRIPT",
        "----------",
        "",
    ]
    for paragraph in _condensed_paragraphs(turns):
        uncertainty = (
            " [speaker attribution uncertain]" if paragraph["uncertain"] else ""
        )
        lines.extend(
            [
                f"{speaker_display(paragraph['speaker'], payload.get('speaker_profiles', []))}{uncertainty}:",
                paragraph["text"],
            ]
        )
        context = _paragraph_context(paragraph)
        if context:
            lines.append(f"[Context: {'; '.join(context)}]")
        tone = _paragraph_tone(paragraph)
        if tone:
            lines.append(f"[Vocal tone estimate: {tone}]")
        lines.append("")
    if not turns:
        lines.append("[No speech was transcribed.]\n")
    return "\n".join(lines).rstrip() + "\n"


def render_srt(segments: list[dict[str, Any]], profiles=()) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        speaker = speaker_display(segment.get("speaker"), profiles)
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


def render_vtt(segments: list[dict[str, Any]], profiles=()) -> str:
    blocks = []
    for segment in segments:
        speaker = speaker_display(segment.get("speaker"), profiles)
        text = str(segment.get("text") or "").strip()
        blocks.append(
            "\n".join(
                (
                    f"{format_timestamp(segment.get('start'))} --> "
                    f"{format_timestamp(segment.get('end'))}",
                    f"<v {escape(str(speaker))}>{escape(text)}</v>",
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
        elif format_name == "detailed_txt":
            atomic_write_text(path, render_detailed_txt(payload))
        elif format_name == "json":
            atomic_write_json(path, payload)
        elif format_name == "srt":
            atomic_write_text(path, render_srt(segments, payload.get("speaker_profiles", [])))
        elif format_name == "vtt":
            atomic_write_text(path, render_vtt(segments, payload.get("speaker_profiles", [])))
        else:
            raise ValueError(f"unsupported output format: {format_name}")
