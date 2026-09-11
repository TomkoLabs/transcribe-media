from __future__ import annotations

import math
from statistics import median
from typing import Any, Optional, Protocol

from .schema import (
    ACOUSTIC_ANALYZER_VERSION,
    HEURISTIC_TONE_VERSION,
    SAMPLE_RATE,
)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def normalize_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    def group_identity(words, fallback):
        identities = [word.get("speaker_identity") for word in words]
        if identities and all(identity == identities[0] for identity in identities):
            return identities[0] or fallback
        unresolved = next((item for item in identities if item and str(item.get("status", "")).startswith("unresolved")), None)
        if unresolved:
            return unresolved
        if any(item and item.get("status") == "human_verified" for item in identities):
            return {"status": "partially_reviewed"}
        return fallback
    normalized: list[dict[str, Any]] = []
    for segment in result.get("segments") or []:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue

        raw_words = segment.get("words") or []
        words: list[dict[str, Any]] = []
        for word in raw_words:
            word_text = str(word.get("word") or word.get("text") or "").strip()
            if not word_text:
                continue
            words.append(
                {
                    "text": word_text,
                    "start": (
                        _number(word.get("start"))
                        if word.get("start") is not None
                        else None
                    ),
                    "end": (
                        _number(word.get("end"))
                        if word.get("end") is not None
                        else None
                    ),
                    "confidence": (
                        _number(word.get("score", word.get("confidence")))
                        if word.get("score", word.get("confidence")) is not None
                        else None
                    ),
                    "speaker": word.get("speaker"),
                    "local_speaker": word.get("local_speaker"),
                    "diarization_speaker": word.get("diarization_speaker"),
                    "sortformer_speaker": word.get("sortformer_speaker"),
                    "sortformer_model_speaker": word.get(
                        "sortformer_model_speaker"
                    ),
                    "speaker_refinement": word.get("speaker_refinement"),
                    "speaker_identity": word.get("speaker_identity"),
                    "speaker_assignment_fallback": word.get("speaker_assignment_fallback", not bool(word.get("speaker"))),
                }
            )

        segment_speaker = str(segment.get("speaker") or "SPEAKER_UNKNOWN")
        segment_local_speaker = str(segment.get("local_speaker") or segment_speaker)
        for word in words:
            word["speaker"] = str(word.get("speaker") or segment_speaker)
            word["local_speaker"] = str(
                word.get("local_speaker") or segment_local_speaker
            )
            if word.get("speaker_identity") is None:
                word["speaker_identity"] = segment.get("speaker_identity")
        word_speakers = {(word["speaker"], word["local_speaker"]) for word in words}

        # Whisper/pyannote can correctly label different words inside one ASR
        # segment. Preserve those boundaries instead of flattening the whole
        # sentence to the segment's majority speaker.
        groups: list[tuple[str, str, str, list[dict[str, Any]], Any, Any, Any]] = []
        if len(word_speakers) > 1:
            current: list[dict[str, Any]] = []
            current_key: Optional[tuple[str, str]] = None
            for word in words:
                key = (str(word["speaker"]), str(word["local_speaker"]))
                if current and key != current_key:
                    assert current_key is not None
                    groups.append(
                        (
                            current_key[0],
                            current_key[1],
                            " ".join(item["text"] for item in current),
                            current,
                            current[0]["start"],
                            current[-1]["end"],
                            group_identity(current, segment.get("speaker_identity")),
                        )
                    )
                    current = []
                current_key = key
                current.append(word)
            if current:
                assert current_key is not None
                groups.append(
                    (
                        current_key[0],
                        current_key[1],
                        " ".join(item["text"] for item in current),
                        current,
                        current[0]["start"],
                        current[-1]["end"],
                        group_identity(current, segment.get("speaker_identity")),
                    )
                )
        else:
            speaker, local_speaker = next(
                iter(word_speakers), (segment_speaker, segment_local_speaker)
            )
            groups.append(
                (
                    speaker,
                    local_speaker,
                    text,
                    words,
                    segment.get("start"),
                    segment.get("end"),
                    group_identity(words, segment.get("speaker_identity")),
                )
            )

        for (
            speaker,
            local_speaker,
            group_text,
            group_words,
            start,
            end,
            speaker_identity,
        ) in groups:
            timed_words = [word for word in group_words if word["start"] is not None]
            if start is None and timed_words:
                start = timed_words[0]["start"]
            timed_ends = [word for word in group_words if word["end"] is not None]
            if end is None and timed_ends:
                end = timed_ends[-1]["end"]
            start_value = _number(start)
            end_value = max(start_value, _number(end, start_value))
            normalized.append(
                {
                    "id": f"segment-{len(normalized):06d}",
                    "start": start_value,
                    "end": end_value,
                    "speaker": speaker,
                    "local_speaker": local_speaker,
                    "diarization_speaker": segment.get("diarization_speaker"),
                    "speaker_refinement": segment.get("speaker_refinement"),
                    "speaker_identity": speaker_identity,
                    "speaker_confidence": segment.get("speaker_confidence"),
                    "asr_confidence": segment.get("score"),
                    "avg_log_probability": segment.get("avg_logprob"),
                    "no_speech_probability": segment.get("no_speech_prob"),
                    "text": group_text,
                    "words": group_words,
                }
            )
    return normalized


def build_turns(
    segments: list[dict[str, Any]],
    max_gap_seconds: float = 1.25,
    max_turn_seconds: float = 30.0,
) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for segment in segments:
        start = _number(segment["start"])
        end = max(start, _number(segment["end"], start))
        speaker = str(segment.get("speaker") or "SPEAKER_UNKNOWN")
        local_speaker = str(segment.get("local_speaker") or speaker)
        text = str(segment.get("text") or "").strip()
        if not text:
            continue

        if turns:
            current = turns[-1]
            if (
                current["speaker"] == speaker
                and current.get("speaker_identity") == segment.get("speaker_identity")
                and start - current["end"] <= max_gap_seconds
                and end - current["start"] <= max_turn_seconds
            ):
                current["end"] = end
                current["text"] = f"{current['text']} {text}".strip()
                current["segment_ids"].append(segment["id"])
                current["words"].extend(segment.get("words") or [])
                if local_speaker not in current["local_speakers"]:
                    current["local_speakers"].append(local_speaker)
                continue

        turns.append(
            {
                "id": f"turn-{len(turns):06d}",
                "start": start,
                "end": end,
                "speaker": speaker,
                "local_speakers": [local_speaker],
                "speaker_identity": segment.get("speaker_identity"),
                "speaker_confidence": segment.get("speaker_confidence"),
                "text": text,
                "segment_ids": [segment["id"]],
                "words": list(segment.get("words") or []),
                "overlaps": [],
                "observations": [],
                "acoustic": {},
                "tone": None,
            }
        )
    annotate_timing_relationships(turns)
    return turns


def annotate_speaker_attribution(
    turns: list[dict[str, Any]],
    refinement_report: Optional[dict[str, Any]] = None,
) -> None:
    """Expose evidence conflicts without pretending speaker IDs are certain.

    The final assigned speaker remains intact for conversational structure and
    cross-recording matching. Unknown or unresolved identities, model conflicts,
    and acoustically ambiguous corrections are explicitly marked for review.
    """
    suspicious_intervals: list[tuple[float, float, str]] = []
    for candidate in (refinement_report or {}).get("evaluated_candidates") or []:
        decision = str(candidate.get("decision") or "")
        strategy = str(candidate.get("strategy") or "")
        unresolved_seam = bool(
            decision == "abstained_below_confidence_threshold"
            and (
                strategy == "bracketed_utterance"
                or candidate.get("sentence_seam_continuation")
                or candidate.get("sandwiched_sentence_continuation")
            )
        )
        candidate_ambiguous = decision == "abstained_candidate_voice_ambiguous"
        if (
            decision != "abstained_sortformer_disagreement"
            and not unresolved_seam
            and not candidate_ambiguous
        ):
            continue
        start = _number(candidate.get("start"), -1.0)
        end = _number(candidate.get("end"), -1.0)
        if start < 0.0 or end <= start:
            continue
        reason = (
            "refinement_model_disagreement"
            if decision == "abstained_sortformer_disagreement"
            else (
                "candidate_voice_ambiguous"
                if candidate_ambiguous else "unresolved_sentence_continuation"
            )
        )
        suspicious_intervals.append((start, end, reason))

    for turn in turns:
        reasons: list[str] = []
        if turn.get("speaker") in (None, "", "SPEAKER_UNKNOWN"):
            reasons.append("unknown_speaker")
        identity = turn.get("speaker_identity") or {}
        if str(identity.get("status") or "").startswith("unresolved"):
            reasons.append("unresolved_voice_identity")
        compared_seconds = 0.0
        disagreement_seconds = 0.0
        for word in turn.get("words") or []:
            if word.get("speaker_assignment_fallback") and (word.get("speaker_identity") or {}).get("status") != "human_verified":
                if "unsupported_word_assignment" not in reasons:
                    reasons.append("unsupported_word_assignment")
            secondary = word.get("sortformer_speaker")
            local = word.get("local_speaker")
            if not secondary or not local:
                continue
            start = word.get("start")
            end = word.get("end")
            duration = (
                max(0.0, _number(end) - _number(start))
                if start is not None and end is not None
                else 0.2
            )
            compared_seconds += duration
            if str(secondary) != str(local):
                disagreement_seconds += duration
        disagreement_fraction = (
            disagreement_seconds / compared_seconds if compared_seconds > 0.0 else 0.0
        )
        if compared_seconds >= 0.5 and disagreement_fraction >= 0.65:
            reasons.append("sortformer_majority_disagrees")

        turn_start = _number(turn.get("start"))
        turn_end = max(turn_start, _number(turn.get("end"), turn_start))
        for start, end, reason in suspicious_intervals:
            overlap = max(0.0, min(turn_end, end) - max(turn_start, start))
            if overlap >= min(0.1, max(0.01, (turn_end - turn_start) * 0.1)):
                if reason not in reasons:
                    reasons.append(reason)

        confidence = turn.get("speaker_confidence")
        if confidence is not None and _number(confidence, 1.0) < 0.5:
            reasons.append("low_speaker_confidence")

        turn["speaker_attribution"] = {
            "status": "uncertain" if reasons else "assigned",
            "assigned_speaker": str(
                turn.get("speaker") or "SPEAKER_UNKNOWN"
            ),
            "reasons": reasons,
            "sortformer_compared_seconds": round(compared_seconds, 3),
            "sortformer_disagreement_fraction": round(disagreement_fraction, 4),
        }


def annotate_timing_relationships(turns: list[dict[str, Any]]) -> None:
    previous_end = 0.0
    for index, turn in enumerate(turns):
        turn["pause_before_seconds"] = max(0.0, turn["start"] - previous_end)
        previous_end = max(previous_end, turn["end"])
        for other in turns[index + 1 :]:
            if other["start"] >= turn["end"]:
                break
            if other["speaker"] == turn["speaker"]:
                continue
            overlap_start = max(turn["start"], other["start"])
            overlap_end = min(turn["end"], other["end"])
            if overlap_end <= overlap_start:
                continue
            duration = overlap_end - overlap_start
            turn["overlaps"].append(
                {
                    "turn_id": other["id"],
                    "speaker": other["speaker"],
                    "start": overlap_start,
                    "end": overlap_end,
                    "duration_seconds": duration,
                }
            )
            other["overlaps"].append(
                {
                    "turn_id": turn["id"],
                    "speaker": turn["speaker"],
                    "start": overlap_start,
                    "end": overlap_end,
                    "duration_seconds": duration,
                }
            )
            other["interruption_of"] = turn["id"]


def annotate_raw_overlap(turns, timeline):
    """Expose simultaneous acoustic speech even with exclusive word assignment."""
    intervals = sorted(timeline, key=lambda item: float(item.get("start", 0)))
    active = []
    events = []
    for item in intervals:
        start, end = float(item["start"]), float(item["end"])
        active = [other for other in active if float(other["end"]) > start]
        for other in active:
            if other.get("speaker") == item.get("speaker"):
                continue
            stop = min(end, float(other["end"]))
            if stop > start:
                events.append({"start": start, "end": stop, "duration_seconds": stop - start,
                               "speakers": [other["speaker"], item["speaker"]]})
        active.append(item)
    for turn in turns:
        turn["acoustic_overlap"] = [event for event in events
                                    if event["start"] < turn["end"] and event["end"] > turn["start"]]
        if turn["acoustic_overlap"]:
            turn.setdefault("observations", []).append({"label": "overlap", "basis": "diarization_timeline",
                "value": "simultaneous speech; words may be missing", "unit": ""})
            attribution = turn.setdefault("speaker_attribution", {"status": "uncertain", "reasons": []})
            attribution["status"] = "uncertain"
            attribution.setdefault("reasons", []).append("overlapping_speech")
        # An overlapping start is not sufficient evidence of an interruption.
        if turn.get("interruption_of"):
            turn["overlapping_start_with"] = turn.pop("interruption_of")


def _dbfs(samples: Any) -> float:
    import numpy as np

    if len(samples) == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1e-6))


def _pitch_statistics(samples: Any) -> dict[str, Optional[float]]:
    import numpy as np

    frame_size = int(0.04 * SAMPLE_RATE)
    if len(samples) < frame_size:
        return {"median_hz": None, "variation_ratio": None}

    possible = max(1, (len(samples) - frame_size) // frame_size + 1)
    frame_indexes = np.unique(
        np.linspace(0, possible - 1, min(60, possible)).astype(int)
    )
    pitches: list[float] = []
    min_lag = SAMPLE_RATE // 650
    max_lag = SAMPLE_RATE // 60

    for frame_index in frame_indexes:
        first = frame_index * frame_size
        frame = np.asarray(samples[first : first + frame_size], dtype=np.float32)
        frame = frame - float(frame.mean())
        energy = float(np.sqrt(np.mean(frame * frame)))
        if energy < 0.005:
            continue
        frame *= np.hanning(len(frame))
        correlation = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
        reference = float(correlation[0])
        if reference <= 0:
            continue
        search = correlation[min_lag : max_lag + 1]
        lag = int(np.argmax(search)) + min_lag
        if float(correlation[lag]) / reference < 0.3:
            continue
        pitches.append(SAMPLE_RATE / lag)

    if len(pitches) < 3:
        return {"median_hz": None, "variation_ratio": None}
    pitch_array = np.asarray(pitches, dtype=np.float32)
    pitch_median = float(np.median(pitch_array))
    deviation = float(np.std(pitch_array))
    return {
        "median_hz": round(pitch_median, 2),
        "variation_ratio": round(deviation / max(pitch_median, 1.0), 3),
    }


class AcousticAnalyzer:
    name = "internal/acoustic-observations"
    version = ACOUSTIC_ANALYZER_VERSION

    def analyze(self, audio: Any, turns: list[dict[str, Any]]) -> None:
        import numpy as np

        audio_array = np.asarray(audio, dtype=np.float32)
        measurements: list[dict[str, Any]] = []
        for turn in turns:
            first = max(0, int(turn["start"] * SAMPLE_RATE))
            last = min(len(audio_array), max(first + 1, int(turn["end"] * SAMPLE_RATE)))
            samples = audio_array[first:last]
            duration = max(0.001, turn["end"] - turn["start"])
            word_count = len(turn.get("words") or []) or len(turn["text"].split())
            first_half, second_half = np.array_split(samples, 2)
            pitch = _pitch_statistics(samples)
            measurements.append(
                {
                    "rms_dbfs": round(_dbfs(samples), 2),
                    "peak_dbfs": round(
                        20.0
                        * math.log10(
                            max(float(np.max(np.abs(samples), initial=0.0)), 1e-6)
                        ),
                        2,
                    ),
                    "speech_rate_words_per_second": round(word_count / duration, 3),
                    "energy_change_db": round(
                        _dbfs(second_half) - _dbfs(first_half), 2
                    ),
                    "pitch_median_hz": pitch["median_hz"],
                    "pitch_variation_ratio": pitch["variation_ratio"],
                }
            )

        baseline = (
            median([item["rms_dbfs"] for item in measurements])
            if measurements
            else -30.0
        )
        levels = {}
        for turn, item in zip(turns, measurements, strict=True):
            if not turn.get("acoustic_overlap"):
                levels.setdefault(turn.get("speaker"), []).append(item["rms_dbfs"])
        baselines = {speaker: median(values) for speaker, values in levels.items()}
        for turn, acoustic in zip(turns, measurements, strict=True):
            speaker_baseline = baselines.get(turn.get("speaker"), baseline)
            acoustic["volume_reference"] = "same_speaker_in_this_recording"
            acoustic["overlap_contaminated"] = bool(turn.get("acoustic_overlap"))
            acoustic["relative_volume_db"] = round(acoustic["rms_dbfs"] - speaker_baseline, 2)
            turn["acoustic"] = acoustic
            observations: list[dict[str, Any]] = list(turn.get("observations") or [])

            def observe(
                label: str,
                basis: str,
                value: Any,
                unit: str,
                destination: list[dict[str, Any]] = observations,
            ) -> None:
                destination.append(
                    {"label": label, "basis": basis, "value": value, "unit": unit}
                )

            relative_volume = acoustic["relative_volume_db"]
            speech_rate = acoustic["speech_rate_words_per_second"]
            if relative_volume >= 4.0:
                observe("elevated volume", "relative_volume_db", relative_volume, "dB")
            elif relative_volume <= -6.0:
                observe("quiet speech", "relative_volume_db", relative_volume, "dB")
            if speech_rate >= 3.2:
                observe(
                    "fast speech",
                    "speech_rate_words_per_second",
                    speech_rate,
                    "words/s",
                )
            elif speech_rate <= 1.2 and len(turn["text"].split()) >= 3:
                observe(
                    "slow speech",
                    "speech_rate_words_per_second",
                    speech_rate,
                    "words/s",
                )
            if turn.get("pause_before_seconds", 0.0) >= 2.0:
                observe(
                    "long pause before",
                    "pause_before_seconds",
                    round(turn["pause_before_seconds"], 3),
                    "seconds",
                )
            if abs(acoustic["energy_change_db"]) >= 6.0:
                observe(
                    "substantial energy change",
                    "energy_change_db",
                    acoustic["energy_change_db"],
                    "dB",
                )
            if (acoustic["pitch_variation_ratio"] or 0.0) >= 0.25:
                observe(
                    "substantial pitch variation",
                    "pitch_variation_ratio",
                    acoustic["pitch_variation_ratio"],
                    "ratio",
                )
            if turn.get("overlaps"):
                observe(
                    "overlap",
                    "overlap_duration_seconds",
                    round(
                        sum(item["duration_seconds"] for item in turn["overlaps"]), 3
                    ),
                    "seconds",
                )
            if turn.get("interruption_of"):
                observe("interruption", "timing", turn["interruption_of"], "turn_id")
            turn["observations"] = observations


class ToneEstimator(Protocol):
    name: str
    version: str

    def estimate(self, audio: Any, turns: list[dict[str, Any]]) -> None: ...


class HeuristicToneEstimator:
    name = "internal/acoustic-tone-heuristic"
    version = HEURISTIC_TONE_VERSION

    def estimate(self, audio: Any, turns: list[dict[str, Any]]) -> None:
        del audio
        for turn in turns:
            acoustic = turn.get("acoustic") or {}
            volume = _number(acoustic.get("relative_volume_db"))
            rate = _number(acoustic.get("speech_rate_words_per_second"), 2.0)
            pitch_variation = _number(acoustic.get("pitch_variation_ratio"), 0.1)
            energy_change = abs(_number(acoustic.get("energy_change_db")))

            activation = max(
                0.0,
                min(
                    1.0,
                    0.45 + volume / 16.0 + (rate - 2.0) / 5.0 + pitch_variation / 2.0,
                ),
            )
            instability = max(
                0.0, min(1.0, pitch_variation * 1.8 + energy_change / 20.0)
            )
            raw = {
                "neutral": 1.25 - abs(activation - 0.45) - instability * 0.35,
                "activated": 0.35 + activation * 0.9,
                "subdued": 0.35 + (1.0 - activation) * 0.75,
                "tense": 0.2 + activation * 0.45 + instability * 0.55,
            }
            exponentials = {label: math.exp(value) for label, value in raw.items()}
            total = sum(exponentials.values())
            scores = sorted(
                (
                    {"label": label, "probability": round(value / total, 4)}
                    for label, value in exponentials.items()
                ),
                key=lambda item: item["probability"],
                reverse=True,
            )[:3]
            turn["tone"] = {
                "kind": "approximate_estimate",
                "model": self.name,
                "model_version": self.version,
                "scores": scores,
                "limitations": (
                    "Low-confidence heuristic derived only from measured energy, rate, "
                    "and pitch variation; it is not a statement of internal emotion."
                ),
            }
