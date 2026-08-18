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
                        _number(word.get("score"))
                        if word.get("score") is not None
                        else None
                    ),
                    "speaker": word.get("speaker"),
                    "local_speaker": word.get("local_speaker"),
                    "diarization_speaker": word.get("diarization_speaker"),
                    "speaker_refinement": word.get("speaker_refinement"),
                    "speaker_identity": word.get("speaker_identity"),
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
                            current[0].get("speaker_identity"),
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
                        current[0].get("speaker_identity"),
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
                    segment.get("speaker_identity"),
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
    min_lag = SAMPLE_RATE // 350
    max_lag = SAMPLE_RATE // 70

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
        for turn, acoustic in zip(turns, measurements, strict=True):
            acoustic["relative_volume_db"] = round(acoustic["rms_dbfs"] - baseline, 2)
            turn["acoustic"] = acoustic
            observations: list[dict[str, Any]] = []

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
