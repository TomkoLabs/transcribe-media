"""Contiguous reference speech and auditable window-level identity evidence."""
from __future__ import annotations

from typing import Any

from .schema import SAMPLE_RATE

EVIDENCE_VERSION = "2.2"


def identity_evidence_summary(item):
    """Distinguish acoustic outliers from disagreement between diarizers.

    Derive this for older packets too, without modifying source-bound evidence.
    Opaque legacy mixed flags remain conservative when detail is unavailable.
    """
    windows = item.get('windows', [])
    detailed = bool(windows) and all(isinstance(w.get('model_disagreement'), bool)
                                    and isinstance(w.get('retained'), bool) for w in windows)
    compared = [w for w in windows if not w.get('model_disagreement')]
    outliers = sum(w.get('retained') is False for w in compared)
    mixed = (len(compared) >= 3 and outliers / len(compared) > .20 + 1e-9) if detailed else bool(item.get('suspected_mixed_speakers'))
    return {'mixed_voice_evidence': mixed, 'acoustic_outlier_windows': outliers,
            'model_disputed_windows': sum(bool(w.get('model_disagreement')) for w in windows)}


def extract_reference_evidence(audio, timeline, result, encode):
    import numpy as np
    from .backends import _exclusive_intervals

    words = [word for segment in result.get("segments", [])
             for word in segment.get("words", [])
             if word.get("start") is not None and word.get("end") is not None]
    evidence = {}
    for speaker in sorted({str(item["speaker"]) for item in timeline
                           if item.get("speaker") not in (None, "", "SPEAKER_UNKNOWN")}):
        intervals = _exclusive_intervals(timeline, speaker)
        candidates = []
        rejected = {"short": 0, "clipping_or_silence": 0, "model_disagreement": 0,
                    "little_recognized_speech": 0, "outlier": 0}
        own_words = [word for word in words if word.get("speaker") == speaker]
        for start, end in intervals:
            # Keep the true waveform and trim change boundaries. Never stitch replies.
            start, end = start + 0.10, end - 0.10
            if end - start < 0.8 - 1e-6:
                rejected["short"] += 1
                continue
            cursor = start
            while end - cursor >= 0.8 - 1e-6:
                stop = min(end, cursor + 6.0)
                candidates.append((cursor, stop))
                cursor = stop
        # Spread a bounded number of windows across the whole recording.
        if len(candidates) > 64:
            indexes = np.linspace(0, len(candidates) - 1, 64).astype(int)
            candidates = [candidates[index] for index in indexes]
        windows = []
        for start, end in candidates:
            duration = round(end - start, 4)
            relevant = [word for word in own_words
                        if float(word["start"]) < end and float(word["end"]) > start]
            disputed = any(word.get("sortformer_ambiguous") or word.get("sortformer_speaker") not in (None, speaker)
                           for word in relevant)
            if disputed:
                rejected["model_disagreement"] += 1
            speech = sum(max(0., min(end, float(word["end"])) - max(start, float(word["start"])))
                         for word in relevant if any(c.isalnum() for c in str(word.get("word", word.get("text", "")))))
            if words and speech < min(0.6, duration * 0.25):
                rejected["little_recognized_speech"] += 1
                continue
            samples = np.asarray(audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)], dtype=np.float32)
            if not len(samples) or not np.isfinite(samples).all():
                rejected["clipping_or_silence"] += 1
                continue
            rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
            clipped = float(np.mean(np.abs(samples) >= 0.995))
            if rms < 0.002 or clipped > 0.01:
                rejected["clipping_or_silence"] += 1
                continue
            embedding = encode(samples)
            if embedding is None:
                continue
            embedding = np.asarray(embedding, dtype=np.float32)
            windows.append({"start": round(start, 4), "end": round(end, 4),
                            "duration": round(duration, 4), "embedding": embedding.tolist(),
                            "rms_dbfs": round(20 * float(np.log10(rms)), 2),
                            "clipped_fraction": round(clipped, 4),
                            "reference_eligible": duration >= 0.8 and speech >= 0.2,
                            "model_disagreement": disputed,
                            "automatic_reference_eligible": not disputed and duration >= 2.5 and speech >= 0.6})
        if not windows:
            continue
        matrix = np.asarray([window["embedding"] for window in windows], dtype=np.float32)
        pairwise = matrix @ matrix.T
        eligible = np.asarray([not w["model_disagreement"] for w in windows])
        indexes = np.flatnonzero(eligible)
        # Manual-only clips must not move the automatic reference anchor.
        medoid = int(indexes[np.argmax(np.median(pairwise[np.ix_(indexes, indexes)], axis=1))]) if len(indexes) else int(np.argmax(np.median(pairwise, axis=1)))
        agreement = pairwise[:, medoid]
        keep = (agreement >= 0.45) & eligible
        retained = matrix[keep]
        centroid = np.mean(retained if len(retained) else matrix, axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
        for index, window in enumerate(windows):
            window["cohesion_similarity"] = round(float(matrix[index] @ centroid), 4)
            window["retained"] = bool(keep[index])
            window["automatic_reference_eligible"] &= bool(keep[index])
        rejected["outlier"] = int((~keep & eligible).sum())
        mixed = identity_evidence_summary({'windows': windows})['mixed_voice_evidence']
        good = [window for window in windows if window["retained"]]
        evidence[speaker] = {
            "embedding": [round(float(value), 7) for value in centroid],
            "clean_seconds": round(sum(window["duration"] for window in good), 3),
            "available_clean_seconds": round(sum(end - start for start, end in intervals), 3),
            "window_count": len(good), "cohesion": round(float(np.median(retained @ centroid)), 4) if len(retained) else 0.,
            "recognized_word_count": len(own_words),
            "recognized_speech_seconds": round(sum(max(0, float(w["end"]) - float(w["start"])) for w in own_words), 3),
            "word_timing_available": bool(words), "lexical_filter_applied": bool(words),
            "windows": windows, "suspected_mixed_speakers": mixed,
            "rejected_windows": rejected, "evidence_version": EVIDENCE_VERSION,
        }
    return evidence
