import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from transcribe_media_app import cli
from transcribe_media_app.analysis import build_turns, normalize_segments
from transcribe_media_app.backends import (
    SUPPORTED_CTRANSLATE2_VERSION,
    Emotion2VecToneEstimator,
    PyannoteDiarizer,
    SpeakerIdentityEncoder,
    WhisperXBackend,
    _exclusive_intervals,
    _recognized_speech_intervals,
)
from transcribe_media_app.renderers import (
    format_timestamp,
    render_srt,
    render_txt,
    render_vtt,
)
from transcribe_media_app.schema import RESULT_SCHEMA_VERSION, RuntimeSettings
from transcribe_media_app.speakers import (
    SpeakerRegistryError,
    VoiceRegistry,
    apply_speaker_identities,
    overlapping_speaker_pairs,
)
from transcribe_media_app.storage import (
    ManifestStore,
    atomic_write_text,
    discover_media,
    expected_outputs,
    parse_extensions,
    source_fingerprint,
    state_is_complete,
)


class FormattingTests(unittest.TestCase):
    def test_timestamp_rounding(self):
        self.assertEqual(format_timestamp(3661.2346), "01:01:01.235")
        self.assertEqual(format_timestamp(1.2, ","), "00:00:01,200")
        self.assertEqual(format_timestamp(float("nan")), "00:00:00.000")

    def test_primary_txt_separates_observation_from_tone(self):
        payload = {
            "source": {"relative_path": "Café clip.mp4"},
            "language": {"output": "en", "task": "transcribe"},
            "processing": {
                "completed_utc": "2026-01-01T00:00:00+00:00",
                "runtime": {"device": "cpu"},
                "provenance": {"transcription_model": "tiny.en"},
                "degraded_stages": [],
            },
            "turns": [
                {
                    "start": 1,
                    "end": 2.5,
                    "speaker": "SPEAKER_00",
                    "text": "Hello there.",
                    "observations": [{"label": "fast speech"}],
                    "tone": {
                        "model": "test-model",
                        "scores": [{"label": "neutral", "probability": 0.6}],
                    },
                }
            ],
        }
        text = render_txt(payload)
        self.assertIn("[00:00:01.000 - 00:00:02.500] SPEAKER_00:", text)
        self.assertIn("[Observed: fast speech]", text)
        self.assertIn("[Tone approx: neutral 60%; model: test-model]", text)
        self.assertIn("not facts about emotion", text)

    def test_primary_txt_distinguishes_model_abstention_from_failure(self):
        tone = {
            "model": "test-model",
            "scores": [
                {"label": "unknown", "probability": 0.64},
                {"label": "angry", "probability": 0.26},
            ],
            "windows": [
                {
                    "start": 1.0,
                    "end": 7.0,
                    "scores": [{"label": "unknown", "probability": 0.64}],
                },
                {
                    "start": 7.0,
                    "end": 13.0,
                    "scores": [{"label": "neutral", "probability": 0.71}],
                },
            ],
            "temporal_variation": True,
        }
        payload = {
            "source": {"relative_path": "clip.wav"},
            "language": {"output": "en", "task": "transcribe"},
            "processing": {
                "completed_utc": "2026-01-01T00:00:00+00:00",
                "runtime": {"device": "cpu"},
                "provenance": {"transcription_model": "tiny.en"},
                "degraded_stages": [],
            },
            "turns": [
                {
                    "start": 1,
                    "end": 13,
                    "speaker": "SPEAKER_00",
                    "text": "A mixed long turn.",
                    "tone": tone,
                }
            ],
        }
        text = render_txt(payload)
        self.assertIn(
            "[Tone approx: varies across 2 windows; unclassified 64%, angry 26%; "
            "model: test-model]",
            text,
        )
        self.assertNotIn("Tone approx: unknown", text)

    def test_primary_txt_separates_active_speakers_from_registry_history(self):
        payload = {
            "source": {"relative_path": "clip.mp4"},
            "language": {"output": "en", "task": "transcribe"},
            "processing": {
                "completed_utc": "2026-01-01T00:00:00+00:00",
                "runtime": {"device": "cpu"},
                "provenance": {"transcription_model": "tiny.en"},
                "degraded_stages": [],
            },
            "speaker_identity": {
                "registry_revision": 7,
                "local_clusters_detected": 3,
                "speaker_groups_after_reconciliation": 3,
                "active_speaker_count": 3,
                "active_speaker_ids": [
                    "VOICE_0001",
                    "VOICE_0004",
                    "VOICE_0006",
                ],
                "profile_count": 6,
            },
            "speaker_refinement": {"corrections_applied": 2},
            "turns": [],
        }
        text = render_txt(payload)
        self.assertIn(
            "Active speakers in this recording: 3 (VOICE_0001, VOICE_0004, VOICE_0006)",
            text,
        )
        self.assertIn(
            "Project voice registry: 6 profile(s) total across recordings", text
        )
        self.assertIn("Speaker refinement: 2 label correction(s)", text)
        self.assertIn("Speaker labels are probabilistic", text)

    def test_srt_and_vtt_include_speaker_and_timestamp(self):
        segments = [{"start": 1, "end": 2.5, "speaker": "SPEAKER_00", "text": "Hi"}]
        self.assertIn("00:00:01,000 --> 00:00:02,500", render_srt(segments))
        self.assertIn("[SPEAKER_00] Hi", render_srt(segments))
        self.assertIn("00:00:01.000 --> 00:00:02.500", render_vtt(segments))


class StorageTests(unittest.TestCase):
    def test_extensions_are_normalized(self):
        self.assertEqual(parse_extensions("MP4, .Wav"), {".mp4", ".wav"})

    def test_discovery_is_unicode_safe_and_excludes_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Video Source"
            review = root / "Review"
            transcript = root / "Transcribed"
            (source / "nested").mkdir(parents=True)
            review.mkdir()
            transcript.mkdir()
            media = source / "nested" / "相談 clip.mp4"
            media.write_bytes(b"not actually valid; processing must report that")
            (review / "generated.wav").write_bytes(b"output")
            found = discover_media(source, (review, transcript), True, None)
            self.assertEqual(found, [media])

    def test_outputs_preserve_source_extension_and_split_directories(self):
        source_dir = Path("/source")
        source = source_dir / "clip.mp4"
        outputs = expected_outputs(
            source,
            source_dir,
            Path("/txt"),
            Path("/review"),
            ("json", "srt"),
        )
        self.assertEqual(outputs["txt"], Path("/txt/clip.mp4.txt"))
        self.assertEqual(outputs["json"], Path("/review/clip.mp4.json"))
        self.assertEqual(outputs["srt"], Path("/review/clip.mp4.srt"))

    def test_fingerprint_changes_when_content_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "clip.wav"
            source.write_bytes(b"first")
            first = source_fingerprint(source)
            source.write_bytes(b"other")
            second = source_fingerprint(source)
            self.assertNotEqual(first["sample_sha256"], second["sample_sha256"])

    def test_completion_requires_manifest_fingerprint_settings_and_valid_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            txt = root / "clip.txt"
            structured = root / "clip.json"
            txt.write_text("transcript", encoding="utf-8")
            structured.write_text(
                json.dumps(
                    {
                        "schema_version": RESULT_SCHEMA_VERSION,
                        "turns": [],
                        "processing": {},
                    }
                ),
                encoding="utf-8",
            )
            fingerprint = {"size": 10, "mtime_ns": 2, "sample_sha256": "abc"}
            state = {
                "status": "complete",
                "completion_state": True,
                "source_fingerprint": fingerprint,
                "settings_hash": "settings",
                "retry_recommended": False,
            }
            outputs = {"txt": txt, "json": structured}
            self.assertTrue(
                state_is_complete(state, fingerprint, "settings", outputs)[0]
            )
            self.assertFalse(
                state_is_complete(
                    state, {**fingerprint, "size": 11}, "settings", outputs
                )[0]
            )
            self.assertFalse(
                state_is_complete(state, fingerprint, "different", outputs)[0]
            )
            structured.write_text("{}", encoding="utf-8")
            self.assertFalse(
                state_is_complete(state, fingerprint, "settings", outputs)[0]
            )
            state["retry_recommended"] = True
            structured.write_text(
                json.dumps(
                    {
                        "schema_version": RESULT_SCHEMA_VERSION,
                        "turns": [],
                        "processing": {},
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(
                state_is_complete(state, fingerprint, "settings", outputs)[0]
            )

    def test_manifest_and_text_writes_are_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "nested" / "result.txt"
            atomic_write_text(path, "complete\n")
            self.assertEqual(path.read_text(), "complete\n")
            self.assertEqual(list(path.parent.glob("*.tmp")), [])
            manifest = ManifestStore(root / "Review" / "transcription_manifest.json")
            manifest.update("clip.mp4", {"status": "processing"})
            loaded = json.loads(manifest.path.read_text())
            self.assertEqual(loaded["sources"]["clip.mp4"]["status"], "processing")


class AnalysisTests(unittest.TestCase):
    def test_normalizes_word_confidence_and_merges_turns(self):
        segments = normalize_segments(
            {
                "segments": [
                    {
                        "start": 0,
                        "end": 1,
                        "text": " Hello ",
                        "speaker": "SPEAKER_00",
                        "words": [
                            {"word": "Hello", "start": 0.1, "end": 0.8, "score": 0.82}
                        ],
                    },
                    {"start": 1.1, "end": 2, "text": "there", "speaker": "SPEAKER_00"},
                ]
            }
        )
        turns = build_turns(segments)
        self.assertEqual(turns[0]["text"], "Hello there")
        self.assertEqual(turns[0]["words"][0]["confidence"], 0.82)

    def test_word_level_speaker_change_splits_an_asr_segment(self):
        segments = normalize_segments(
            {
                "segments": [
                    {
                        "start": 0,
                        "end": 2,
                        "text": "How are you? Fine.",
                        "speaker": "SPEAKER_00",
                        "words": [
                            {
                                "word": "How",
                                "start": 0,
                                "end": 0.4,
                                "speaker": "SPEAKER_00",
                            },
                            {
                                "word": "are you?",
                                "start": 0.4,
                                "end": 1.0,
                                "speaker": "SPEAKER_00",
                            },
                            {
                                "word": "Fine.",
                                "start": 1.2,
                                "end": 1.8,
                                "speaker": "SPEAKER_01",
                            },
                        ],
                    }
                ]
            }
        )
        self.assertEqual(
            [item["speaker"] for item in segments],
            ["SPEAKER_00", "SPEAKER_01"],
        )
        self.assertEqual([item["text"] for item in segments], ["How are you?", "Fine."])

    def test_emotion2vec_labels_are_normalized(self):
        self.assertEqual(Emotion2VecToneEstimator._label("中立/neutral"), "neutral")
        self.assertEqual(Emotion2VecToneEstimator._label("<unk>"), "unknown")

    def test_emotion2vec_scores_are_ranked_without_forcing_unknown(self):
        scores = Emotion2VecToneEstimator._rank_scores(
            ["4/neutral", "8/<unk>", "0/angry"],
            [0.2, 0.6, 0.15],
        )
        self.assertEqual(
            scores,
            [
                {"label": "unknown", "probability": 0.6},
                {"label": "neutral", "probability": 0.2},
                {"label": "angry", "probability": 0.15},
            ],
        )

    def test_overlap_and_interruption_are_explicit(self):
        segments = normalize_segments(
            {
                "segments": [
                    {"start": 0, "end": 3, "text": "First", "speaker": "SPEAKER_00"},
                    {"start": 2, "end": 4, "text": "Second", "speaker": "SPEAKER_01"},
                ]
            }
        )
        turns = build_turns(segments)
        self.assertAlmostEqual(turns[0]["overlaps"][0]["duration_seconds"], 1)
        self.assertEqual(turns[1]["interruption_of"], turns[0]["id"])


class SpeakerIdentityTests(unittest.TestCase):
    @staticmethod
    def _evidence(vector, seconds=20.0, cohesion=0.8):
        return {
            "embedding": vector,
            "clean_seconds": seconds,
            "window_count": 5,
            "cohesion": cohesion,
        }

    def test_first_recording_enrolls_and_swapped_local_labels_stay_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speaker_registry.json"
            registry = VoiceRegistry(path)
            first = registry.identify(
                source_key="first.mp4",
                source_fingerprint="first",
                local_speakers=("SPEAKER_00", "SPEAKER_01"),
                evidence={
                    "SPEAKER_00": self._evidence([1.0, 0.0, 0.0]),
                    "SPEAKER_01": self._evidence([0.0, 1.0, 0.0]),
                },
            )
            second = registry.identify(
                source_key="second.mp4",
                source_fingerprint="second",
                local_speakers=("SPEAKER_00", "SPEAKER_01"),
                evidence={
                    "SPEAKER_00": self._evidence([0.01, 0.99, 0.0]),
                    "SPEAKER_01": self._evidence([0.99, 0.01, 0.0]),
                },
            )

            self.assertEqual(
                [item["speaker"] for item in first["matches"]],
                ["VOICE_0001", "VOICE_0002"],
            )
            self.assertEqual(
                [item["status"] for item in first["matches"]],
                ["enrolled", "enrolled"],
            )
            self.assertEqual(
                [item["speaker"] for item in second["matches"]],
                ["VOICE_0002", "VOICE_0001"],
            )
            self.assertTrue(
                all(item["status"] == "matched" for item in second["matches"])
            )
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertNotIn('"embedding":', json.dumps(second))

    def test_high_voice_id_does_not_inflate_active_recording_speaker_count(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = VoiceRegistry(Path(directory) / "speaker_registry.json")
            vectors = [
                [1.0 if index == dimension else 0.0 for index in range(6)]
                for dimension in range(6)
            ]
            for index, vector in enumerate(vectors, start=1):
                report = registry.identify(
                    source_key=f"history-{index}.mp4",
                    source_fingerprint=f"history-{index}",
                    local_speakers=("SPEAKER_00",),
                    evidence={"SPEAKER_00": self._evidence(vector)},
                )
                self.assertEqual(report["profile_count"], index)

            current = registry.identify(
                source_key="current.mp4",
                source_fingerprint="current",
                local_speakers=("SPEAKER_00", "SPEAKER_01", "SPEAKER_02"),
                evidence={
                    "SPEAKER_00": self._evidence(vectors[0]),
                    "SPEAKER_01": self._evidence(vectors[2]),
                    "SPEAKER_02": self._evidence(vectors[5]),
                },
                minimum_groups=2,
            )
            self.assertEqual(current["profile_count"], 6)
            self.assertEqual(current["active_speaker_count"], 3)
            self.assertEqual(
                current["active_speaker_ids"],
                ["VOICE_0001", "VOICE_0003", "VOICE_0006"],
            )
            self.assertEqual(current["scope"], "whole_recording")

    def test_oversegmented_local_clusters_enroll_one_voice(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = VoiceRegistry(Path(directory) / "speaker_registry.json")
            report = registry.identify(
                source_key="split.mp4",
                source_fingerprint="split",
                local_speakers=("SPEAKER_00", "SPEAKER_01", "SPEAKER_02"),
                evidence={
                    "SPEAKER_00": self._evidence([1.0, 0.0, 0.0], seconds=8.0),
                    "SPEAKER_01": self._evidence([0.99, 0.01, 0.0], seconds=8.0),
                    "SPEAKER_02": self._evidence([0.0, 1.0, 0.0]),
                },
            )
            decisions = {item["local_speaker"]: item for item in report["matches"]}
            self.assertEqual(decisions["SPEAKER_00"]["speaker"], "VOICE_0001")
            self.assertEqual(decisions["SPEAKER_01"]["speaker"], "VOICE_0001")
            self.assertTrue(decisions["SPEAKER_00"]["local_cluster_merged"])
            self.assertEqual(decisions["SPEAKER_02"]["speaker"], "VOICE_0002")
            self.assertEqual(report["local_clusters_detected"], 3)
            self.assertEqual(report["speaker_groups_after_reconciliation"], 2)
            self.assertEqual(report["merged_local_clusters"], 1)
            self.assertEqual(report["profile_count"], 2)

    def test_overlapping_similar_clusters_are_not_merged(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = VoiceRegistry(Path(directory) / "speaker_registry.json")
            report = registry.identify(
                source_key="overlap.mp4",
                source_fingerprint="overlap",
                local_speakers=("SPEAKER_00", "SPEAKER_01"),
                evidence={
                    "SPEAKER_00": self._evidence([1.0, 0.0, 0.0]),
                    "SPEAKER_01": self._evidence([0.99, 0.01, 0.0]),
                },
                incompatible_pairs=(frozenset(("SPEAKER_00", "SPEAKER_01")),),
            )
            self.assertEqual(report["speaker_groups_after_reconciliation"], 2)
            self.assertEqual(report["merged_local_clusters"], 0)
            self.assertEqual(report["profile_count"], 2)
            self.assertNotEqual(
                report["matches"][0]["speaker"], report["matches"][1]["speaker"]
            )

    def test_reconciliation_respects_a_user_supplied_speaker_minimum(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = VoiceRegistry(Path(directory) / "speaker_registry.json")
            report = registry.identify(
                source_key="known-count.mp4",
                source_fingerprint="known-count",
                local_speakers=("SPEAKER_00", "SPEAKER_01", "SPEAKER_02"),
                evidence={
                    "SPEAKER_00": self._evidence([1.0, 0.0, 0.0]),
                    "SPEAKER_01": self._evidence([0.99, 0.01, 0.0]),
                    "SPEAKER_02": self._evidence([0.0, 1.0, 0.0]),
                },
                minimum_groups=3,
            )
            self.assertEqual(report["minimum_speaker_groups"], 3)
            self.assertEqual(report["speaker_groups_after_reconciliation"], 3)
            self.assertEqual(report["merged_local_clusters"], 0)
            self.assertEqual(report["profile_count"], 3)

    def test_nonlexical_sound_does_not_create_persistent_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = VoiceRegistry(Path(directory) / "speaker_registry.json")
            evidence = self._evidence([1.0, 0.0, 0.0])
            evidence.update(
                {
                    "word_timing_available": True,
                    "recognized_word_count": 0,
                    "recognized_speech_seconds": 0.0,
                }
            )
            report = registry.identify(
                source_key="vocalization.mp4",
                source_fingerprint="vocalization",
                local_speakers=("SPEAKER_00",),
                evidence={"SPEAKER_00": evidence},
            )
            self.assertEqual(
                report["matches"][0]["status"], "unresolved_insufficient_audio"
            )
            self.assertEqual(report["profile_count"], 0)

    def test_nonlexical_sound_does_not_match_an_existing_voice(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = VoiceRegistry(Path(directory) / "speaker_registry.json")
            registry.identify(
                source_key="speech.mp4",
                source_fingerprint="speech",
                local_speakers=("SPEAKER_00",),
                evidence={"SPEAKER_00": self._evidence([1.0, 0.0, 0.0])},
            )
            evidence = self._evidence([1.0, 0.0, 0.0])
            evidence.update(
                {
                    "word_timing_available": True,
                    "recognized_word_count": 0,
                    "recognized_speech_seconds": 0.0,
                }
            )
            report = registry.identify(
                source_key="sound.mp4",
                source_fingerprint="sound",
                local_speakers=("SPEAKER_00",),
                evidence={"SPEAKER_00": evidence},
            )
            self.assertEqual(
                report["matches"][0]["status"], "unresolved_insufficient_audio"
            )
            self.assertEqual(report["matches"][0]["speaker"], "SPEAKER_00")
            self.assertEqual(report["registry_revision"], 1)

    def test_ambiguous_voice_abstains_without_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = VoiceRegistry(Path(directory) / "speaker_registry.json")
            first = registry.identify(
                source_key="first.mp4",
                source_fingerprint="first",
                local_speakers=("SPEAKER_00", "SPEAKER_01"),
                evidence={
                    "SPEAKER_00": self._evidence([1.0, 0.0, 0.0]),
                    "SPEAKER_01": self._evidence([0.0, 1.0, 0.0]),
                },
            )
            ambiguous = registry.identify(
                source_key="ambiguous.mp4",
                source_fingerprint="ambiguous",
                local_speakers=("SPEAKER_00",),
                evidence={
                    "SPEAKER_00": self._evidence([0.71, 0.70, 0.0]),
                },
            )
            decision = ambiguous["matches"][0]
            self.assertEqual(decision["speaker"], "SPEAKER_00")
            self.assertEqual(decision["status"], "unresolved_ambiguous")
            self.assertEqual(ambiguous["registry_revision"], first["registry_revision"])
            self.assertEqual(ambiguous["profile_count"], 2)

    def test_distinct_new_voice_is_enrolled_and_rerun_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = VoiceRegistry(Path(directory) / "speaker_registry.json")
            first = registry.identify(
                source_key="first.mp4",
                source_fingerprint="first",
                local_speakers=("SPEAKER_00",),
                evidence={"SPEAKER_00": self._evidence([1.0, 0.0, 0.0])},
            )
            rerun = registry.identify(
                source_key="first.mp4",
                source_fingerprint="first",
                local_speakers=("SPEAKER_00",),
                evidence={"SPEAKER_00": self._evidence([1.0, 0.0, 0.0])},
            )
            novel = registry.identify(
                source_key="guest.mp4",
                source_fingerprint="guest",
                local_speakers=("SPEAKER_00",),
                evidence={"SPEAKER_00": self._evidence([0.0, 0.0, 1.0])},
            )
            self.assertEqual(rerun["registry_revision"], first["registry_revision"])
            self.assertEqual(novel["matches"][0]["status"], "enrolled")
            self.assertEqual(novel["matches"][0]["speaker"], "VOICE_0002")
            self.assertEqual(novel["profile_count"], 2)

    def test_insufficient_voice_stays_local(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = VoiceRegistry(Path(directory) / "speaker_registry.json")
            report = registry.identify(
                source_key="short.mp4",
                source_fingerprint="short",
                local_speakers=("SPEAKER_00",),
                evidence={"SPEAKER_00": self._evidence([1.0, 0.0, 0.0], seconds=2.0)},
            )
            self.assertEqual(
                report["matches"][0]["status"],
                "unresolved_insufficient_audio",
            )
            self.assertEqual(report["matches"][0]["speaker"], "SPEAKER_00")
            self.assertEqual(report["profile_count"], 0)

    def test_corrupt_registry_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speaker_registry.json"
            path.write_text("not json", encoding="utf-8")
            registry = VoiceRegistry(path)
            with self.assertRaisesRegex(SpeakerRegistryError, "unreadable"):
                registry.validate()
            self.assertEqual(path.read_text(encoding="utf-8"), "not json")

    def test_registry_state_hash_covers_private_embedding_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speaker_registry.json"
            registry = VoiceRegistry(path)
            original = registry.identify(
                source_key="first.mp4",
                source_fingerprint="first",
                local_speakers=("SPEAKER_00",),
                evidence={"SPEAKER_00": self._evidence([1.0, 0.0, 0.0])},
            )
            data = json.loads(path.read_text(encoding="utf-8"))
            data["profiles"]["VOICE_0001"]["centroid"] = [0.99, 0.01, 0.0]
            data["profiles"]["VOICE_0001"]["observations"][0]["embedding"] = [
                0.99,
                0.01,
                0.0,
            ]
            path.write_text(json.dumps(data), encoding="utf-8")
            changed = registry.validate()
            self.assertNotEqual(original["registry_state_hash"], changed["state_hash"])

    def test_identity_mapping_preserves_recording_local_labels(self):
        result = {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "hello",
                    "words": [{"word": "hello", "speaker": "SPEAKER_00"}],
                }
            ],
            "speaker_timeline": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
        }
        report = {
            "matches": [
                {
                    "local_speaker": "SPEAKER_00",
                    "speaker": "VOICE_0007",
                    "status": "matched",
                    "similarity": 0.81,
                    "margin": 0.3,
                }
            ]
        }
        apply_speaker_identities(result, report)
        segment = result["segments"][0]
        self.assertEqual(segment["speaker"], "VOICE_0007")
        self.assertEqual(segment["local_speaker"], "SPEAKER_00")
        self.assertEqual(segment["words"][0]["speaker"], "VOICE_0007")
        normalized = normalize_segments(result)
        self.assertEqual(normalized[0]["speaker"], "VOICE_0007")
        self.assertEqual(normalized[0]["local_speaker"], "SPEAKER_00")

    def test_overlap_is_removed_before_voice_embedding(self):
        timeline = [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 5.0, "speaker": "SPEAKER_01"},
        ]
        self.assertEqual(_exclusive_intervals(timeline, "SPEAKER_00"), [(0.08, 1.92)])
        self.assertEqual(_exclusive_intervals(timeline, "SPEAKER_01"), [(3.08, 4.92)])

    def test_meaningful_overlap_creates_a_cannot_merge_constraint(self):
        timeline = [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
            {"start": 2.5, "end": 4.0, "speaker": "SPEAKER_01"},
            {"start": 5.0, "end": 6.0, "speaker": "SPEAKER_02"},
        ]
        self.assertEqual(
            overlapping_speaker_pairs(timeline),
            {frozenset(("SPEAKER_00", "SPEAKER_01"))},
        )

    def test_identity_evidence_counts_recognized_speech_only(self):
        result = {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "words": [
                        {
                            "word": "Hello",
                            "start": 1.0,
                            "end": 1.4,
                            "speaker": "SPEAKER_00",
                        },
                        {
                            "word": "[noise]",
                            "start": 2.0,
                            "end": 2.3,
                            "speaker": "SPEAKER_00",
                        },
                    ],
                }
            ]
        }
        intervals, count, timing_available = _recognized_speech_intervals(
            result, "SPEAKER_00"
        )
        self.assertTrue(timing_available)
        self.assertEqual(count, 1)
        self.assertEqual(intervals, [(0.94, 1.46)])


class SpeakerRefinementTests(unittest.TestCase):
    @staticmethod
    def _result(overlap=False):
        timeline = [
            {"start": 0.0, "end": 3.2, "speaker": "SPEAKER_00"},
            {"start": 3.8, "end": 4.8, "speaker": "SPEAKER_01"},
            {"start": 5.0, "end": 8.0, "speaker": "SPEAKER_00"},
            {"start": 8.2, "end": 12.0, "speaker": "SPEAKER_01"},
        ]
        if overlap:
            timeline.append(
                {"start": 3.9, "end": 4.6, "speaker": "SPEAKER_00"}
            )
        return {
            "segments": [
                {
                    "start": 0.0,
                    "end": 3.0,
                    "text": "I cannot stand it anymore. Look",
                    "speaker": "SPEAKER_00",
                    "words": [
                        {
                            "word": "I cannot stand it anymore. Look",
                            "start": 0.0,
                            "end": 3.0,
                            "speaker": "SPEAKER_00",
                        }
                    ],
                },
                {
                    "start": 3.9,
                    "end": 4.7,
                    "text": "at the way you're talking to me.",
                    "speaker": "SPEAKER_01",
                    "words": [
                        {
                            "word": "at the way you're talking to me.",
                            "start": 3.9,
                            "end": 4.7,
                            "speaker": "SPEAKER_01",
                        }
                    ],
                },
                {
                    "start": 5.0,
                    "end": 8.0,
                    "text": "This is still the first speaker.",
                    "speaker": "SPEAKER_00",
                    "words": [
                        {
                            "word": "This is still the first speaker.",
                            "start": 5.0,
                            "end": 8.0,
                            "speaker": "SPEAKER_00",
                        }
                    ],
                },
            ],
            "speaker_timeline": timeline,
            "speaker_assignment_timeline": [dict(item) for item in timeline[:4]],
        }

    @staticmethod
    def _encoder(candidate_embedding):
        encoder = SpeakerIdentityEncoder.__new__(SpeakerIdentityEncoder)
        encoder.device = "cpu"
        encoder.extract = mock.Mock(
            return_value={
                "SPEAKER_00": {
                    "embedding": [1.0, 0.0],
                    "clean_seconds": 20.0,
                    "window_count": 5,
                    "cohesion": 0.9,
                },
                "SPEAKER_01": {
                    "embedding": [0.0, 1.0],
                    "clean_seconds": 20.0,
                    "window_count": 5,
                    "cohesion": 0.9,
                },
            }
        )
        encoder._encode_interval = mock.Mock(return_value=candidate_embedding)
        return encoder

    @staticmethod
    def _micro_result(middle_text="have been the"):
        return {
            "language": "en",
            "segments": [
                {
                    "start": 97.8,
                    "end": 100.75,
                    "text": "I understand. What would",
                    "speaker": "SPEAKER_00",
                    "words": [
                        {
                            "word": "I understand. What would",
                            "start": 97.8,
                            "end": 100.75,
                            "speaker": "SPEAKER_00",
                        }
                    ],
                },
                {
                    "start": 100.798,
                    "end": 101.158,
                    "text": middle_text,
                    "speaker": "SPEAKER_01",
                    "words": [
                        {
                            "word": middle_text,
                            "start": 100.798,
                            "end": 101.158,
                            "speaker": "SPEAKER_01",
                        }
                    ],
                },
                {
                    "start": 101.19,
                    "end": 104.2,
                    "text": "next step?",
                    "speaker": "SPEAKER_00",
                    "words": [
                        {
                            "word": "next step?",
                            "start": 101.19,
                            "end": 104.2,
                            "speaker": "SPEAKER_00",
                        }
                    ],
                },
            ],
            "speaker_timeline": [
                {"start": 97.8, "end": 100.75, "speaker": "SPEAKER_00"},
                {"start": 100.798, "end": 101.158, "speaker": "SPEAKER_01"},
                {"start": 101.19, "end": 104.2, "speaker": "SPEAKER_00"},
            ],
            "speaker_assignment_timeline": [
                {"start": 97.8, "end": 100.75, "speaker": "SPEAKER_00"},
                {"start": 100.798, "end": 101.158, "speaker": "SPEAKER_01"},
                {"start": 101.19, "end": 104.2, "speaker": "SPEAKER_00"},
            ],
        }

    @staticmethod
    def _micro_encoder():
        encoder = SpeakerIdentityEncoder.__new__(SpeakerIdentityEncoder)
        encoder.device = "cpu"
        encoder.extract = mock.Mock(
            return_value={
                "SPEAKER_00": {
                    "embedding": [1.0, 0.0, 0.0],
                    "clean_seconds": 20.0,
                    "window_count": 5,
                    "cohesion": 0.9,
                },
                "SPEAKER_01": {
                    "embedding": [0.0, 1.0, 0.0],
                    "clean_seconds": 20.0,
                    "window_count": 5,
                    "cohesion": 0.9,
                },
            }
        )
        encoder._encode_interval = mock.Mock(
            return_value=[0.47, 0.17, 0.865910]
        )
        return encoder

    @staticmethod
    def _sentence_seam_result(
        previous_text="I am",
        continuation_text="still being criticized. And pointing fingers.",
        continuation_end=26.5,
    ):
        return {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 20.0,
                    "text": previous_text,
                    "speaker": "SPEAKER_00",
                    "words": [
                        {
                            "word": previous_text,
                            "start": 0.0,
                            "end": 20.0,
                            "speaker": "SPEAKER_00",
                        }
                    ],
                },
                {
                    "start": 20.02,
                    "end": continuation_end,
                    "text": continuation_text,
                    "speaker": "SPEAKER_01",
                    "words": [
                        {
                            "word": continuation_text,
                            "start": 20.02,
                            "end": continuation_end,
                            "speaker": "SPEAKER_01",
                        }
                    ],
                },
            ],
            "speaker_timeline": [
                {"start": 0.0, "end": 20.0, "speaker": "SPEAKER_00"},
                {
                    "start": 20.02,
                    "end": continuation_end,
                    "speaker": "SPEAKER_01",
                },
            ],
            "speaker_assignment_timeline": [
                {"start": 0.0, "end": 20.0, "speaker": "SPEAKER_00"},
                {
                    "start": 20.02,
                    "end": continuation_end,
                    "speaker": "SPEAKER_01",
                },
            ],
        }

    @staticmethod
    def _one_sided_micro_result():
        return {
            "language": "en",
            "segments": [
                {
                    "start": 49.174,
                    "end": 50.395,
                    "text": "Do you think she's listening",
                    "speaker": "SPEAKER_00",
                    "words": [
                        {
                            "word": "Do you think she's listening",
                            "start": 49.174,
                            "end": 50.395,
                            "speaker": "SPEAKER_00",
                        }
                    ],
                },
                {
                    "start": 50.435,
                    "end": 50.816,
                    "text": "to me?",
                    "speaker": "SPEAKER_01",
                    "words": [
                        {
                            "word": "to me?",
                            "start": 50.435,
                            "end": 50.816,
                            "speaker": "SPEAKER_01",
                        }
                    ],
                },
            ],
            "speaker_timeline": [
                {"start": 49.174, "end": 50.395, "speaker": "SPEAKER_00"},
                {"start": 50.435, "end": 50.816, "speaker": "SPEAKER_01"},
            ],
            "speaker_assignment_timeline": [
                {"start": 49.174, "end": 50.395, "speaker": "SPEAKER_00"},
                {"start": 50.435, "end": 50.816, "speaker": "SPEAKER_01"},
            ],
        }

    def test_short_acoustic_misassignment_is_corrected_and_audited(self):
        result = self._result()
        report = self._encoder([0.99, 0.01]).refine([], result)
        self.assertEqual(report["corrections_applied"], 1)
        correction = report["corrections"][0]
        self.assertEqual(correction["from_local_speaker"], "SPEAKER_01")
        self.assertEqual(correction["to_local_speaker"], "SPEAKER_00")
        corrected_segment = result["segments"][1]
        self.assertEqual(corrected_segment["speaker"], "SPEAKER_00")
        self.assertEqual(
            corrected_segment["diarization_speaker"], "SPEAKER_01"
        )
        corrected_word = corrected_segment["words"][0]
        self.assertEqual(corrected_word["speaker"], "SPEAKER_00")
        self.assertEqual(corrected_word["diarization_speaker"], "SPEAKER_01")
        self.assertIn("speaker_assignment_timeline_original", result)
        refined_at_four_seconds = [
            item
            for item in result["speaker_assignment_timeline"]
            if item["start"] <= 4.0 < item["end"]
        ]
        self.assertEqual(refined_at_four_seconds[0]["speaker"], "SPEAKER_00")
        normalized = normalize_segments(result)
        corrected = next(item for item in normalized if item["start"] == 3.9)
        self.assertEqual(corrected["diarization_speaker"], "SPEAKER_01")
        self.assertEqual(
            corrected["words"][0]["speaker_refinement"]["to_local_speaker"],
            "SPEAKER_00",
        )

    def test_ambiguous_short_run_is_not_forced_to_neighbor(self):
        result = self._result()
        report = self._encoder([0.71, 0.70]).refine([], result)
        self.assertEqual(report["corrections_applied"], 0)
        self.assertEqual(
            report["evaluated_candidates"][0]["decision"],
            "abstained_below_confidence_threshold",
        )
        self.assertEqual(result["segments"][1]["speaker"], "SPEAKER_01")
        self.assertNotIn("speaker_assignment_timeline_original", result)

    def test_overlapping_short_run_is_never_smoothed(self):
        result = self._result(overlap=True)
        encoder = self._encoder([0.99, 0.01])
        report = encoder.refine([], result)
        self.assertEqual(report["corrections_applied"], 0)
        self.assertEqual(report["evaluated_candidates"], [])
        self.assertEqual(result["segments"][1]["speaker"], "SPEAKER_01")
        encoder._encode_interval.assert_not_called()

    def test_neighbor_is_not_selected_when_a_third_voice_matches_best(self):
        result = self._result()
        encoder = self._encoder([0.0, 0.0, 1.0])
        encoder.extract.return_value = {
            "SPEAKER_00": {
                "embedding": [1.0, 0.0, 0.0],
                "clean_seconds": 20.0,
                "window_count": 5,
                "cohesion": 0.9,
            },
            "SPEAKER_01": {
                "embedding": [0.0, 1.0, 0.0],
                "clean_seconds": 20.0,
                "window_count": 5,
                "cohesion": 0.9,
            },
            "SPEAKER_02": {
                "embedding": [0.0, 0.0, 1.0],
                "clean_seconds": 20.0,
                "window_count": 5,
                "cohesion": 0.9,
            },
        }
        report = encoder.refine([], result)
        self.assertEqual(report["corrections_applied"], 0)
        self.assertEqual(
            report["evaluated_candidates"][0]["decision"],
            "abstained_best_voice_not_supported_by_context",
        )
        self.assertEqual(result["segments"][1]["speaker"], "SPEAKER_01")

    def test_micro_label_island_inside_english_sentence_is_corrected(self):
        result = self._micro_result()
        report = self._micro_encoder().refine([], result)
        self.assertEqual(report["corrections_applied"], 1)
        correction = report["corrections"][0]
        self.assertTrue(correction["sandwiched_sentence_continuation"])
        self.assertEqual(correction["to_local_speaker"], "SPEAKER_00")
        self.assertEqual(result["segments"][1]["speaker"], "SPEAKER_00")

    def test_punctuated_micro_interjection_is_not_smoothed(self):
        result = self._micro_result("No!")
        report = self._micro_encoder().refine([], result)
        self.assertEqual(report["corrections_applied"], 0)
        candidate = report["evaluated_candidates"][0]
        self.assertFalse(candidate["sandwiched_sentence_continuation"])
        self.assertEqual(
            candidate["decision"], "abstained_below_confidence_threshold"
        )
        self.assertEqual(result["segments"][1]["speaker"], "SPEAKER_01")

    def test_long_continuation_at_sentence_seam_is_corrected(self):
        result = self._sentence_seam_result()
        report = self._encoder([0.99, 0.01]).refine([], result)
        self.assertEqual(report["corrections_applied"], 1)
        correction = report["corrections"][0]
        self.assertGreater(correction["duration_seconds"], 6.0)
        self.assertTrue(correction["sentence_seam_continuation"])
        self.assertEqual(correction["to_local_speaker"], "SPEAKER_00")
        self.assertEqual(result["segments"][1]["speaker"], "SPEAKER_00")
        turns = build_turns(normalize_segments(result))
        self.assertEqual(len(turns), 1)
        self.assertIn("I am still being criticized", turns[0]["text"])

    def test_one_sided_micro_sentence_ending_is_corrected(self):
        result = self._one_sided_micro_result()
        report = self._micro_encoder().refine([], result)
        self.assertEqual(report["corrections_applied"], 1)
        correction = report["corrections"][0]
        self.assertTrue(correction["sentence_seam_continuation"])
        self.assertFalse(correction["sandwiched_sentence_continuation"])
        self.assertEqual(result["segments"][1]["speaker"], "SPEAKER_00")
        turns = build_turns(normalize_segments(result))
        self.assertEqual(len(turns), 1)
        self.assertIn("listening to me?", turns[0]["text"])

    def test_long_new_turn_after_completed_sentence_is_not_reclassified(self):
        result = self._sentence_seam_result(
            previous_text="I finished my point.",
            continuation_text="Now I would like to answer that.",
        )
        encoder = self._encoder([0.99, 0.01])
        report = encoder.refine([], result)
        self.assertEqual(report["corrections_applied"], 0)
        self.assertEqual(report["evaluated_candidates"], [])
        self.assertEqual(result["segments"][1]["speaker"], "SPEAKER_01")
        encoder._encode_interval.assert_not_called()


class ConfigurationTests(unittest.TestCase):
    def test_exact_speaker_count_sets_both_diarization_bounds(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--speakers", "3"])
        cli._validate_args(parser, args)
        self.assertEqual(args.min_speakers, 3)
        self.assertEqual(args.max_speakers, 3)

    def test_speaker_refinement_is_default_with_explicit_opt_out(self):
        parser = cli.build_parser()
        enabled = parser.parse_args([])
        disabled = parser.parse_args(["--no-speaker-refinement"])
        self.assertTrue(enabled.speaker_refinement)
        self.assertFalse(disabled.speaker_refinement)

    def test_processing_identity_includes_program_version(self):
        settings = cli.processing_settings(
            cli.build_parser().parse_args([]), "off", "off"
        )
        runtime = RuntimeSettings(
            device="cpu",
            compute_type="int8",
            batch_size=1,
            threads=1,
            device_was_auto=False,
            description="test",
            analysis_device="cpu",
        )
        identity = cli.settings_identity(settings, runtime)
        self.assertEqual(identity["program_version"], cli.__version__)

    def test_token_round_trip_and_private_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.env"
            cli.save_hf_token("hf_value with-special", path)
            self.assertEqual(cli.load_saved_hf_token(path), "hf_value with-special")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_project_defaults_match_product_directories(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.dict(os.environ, {"TRANSCRIBE_MEDIA_ROOT": directory}),
        ):
            paths = cli.resolve_paths(cli.build_parser().parse_args([]))
            self.assertEqual(paths.source_dir, Path(directory) / "Video Source")
            self.assertEqual(paths.transcript_dir, Path(directory) / "Transcribed")
            self.assertEqual(paths.review_dir, Path(directory) / "Review")

    def test_whisperx_explicitly_selects_visible_cuda_device_zero(self):
        backend = WhisperXBackend.__new__(WhisperXBackend)
        backend.whisperx = mock.Mock()
        backend.model_name = "large-v3"
        backend.language = "en"
        backend.task = "transcribe"
        backend.hf_token = None
        runtime = RuntimeSettings("cuda", "float16", 8, 4, True, "test")
        backend._load_model(runtime)
        self.assertEqual(
            backend.whisperx.load_model.call_args.kwargs["device_index"], 0
        )

    def test_known_ctranslate2_cuda_failure_has_repair_instructions(self):
        backend = WhisperXBackend.__new__(WhisperXBackend)
        backend.runtime = RuntimeSettings("cuda", "float16", 8, 4, True, "test")
        backend.model = mock.Mock()
        backend.model.transcribe.side_effect = RuntimeError(
            "parallel_for failed: cudaErrorInvalidDevice: invalid device ordinal"
        )
        backend._decode_audio = mock.Mock(return_value=[])
        with (
            mock.patch(
                "transcribe_media_app.backends.package_version",
                return_value="4.8.1",
            ),
            self.assertRaisesRegex(RuntimeError, r"4\.8\.1.*Run ./install\.sh"),
        ):
            backend.transcribe(Path("clip.mp4"), align=False, verbose=False)

    def test_doctor_rejects_unpinned_ctranslate2_on_cuda(self):
        fake_torch = mock.Mock()
        fake_torch.cuda.is_available.return_value = True
        with (
            mock.patch.object(cli, "package_version", return_value="4.8.1"),
            mock.patch.dict("sys.modules", {"torch": fake_torch}),
            self.assertRaisesRegex(RuntimeError, r"4\.7\.2.*Run ./install\.sh"),
        ):
            cli._ctranslate2_check()

    def test_ten_gib_gpu_keeps_asr_on_cuda_and_analysis_on_cpu(self):
        fake_torch = mock.Mock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.get_device_properties.return_value.total_memory = 10 * 1024**3
        args = cli.build_parser().parse_args([])
        with mock.patch.dict("sys.modules", {"torch": fake_torch}):
            runtime = cli.resolve_runtime(args)
        self.assertEqual(runtime.device, "cuda")
        self.assertEqual(runtime.analysis_device, "cpu")
        self.assertEqual(runtime.diarization_batch_size, 4)
        self.assertIn("10.0 GiB GPU safeguard", runtime.description)

    def test_speaker_validation_rejects_silent_diarization_failure(self):
        result = {
            "segments": [{"start": 0, "end": 1, "text": "hello"}],
            "speaker_timeline": [],
        }
        with self.assertRaisesRegex(RuntimeError, "no usable speaker"):
            cli._validate_speaker_result(result, None, None)

    def test_pyannote_retries_its_own_batch_size(self):
        class FakeModel:
            segmentation_batch_size = 32
            embedding_batch_size = 32

        class FakeFrame:
            @staticmethod
            def iterrows():
                yield 0, {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}

        class FakePipeline:
            def __init__(self):
                self.model = FakeModel()
                self.calls = []

            def __call__(self, audio, **kwargs):
                del audio, kwargs
                self.calls.append(self.model.segmentation_batch_size)
                if self.model.segmentation_batch_size > 1:
                    raise MemoryError("batch_size is probably too large")
                return FakeFrame()

        fake_whisperx = mock.Mock()

        def assign_speakers(diarization, result):
            del diarization
            result["segments"][0]["speaker"] = "SPEAKER_00"
            return result

        fake_whisperx.assign_word_speakers.side_effect = assign_speakers
        fake_torch = mock.Mock()
        fake_torch.cuda.is_available.return_value = False
        diarizer = PyannoteDiarizer.__new__(PyannoteDiarizer)
        diarizer.device = "cpu"
        diarizer.batch_size = 4
        diarizer.last_run = {}
        diarizer.pipeline = FakePipeline()
        result = {"segments": [{"start": 0, "end": 1, "text": "hello"}]}
        with mock.patch.dict(
            "sys.modules", {"whisperx": fake_whisperx, "torch": fake_torch}
        ):
            assigned = diarizer.assign(Path("clip.wav"), [], result, None, None)
        self.assertEqual(diarizer.pipeline.calls, [4, 2, 1])
        self.assertEqual(diarizer.last_run["batch_size_used"], 1)
        self.assertEqual(assigned["segments"][0]["speaker"], "SPEAKER_00")

    def test_pyannote_uses_exact_count_and_exclusive_word_assignment(self):
        class Segment:
            def __init__(self, start, end):
                self.start = start
                self.end = end

        class Annotation:
            def __init__(self, rows):
                self.rows = rows

            def itertracks(self, yield_label=False):
                self.assert_yield_label = yield_label
                yield from self.rows

        class FakeModel:
            segmentation_batch_size = 1
            embedding_batch_size = 1

            def __init__(self):
                self.options = None

            def __call__(self, audio, **options):
                del audio
                self.options = options
                return type(
                    "Output",
                    (),
                    {
                        "speaker_diarization": Annotation(
                            [
                                (Segment(0.0, 2.0), None, "SPEAKER_00"),
                                (Segment(1.0, 3.0), None, "SPEAKER_01"),
                            ]
                        ),
                        "exclusive_speaker_diarization": Annotation(
                            [
                                (Segment(0.0, 1.5), None, "SPEAKER_00"),
                                (Segment(1.5, 3.0), None, "SPEAKER_01"),
                            ]
                        ),
                    },
                )()

        fake_model = FakeModel()
        diarizer = PyannoteDiarizer.__new__(PyannoteDiarizer)
        diarizer.device = "cpu"
        diarizer.batch_size = 1
        diarizer.last_run = {}
        diarizer.pipeline = type("Pipeline", (), {"model": fake_model})()
        fake_whisperx = mock.Mock()

        def assign_speakers(diarization, result):
            self.assertEqual(diarization["start"].tolist(), [0.0, 1.5])
            result["segments"][0]["speaker"] = "SPEAKER_00"
            return result

        fake_whisperx.assign_word_speakers.side_effect = assign_speakers
        result = {"segments": [{"start": 0, "end": 1, "text": "hello"}]}
        with mock.patch.dict("sys.modules", {"whisperx": fake_whisperx}):
            assigned = diarizer.assign(Path("clip.wav"), [0.0] * 16000, result, 2, 2)
        self.assertEqual(fake_model.options, {"num_speakers": 2})
        self.assertEqual(len(assigned["speaker_timeline"]), 2)
        self.assertEqual(assigned["speaker_assignment_timeline"][1]["start"], 1.5)
        self.assertEqual(diarizer.last_run["word_assignment_timeline"], "exclusive")


class FakeBackend:
    name = "fake-asr"

    def __init__(
        self,
        failure_name=None,
        interrupt_name=None,
        mutate=False,
        degraded=None,
    ):
        self.failure_name = failure_name
        self.interrupt_name = interrupt_name
        self.mutate = mutate
        self.degraded = degraded
        self.calls = []

    def transcribe(self, source, align, verbose):
        del align, verbose
        self.calls.append(source.name)
        if source.name == self.interrupt_name:
            raise KeyboardInterrupt()
        if source.name == self.failure_name:
            raise RuntimeError("deliberate corrupt media error")
        if self.mutate:
            source.write_bytes(source.read_bytes() + b"changed")
        return (
            {
                "language": "en",
                "segments": [
                    {
                        "start": 0,
                        "end": 1,
                        "text": "A faithful sample.",
                        "words": [{"word": "A", "start": 0, "end": 0.2, "score": 0.9}],
                    }
                ],
            },
            [],
            {"transcription_engine": "fake", "transcription_model": "test"},
            [self.degraded] if self.degraded else [],
        )


class FlowTests(unittest.TestCase):
    def _args(self, root, *extra):
        return [
            str(root / "Video Source"),
            "--transcript-dir",
            str(root / "Transcribed"),
            "--review-dir",
            str(root / "Review"),
            "--device",
            "cpu",
            "--no-diarize",
            "--no-acoustic",
            "--no-tone",
            *extra,
        ]

    @staticmethod
    def _runtime():
        return RuntimeSettings("cpu", "int8", 1, 1, False, "test")

    def test_dry_run_never_loads_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Video Source").mkdir()
            (root / "Video Source" / "clip.mp4").write_bytes(b"media")
            with mock.patch.object(
                cli.WhisperXBackend, "create", side_effect=AssertionError("loaded")
            ):
                self.assertEqual(cli.main(self._args(root, "--dry-run")), 0)

    def test_success_then_identical_rerun_skips_without_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Video Source").mkdir()
            (root / "Video Source" / "clip.mp4").write_bytes(b"media")
            with (
                mock.patch.object(cli, "resolve_runtime", return_value=self._runtime()),
                mock.patch.object(
                    cli.WhisperXBackend,
                    "create",
                    return_value=(FakeBackend(), self._runtime()),
                ),
                mock.patch.object(cli, "ffprobe_duration", return_value=1.0),
            ):
                self.assertEqual(cli.main(self._args(root)), 0)
            with (
                mock.patch.object(cli, "resolve_runtime", return_value=self._runtime()),
                mock.patch.object(
                    cli.WhisperXBackend, "create", side_effect=AssertionError("loaded")
                ),
            ):
                self.assertEqual(cli.main(self._args(root)), 0)

            payload = json.loads((root / "Review" / "clip.mp4.json").read_text())
            self.assertEqual(payload["segments"][0]["words"][0]["confidence"], 0.9)
            self.assertTrue(payload["processing"]["local_processing"])

    def test_corrupt_file_fails_but_next_file_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Video Source"
            source.mkdir()
            (source / "a-corrupt.mp4").write_bytes(b"broken")
            (source / "b-good.mp4").write_bytes(b"media")
            backend = FakeBackend(failure_name="a-corrupt.mp4")
            with (
                mock.patch.object(cli, "resolve_runtime", return_value=self._runtime()),
                mock.patch.object(
                    cli.WhisperXBackend,
                    "create",
                    return_value=(backend, self._runtime()),
                ),
                mock.patch.object(cli, "ffprobe_duration", return_value=1.0),
            ):
                self.assertEqual(cli.main(self._args(root)), 1)
            manifest = json.loads(
                (root / "Review" / "transcription_manifest.json").read_text()
            )
            self.assertEqual(manifest["sources"]["a-corrupt.mp4"]["status"], "failed")
            self.assertEqual(manifest["sources"]["b-good.mp4"]["status"], "complete")
            self.assertTrue((root / "Transcribed" / "b-good.mp4.txt").is_file())

    def test_adding_a_file_processes_only_the_new_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Video Source"
            source.mkdir()
            (source / "first.mp4").write_bytes(b"first")
            backend = FakeBackend()
            patches = (
                mock.patch.object(cli, "resolve_runtime", return_value=self._runtime()),
                mock.patch.object(
                    cli.WhisperXBackend,
                    "create",
                    return_value=(backend, self._runtime()),
                ),
                mock.patch.object(cli, "ffprobe_duration", return_value=1.0),
            )
            with patches[0], patches[1], patches[2]:
                self.assertEqual(cli.main(self._args(root)), 0)
            (source / "second.mp4").write_bytes(b"second")
            backend.calls.clear()
            with (
                mock.patch.object(cli, "resolve_runtime", return_value=self._runtime()),
                mock.patch.object(
                    cli.WhisperXBackend,
                    "create",
                    return_value=(backend, self._runtime()),
                ),
                mock.patch.object(cli, "ffprobe_duration", return_value=1.0),
            ):
                self.assertEqual(cli.main(self._args(root)), 0)
            self.assertEqual(backend.calls, ["second.mp4"])

    def test_interruption_is_recorded_for_safe_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Video Source"
            source.mkdir()
            (source / "clip.mp4").write_bytes(b"media")
            backend = FakeBackend(interrupt_name="clip.mp4")
            with (
                mock.patch.object(cli, "resolve_runtime", return_value=self._runtime()),
                mock.patch.object(
                    cli.WhisperXBackend,
                    "create",
                    return_value=(backend, self._runtime()),
                ),
            ):
                self.assertEqual(cli.main(self._args(root)), 130)
            manifest = json.loads(
                (root / "Review" / "transcription_manifest.json").read_text()
            )
            self.assertEqual(manifest["sources"]["clip.mp4"]["status"], "interrupted")

    def test_source_change_during_processing_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Video Source"
            source.mkdir()
            (source / "clip.mp4").write_bytes(b"media")
            backend = FakeBackend(mutate=True)
            with (
                mock.patch.object(cli, "resolve_runtime", return_value=self._runtime()),
                mock.patch.object(
                    cli.WhisperXBackend,
                    "create",
                    return_value=(backend, self._runtime()),
                ),
            ):
                self.assertEqual(cli.main(self._args(root)), 1)
            self.assertFalse((root / "Transcribed" / "clip.mp4.txt").exists())

    def test_unexpected_stage_degradation_is_not_marked_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Video Source"
            source.mkdir()
            (source / "clip.mp4").write_bytes(b"media")
            backend = FakeBackend(degraded="alignment: deliberate test failure")
            with (
                mock.patch.object(cli, "resolve_runtime", return_value=self._runtime()),
                mock.patch.object(
                    cli.WhisperXBackend,
                    "create",
                    return_value=(backend, self._runtime()),
                ),
                mock.patch.object(cli, "ffprobe_duration", return_value=1.0),
            ):
                self.assertEqual(cli.main(self._args(root)), 1)
            manifest = json.loads(
                (root / "Review" / "transcription_manifest.json").read_text()
            )
            state = manifest["sources"]["clip.mp4"]
            self.assertEqual(state["status"], "degraded")
            self.assertFalse(state["completion_state"])
            self.assertTrue(state["retry_recommended"])

    def test_diarization_memory_failure_cannot_publish_unknown_speakers(self):
        class FailingDiarizer:
            name = "test-diarizer"
            model_name = "test-model"

            @staticmethod
            def assign(*args, **kwargs):
                del args, kwargs
                raise MemoryError("batch_size 32 is probably too large")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Video Source"
            source.mkdir()
            (source / "clip.mp4").write_bytes(b"media")
            args = [
                str(source),
                "--transcript-dir",
                str(root / "Transcribed"),
                "--review-dir",
                str(root / "Review"),
                "--device",
                "cpu",
                "--diarization-backend",
                "pyannote",
                "--hf-token",
                "test-token",
                "--no-acoustic",
                "--no-tone",
            ]
            with (
                mock.patch.object(cli, "resolve_runtime", return_value=self._runtime()),
                mock.patch.object(
                    cli.WhisperXBackend,
                    "create",
                    return_value=(FakeBackend(), self._runtime()),
                ),
                mock.patch.object(
                    cli, "create_diarizer", return_value=FailingDiarizer()
                ),
            ):
                self.assertEqual(cli.main(args), 1)
            self.assertFalse((root / "Transcribed" / "clip.mp4.txt").exists())
            manifest = json.loads(
                (root / "Review" / "transcription_manifest.json").read_text()
            )
            state = manifest["sources"]["clip.mp4"]
            self.assertEqual(state["status"], "failed")
            self.assertIn("required speaker diarization failed", state["error"])

    def test_default_speaker_refinement_is_written_to_canonical_output(self):
        class WorkingDiarizer:
            name = "test-diarizer"
            model_name = "test-model"
            last_run = {"device": "cpu"}

            @staticmethod
            def assign(source, audio, result, min_speakers, max_speakers):
                del source, audio, min_speakers, max_speakers
                result["segments"][0]["speaker"] = "SPEAKER_00"
                result["segments"][0]["words"][0]["speaker"] = "SPEAKER_00"
                result["speaker_timeline"] = [
                    {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}
                ]
                result["speaker_assignment_timeline"] = [
                    {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}
                ]
                return result

        class WorkingRefiner:
            name = "test-embedding-model"
            version = "test-revision"

            @staticmethod
            def refine(audio, result):
                del audio
                report = {
                    "enabled": True,
                    "algorithm": "test-refinement",
                    "algorithm_version": "test",
                    "prototype_count": 1,
                    "word_runs": 1,
                    "candidates_examined": 0,
                    "corrections_applied": 0,
                    "corrections": [],
                    "raw_diarization_preserved": True,
                }
                result["speaker_refinement"] = report
                return report

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Video Source"
            source.mkdir()
            (source / "clip.mp4").write_bytes(b"media")
            args = [
                str(source),
                "--transcript-dir",
                str(root / "Transcribed"),
                "--review-dir",
                str(root / "Review"),
                "--device",
                "cpu",
                "--diarization-backend",
                "pyannote",
                "--hf-token",
                "test-token",
                "--no-speaker-identity",
                "--no-acoustic",
                "--no-tone",
            ]
            with (
                mock.patch.object(cli, "resolve_runtime", return_value=self._runtime()),
                mock.patch.object(
                    cli.WhisperXBackend,
                    "create",
                    return_value=(FakeBackend(), self._runtime()),
                ),
                mock.patch.object(
                    cli, "create_diarizer", return_value=WorkingDiarizer()
                ),
                mock.patch.object(
                    cli, "SpeakerIdentityEncoder", return_value=WorkingRefiner()
                ),
                mock.patch.object(cli, "ffprobe_duration", return_value=1.0),
            ):
                self.assertEqual(cli.main(args), 0)
            payload = json.loads(
                (root / "Review" / "clip.mp4.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["speaker_refinement"]["algorithm"], "test-refinement"
            )
            self.assertEqual(
                payload["processing"]["provenance"]["speaker_refinement"][
                    "corrections_applied"
                ],
                0,
            )

    def test_missing_initialized_voice_registry_stops_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Video Source"
            review = root / "Review"
            source.mkdir()
            review.mkdir()
            (source / "clip.mp4").write_bytes(b"media")
            manifest = ManifestStore(review / "transcription_manifest.json")
            manifest.data["sources"]["previous.mp4"] = {
                "provenance": {
                    "speaker_identity": {
                        "profile_count": 2,
                        "registry_revision": 4,
                    }
                }
            }
            manifest.save()
            args = [
                str(source),
                "--transcript-dir",
                str(root / "Transcribed"),
                "--review-dir",
                str(review),
                "--device",
                "cpu",
                "--diarization-backend",
                "speechbrain",
                "--no-acoustic",
                "--no-tone",
            ]
            with mock.patch.object(
                cli.WhisperXBackend,
                "create",
                side_effect=AssertionError("model must not load"),
            ):
                self.assertEqual(cli.main(args), 1)

    def test_orphaned_voice_registry_stops_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Video Source"
            review = root / "Review"
            source.mkdir()
            review.mkdir()
            (source / "clip.mp4").write_bytes(b"media")
            registry = VoiceRegistry(review / "speaker_registry.json")
            registry.identify(
                source_key="old.mp4",
                source_fingerprint="old",
                local_speakers=("SPEAKER_00",),
                evidence={
                    "SPEAKER_00": {
                        "embedding": [1.0, 0.0, 0.0],
                        "clean_seconds": 20.0,
                        "window_count": 5,
                        "cohesion": 0.8,
                    }
                },
            )
            args = [
                str(source),
                "--transcript-dir",
                str(root / "Transcribed"),
                "--review-dir",
                str(review),
                "--device",
                "cpu",
                "--diarization-backend",
                "speechbrain",
                "--no-acoustic",
                "--no-tone",
            ]
            with mock.patch.object(
                cli.WhisperXBackend,
                "create",
                side_effect=AssertionError("model must not load"),
            ):
                self.assertEqual(cli.main(args), 1)

    def test_reset_speaker_registry_archives_state_recoverably(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Video Source"
            review = root / "Review"
            source.mkdir()
            review.mkdir()
            registry = review / "speaker_registry.json"
            manifest = review / "transcription_manifest.json"
            registry.write_text("private registry", encoding="utf-8")
            manifest.write_text("durable manifest", encoding="utf-8")
            result = cli.main(
                [
                    str(source),
                    "--transcript-dir",
                    str(root / "Transcribed"),
                    "--review-dir",
                    str(review),
                    "--reset-speaker-registry",
                ]
            )
            self.assertEqual(result, 0)
            self.assertFalse(registry.exists())
            self.assertFalse(manifest.exists())
            backups = list((review / "speaker-registry-backups").iterdir())
            self.assertEqual(len(backups), 1)
            self.assertEqual(
                (backups[0] / registry.name).read_text(encoding="utf-8"),
                "private registry",
            )
            self.assertEqual(
                (backups[0] / manifest.name).read_text(encoding="utf-8"),
                "durable manifest",
            )
            self.assertEqual(backups[0].stat().st_mode & 0o777, 0o700)

    def test_changed_voice_registry_stops_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Video Source"
            review = root / "Review"
            source.mkdir()
            review.mkdir()
            (source / "clip.mp4").write_bytes(b"media")
            registry = VoiceRegistry(review / "speaker_registry.json")
            report = registry.identify(
                source_key="previous.mp4",
                source_fingerprint="previous",
                local_speakers=("SPEAKER_00",),
                evidence={
                    "SPEAKER_00": {
                        "embedding": [1.0, 0.0, 0.0],
                        "clean_seconds": 20.0,
                        "window_count": 5,
                        "cohesion": 0.8,
                    }
                },
            )
            manifest = ManifestStore(review / "transcription_manifest.json")
            manifest.data["speaker_registry"] = {
                "profile_count": 1,
                "revision": report["registry_revision"],
                "state_hash": "different-state",
            }
            manifest.save()
            args = [
                str(source),
                "--transcript-dir",
                str(root / "Transcribed"),
                "--review-dir",
                str(review),
                "--device",
                "cpu",
                "--diarization-backend",
                "speechbrain",
                "--no-acoustic",
                "--no-tone",
            ]
            with mock.patch.object(
                cli.WhisperXBackend,
                "create",
                side_effect=AssertionError("model must not load"),
            ):
                self.assertEqual(cli.main(args), 1)


class DocumentationConsistencyTests(unittest.TestCase):
    def test_documented_commands_use_real_flags(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        documented = set(
            re.findall(r"transcribe-media [^\n`]*?(--[a-z][a-z-]*)", readme)
        )
        implemented = {
            option
            for action in cli.build_parser()._actions
            for option in action.option_strings
        }
        self.assertFalse(documented - implemented)

    def test_installer_help_is_valid(self):
        result = subprocess.run(
            ["bash", "install.sh", "--help"], check=True, capture_output=True, text=True
        )
        self.assertIn("--cpu", result.stdout)
        self.assertIn("--skip-model-downloads", result.stdout)

    def test_cuda_regression_workaround_is_pinned(self):
        requirements = Path("requirements.txt").read_text(encoding="utf-8")
        self.assertIn(
            f"ctranslate2=={SUPPORTED_CTRANSLATE2_VERSION}", requirements.splitlines()
        )
        self.assertIn("whisperx==3.8.6", requirements.splitlines())
        self.assertIn("funasr==1.4.2", requirements.splitlines())


if __name__ == "__main__":
    unittest.main()
