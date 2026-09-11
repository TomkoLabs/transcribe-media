import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import numpy as np

from transcribe_media_app import cli
from transcribe_media_app.analysis import annotate_raw_overlap, annotate_speaker_attribution, build_turns, normalize_segments
from transcribe_media_app.evidence import extract_reference_evidence
from transcribe_media_app.evaluation import evaluate_registry
from transcribe_media_app.backends import WhisperXBackend, _map_secondary_timeline
from transcribe_media_app.review import (save_review, apply_review, merge_project_profiles,
                                         restore_reviewed_choices, refresh_reviews, packet_digest)
from transcribe_media_app.schema import RESULT_SCHEMA_VERSION, SAMPLE_RATE, RuntimeSettings
from transcribe_media_app.speakers import VoiceRegistry, SpeakerRegistryError
from transcribe_media_app.storage import (ManifestStore, StateTransaction, atomic_write_json,
                                         source_fingerprint, project_lock, output_is_valid)


def evidence(vector, start=0):
    return {"embedding": vector, "clean_seconds": 12, "window_count": 3, "cohesion": 0.95,
            "windows": [{"start": start + i * 4, "end": start + (i + 1) * 4,
                         "duration": 4., "embedding": vector, "retained": True,
                         "reference_eligible": True, "cohesion_similarity": 0.95} for i in range(3)]}


def fixture(root):
    review = root / "Review"
    source = root / "conversation.wav"
    source.write_bytes(b"source-content")
    fingerprint = source_fingerprint(source)
    local = {"language": "en", "segments": [], "speaker_timeline": []}
    for index, speaker in enumerate(("A", "B", "C")):
        start, end = index * 12., (index + 1) * 12.
        text = ("Adult A words", "Adult B words", "Child speaks here")[index]
        local["segments"].append({"speaker": speaker, "start": start, "end": end,
            "text": text, "words": [{"word": text, "start": start, "end": end, "speaker": speaker, "score": .9}]})
        local["speaker_timeline"].append({"start": start, "end": end, "speaker": speaker})
    vectors = {"A": [1., 0., 0.], "B": [0., 1., 0.], "C": [0., 0., 1.]}
    samples = {key: evidence(value, i * 12) for i, (key, value) in enumerate(vectors.items())}
    registry = VoiceRegistry(review / "speaker_registry.json", reviewed=True)
    report = registry.identify(source_key=source.name, source_fingerprint=fingerprint["sha256"],
                               local_speakers=vectors, evidence=samples)
    payload = {"schema_version": RESULT_SCHEMA_VERSION,
               "source": {"relative_path": source.name, "fingerprint": fingerprint},
               "language": {"output": "en", "task": "transcribe"},
               "processing": {"completed_utc": "today", "settings_hash": "settings",
                              "runtime": {"device": "cpu"}, "provenance": {}, "degraded_stages": []},
               "segments": normalize_segments(local), "turns": build_turns(normalize_segments(local)),
               "speaker_timeline": copy.deepcopy(local["speaker_timeline"]), "speaker_identity": report}
    outputs = {"json": review / "conversation.json", "txt": root / "Transcribed/conversation.txt",
               "detailed_txt": review / "conversation.detailed.txt"}
    path = save_review(review, source, source.name, fingerprint, local, samples, payload, outputs,
                       registry, np.zeros(36 * SAMPLE_RATE, dtype=np.float32))
    manifest = ManifestStore(review / "transcription_manifest.json")
    manifest.update(source.name, {"source_fingerprint": fingerprint, "outputs": {key: str(value) for key, value in outputs.items()}})
    packet = json.loads(path.read_text())
    decisions = {"packet": path.name, "review_id": packet["review_id"],
                 "assignments": {"A": "new:Adult A", "B": "new:Adult B", "C": "new:Child"},
                 "new_profiles": {"new:Adult A": {"label": "Adult A", "role": "adult"},
                                  "new:Adult B": {"label": "Adult B", "role": "adult"},
                                  "new:Child": {"label": "Child", "role": "child"}}}
    decision_path = root / "decisions.json"
    atomic_write_json(decision_path, decisions)
    return review, path, decision_path, registry, local, samples, outputs


class ReviewTests(unittest.TestCase):
    def test_partial_range_separates_two_voices_within_one_local_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            review, path, decisions_path, registry, _, _, outputs = fixture(Path(directory))
            packet = json.loads(path.read_text())
            segment = packet["local_result"]["segments"][0]
            segment["words"] = [{"word": "Adult", "start": 0., "end": 8., "speaker": "A", "score": .9},
                                {"word": "child", "start": 8., "end": 12., "speaker": "A", "score": .9}]
            packet["payload"]["segments"] = normalize_segments(packet["local_result"])
            packet["review_id"] = packet_digest(packet)
            atomic_write_json(path, packet)
            decisions = json.loads(decisions_path.read_text())
            decisions["review_id"] = packet["review_id"]
            first_turn = build_turns(normalize_segments(packet["local_result"]))[0]
            decisions["range_overrides"] = [{"turn_id": first_turn["id"], "start": 8., "end": 12., "profile": "new:Child"}]
            atomic_write_json(decisions_path, decisions)
            applied = apply_review(review, decisions_path)
            payload = json.loads(outputs["json"].read_text())
            self.assertEqual(payload["turns"][0]["speaker"], applied["created_profiles"]["new:Adult A"])
            self.assertEqual(payload["turns"][1]["speaker"], applied["created_profiles"]["new:Child"])
            profiles = registry.profiles_for_review()
            self.assertEqual(next(p for p in profiles if p["label"] == "Adult A")["verified_windows"], 2)
            self.assertEqual(next(p for p in profiles if p["label"] == "Child")["verified_windows"], 4)

    def test_unchecking_old_reference_removes_it_from_training_without_reusing_id(self):
        with tempfile.TemporaryDirectory() as directory:
            review, path, decision_path, registry, _, _, _ = fixture(Path(directory))
            created = apply_review(review, decision_path)["created_profiles"]
            packet = json.loads(path.read_text())
            decisions = {"packet": path.name, "review_id": packet["review_id"],
                         "assignments": {"A": created["new:Adult A"]},
                         "exclude_windows": ["A:0", "A:1", "A:2"]}
            atomic_write_json(decision_path, decisions)
            apply_review(review, decision_path)
            profile = next(p for p in registry.profiles_for_review() if p["label"] == "Adult A")
            self.assertEqual(profile["voice_id"], created["new:Adult A"])
            self.assertEqual(profile["verified_windows"], 0)
            registry.validate()

    def test_human_unknown_reopens_review_and_marks_transcript_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            review, path, decisions, _, _, _, outputs = fixture(Path(directory))
            apply_review(review, decisions)
            packet = json.loads(path.read_text())
            atomic_write_json(decisions, {"packet": path.name, "review_id": packet["review_id"],
                                          "assignments": {"A": "unknown"}})
            self.assertEqual(apply_review(review, decisions)["pending"], 1)
            self.assertTrue(outputs["txt"].read_text().startswith("DRAFT:"))

    def test_refresh_preserves_human_choices_and_does_not_change_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            review, packet, decisions, registry, _, _, outputs = fixture(Path(directory))
            apply_review(review, decisions)
            before = registry.path.read_bytes()
            refresh_reviews(review)
            self.assertEqual(registry.path.read_bytes(), before)
            payload = json.loads(outputs["json"].read_text())
            self.assertEqual(payload["speaker_review"]["pending"], [])
            self.assertTrue(all(turn["speaker_identity"]["status"] == "human_verified" for turn in payload["turns"]))

    def test_first_run_confirm_then_match_frozen_across_another_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            review, packet, decisions, registry, local, samples, outputs = fixture(Path(directory))
            self.assertEqual(registry.validate()["profile_count"], 0)
            applied = apply_review(review, decisions)
            self.assertEqual(applied["pending"], 0)
            self.assertEqual(registry.validate()["profile_count"], 3)
            profiles = registry.profiles_for_review()
            self.assertTrue(all(item["verified_windows"] == 3 for item in profiles))
            before = registry.path.read_bytes()
            frozen = VoiceRegistry(registry.path, reviewed=True, learn=False)
            result = frozen.identify(source_key="another.wav", source_fingerprint="another",
                                     local_speakers=samples, evidence=samples)
            self.assertEqual({item["status"] for item in result["matches"]}, {"matched"})
            self.assertEqual(registry.path.read_bytes(), before)
            self.assertTrue(apply_review(review, decisions)["already_applied"])
            payload = json.loads(outputs["json"].read_text())
            self.assertEqual(len({turn["speaker"] for turn in payload["turns"]}), 3)
            self.assertEqual([word["confidence"] for segment in payload["segments"] for word in segment["words"]], [.9, .9, .9])
            restored = copy.deepcopy(local)
            restore_reviewed_choices(review, "conversation.wav", source_fingerprint(Path(directory) / "conversation.wav"), local, restored)
            self.assertEqual({word["speaker"] for segment in restored["segments"] for word in segment["words"]},
                             {item["voice_id"] for item in profiles})
            self.assertEqual(evaluate_registry(review)["trials"], 0)

    def test_source_change_and_invalid_profile_do_not_mutate_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            review, packet, decisions, registry, _, _, _ = fixture(Path(directory))
            before = registry.path.read_bytes()
            data = json.loads(decisions.read_text())
            data["assignments"]["A"] = "VOICE_9999"
            atomic_write_json(decisions, data)
            with self.assertRaisesRegex(SpeakerRegistryError, "unknown profile"):
                apply_review(review, decisions)
            self.assertEqual(registry.path.read_bytes(), before)
            (Path(directory) / "conversation.wav").write_bytes(b"changed")
            with self.assertRaisesRegex(SpeakerRegistryError, "source recording changed"):
                apply_review(review, decisions)
            self.assertEqual(registry.path.read_bytes(), before)

    def test_failed_output_rolls_back_first_enrollment_and_can_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            review, packet, decisions, registry, _, _, _ = fixture(Path(directory))
            before = registry.path.read_bytes()
            with mock.patch("transcribe_media_app.review.write_outputs", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    apply_review(review, decisions)
            self.assertEqual(registry.path.read_bytes(), before)
            self.assertFalse((review / ".state-transaction.json").exists())
            self.assertEqual(apply_review(review, decisions)["pending"], 0)

    def test_turn_override_can_isolate_child_from_an_adult_cluster(self):
        with tempfile.TemporaryDirectory() as directory:
            review, path, decisions_path, registry, _, _, outputs = fixture(Path(directory))
            decisions = json.loads(decisions_path.read_text())
            packet = json.loads(path.read_text())
            # Same local group is intentionally assigned Adult A; the child turn overrides it.
            decisions["assignments"]["C"] = "new:Adult A"
            child_turn = build_turns(normalize_segments(packet["local_result"]))[2]
            decisions["turn_overrides"] = {child_turn["id"]: "new:Child"}
            atomic_write_json(decisions_path, decisions)
            apply_review(review, decisions_path)
            payload = json.loads(outputs["json"].read_text())
            self.assertNotEqual(payload["turns"][0]["speaker"], payload["turns"][2]["speaker"])
            self.assertEqual(next(p for p in registry.profiles_for_review() if p["role"] == "child")["verified_windows"], 3)

    def test_explicit_merge_updates_transcripts_and_keeps_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            review, packet, decisions, registry, _, _, outputs = fixture(Path(directory))
            created = apply_review(review, decisions)["created_profiles"]
            source, target = created["new:Adult B"], created["new:Adult A"]
            result = merge_project_profiles(review, source, target)
            self.assertEqual(result["aliases"][source], target)
            payload = json.loads(outputs["json"].read_text())
            self.assertEqual(payload["turns"][0]["speaker"], payload["turns"][1]["speaker"])
            self.assertEqual(registry.validate()["profile_count"], 2)


class EvidenceTests(unittest.TestCase):
    def test_multiple_verified_recording_conditions_resolve_changed_voice(self):
        with tempfile.TemporaryDirectory() as directory:
            review, _, decisions, registry, _, samples, _ = fixture(Path(directory))
            created = apply_review(review, decisions)["created_profiles"]
            changed = [0.60, 0.59, 0.54]
            report = registry.identify(source_key="changed", source_fingerprint="changed", local_speakers=["A"], evidence={"A": evidence(changed)})
            self.assertTrue(report["matches"][0]["status"].startswith("unresolved"))
            refs = [{"profile": created["new:Adult A"], "observation": VoiceRegistry._observation("changed.wav", "new-condition", str(index),
                     {"embedding": changed, "clean_seconds": 4., "window_count": 1, "cohesion": 1.})} for index in range(3)]
            registry.confirm_references(refs, review_id="another-condition")
            report = registry.identify(source_key="future", source_fingerprint="future", local_speakers=["A"], evidence={"A": evidence(changed)})
            self.assertEqual(report["matches"][0]["speaker"], created["new:Adult A"])
            self.assertEqual(report["matches"][0]["status"], "matched")

    def test_short_disjoint_replies_are_not_stitched_into_reference_windows(self):
        timeline = [{"start": i * 3, "end": i * 3 + .7, "speaker": "A"} for i in range(5)]
        encoded = mock.Mock(return_value=np.array([1., 0.]))
        result = extract_reference_evidence(np.ones(16 * SAMPLE_RATE) * .1, timeline, {}, encoded)
        self.assertEqual(result, {})
        encoded.assert_not_called()

    def test_disagreement_is_manual_only_and_clipping_is_unusable(self):
        timeline = [{"start": 0., "end": 6., "speaker": "A"}]
        encoded = mock.Mock(return_value=np.array([1., 0.]))
        words = {"segments": [{"words": [{"start": 0., "end": 6., "word": "hello", "speaker": "A", "sortformer_speaker": "B"}]}]}
        disputed = extract_reference_evidence(np.ones(6 * SAMPLE_RATE) * .1, timeline, words, encoded)["A"]
        self.assertEqual(disputed["window_count"], 0)
        self.assertEqual(disputed["clean_seconds"], 0)
        self.assertTrue(disputed["windows"][0]["reference_eligible"])
        self.assertFalse(disputed["windows"][0]["automatic_reference_eligible"])
        encoded.reset_mock()
        self.assertEqual(extract_reference_evidence(np.ones(6 * SAMPLE_RATE), timeline, {}, encoded), {})
        encoded.assert_not_called()

    def test_disputed_majority_does_not_move_the_automatic_reference_anchor(self):
        timeline = [{'start': i*7., 'end': i*7.+6., 'speaker': 'A'} for i in range(5)]
        words = {'segments': [{'words': [{**turn, 'word': 'hello',
                  'sortformer_speaker': 'B' if i < 3 else 'A'} for i, turn in enumerate(timeline)]}]}
        encoded = mock.Mock(side_effect=[np.array([0., 1.])]*3 + [np.array([1., 0.])]*2)
        result = extract_reference_evidence(np.ones(35*SAMPLE_RATE)*.1, timeline, words, encoded)['A']
        self.assertEqual(result['window_count'], 2)
        self.assertEqual(result['embedding'], [1., 0.])
        self.assertFalse(any(w['automatic_reference_eligible'] for w in result['windows'][:3]))

    def test_mixed_cluster_requires_review_even_with_a_strong_centroid(self):
        with tempfile.TemporaryDirectory() as directory:
            review, _, decisions, registry, _, samples, _ = fixture(Path(directory))
            apply_review(review, decisions)
            samples["A"]["suspected_mixed_speakers"] = True
            result = registry.identify(source_key="test", source_fingerprint="test", local_speakers=["A"], evidence={"A": samples["A"]})
            self.assertTrue(result["matches"][0]["status"].startswith("unresolved"))


class IntegrityTests(unittest.TestCase):
    def test_quality_decoder_uses_real_fallback_controls_and_keeps_diagnostics(self):
        backend = WhisperXBackend.__new__(WhisperXBackend)
        backend.quality = True
        backend.runtime = RuntimeSettings("cpu", "int8", 1, 1, False, "test")
        backend.language, backend.task, backend.model_name, backend.vad_device = "en", "transcribe", "large-v3", "cpu"
        backend._decode_audio = mock.Mock(return_value=np.zeros(SAMPLE_RATE))
        backend.model = mock.Mock()
        backend.model.model.transcribe.return_value = ([SimpleNamespace(start=0., end=1., text="Hello", avg_logprob=-.2,
            no_speech_prob=.01, compression_ratio=1., temperature=0.)], SimpleNamespace(language="en"))
        result, _, provenance, degraded = backend.transcribe(Path("test.wav"), False, False)
        options = backend.model.model.transcribe.call_args.kwargs
        self.assertEqual(options["temperature"], (0., .2, .4, .6, .8, 1.))
        self.assertEqual(options["log_prob_threshold"], -1.)
        backend.model.transcribe.assert_not_called()
        self.assertEqual(provenance["decoding_mode"], "sequential_temperature_fallback")
        self.assertEqual(result["asr_diagnostics"][0]["avg_logprob"], -.2)
        self.assertEqual(degraded, [])

    def test_ambiguous_secondary_label_mapping_abstains(self):
        mapped, report = _map_secondary_timeline(
            [{"start": 0., "end": 10., "speaker": "A"}, {"start": 10., "end": 20., "speaker": "B"}],
            [{"start": 0., "end": 20., "speaker": "X"}])
        self.assertIsNone(mapped[0]["speaker"])
        self.assertFalse(report[0]["mapping_confident"])

    def test_quality_cli_review_and_policy_rerun_reuse_asr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, packet_path_, decisions, _, local, samples, _ = fixture(root)
            audio = np.zeros(36 * SAMPLE_RATE, dtype=np.float32)
            runtime = RuntimeSettings("cpu", "int8", 1, 1, False, "test")
            backend = mock.Mock()
            backend.transcribe.side_effect = lambda *_: (copy.deepcopy(local), audio, {}, [])
            backend._decode_audio.return_value = audio
            diarizer = mock.Mock(name="diarizer")
            diarizer.name = "speechbrain"
            diarizer.model_name = "test"
            diarizer.last_run = {}
            diarizer.assign.side_effect = lambda source, audio, result, *_: result
            encoder = mock.Mock()
            encoder.name, encoder.version = "test", "test"
            encoder.extract.return_value = samples
            args = [str(root), "--extensions", "wav", "--device", "cpu",
                    "--review-dir", str(review), "--transcript-dir", str(root / "Transcribed"),
                    "--diarization-backend", "speechbrain", "--no-acoustic", "--no-speaker-refinement"]
            with mock.patch.object(cli, "resolve_runtime", return_value=runtime), \
                 mock.patch.object(cli.WhisperXBackend, "create", return_value=(backend, runtime)), \
                 mock.patch.object(cli, "create_diarizer", return_value=diarizer), \
                 mock.patch.object(cli, "SpeakerIdentityEncoder", return_value=encoder), \
                 mock.patch.object(cli, "ffprobe_duration", return_value=36.):
                self.assertEqual(cli.main(args), 0)
                packet = json.loads(packet_path_.read_text())
                decision_data = json.loads(decisions.read_text())
                decision_data["review_id"] = packet["review_id"]
                atomic_write_json(decisions, decision_data)
                self.assertEqual(cli.main(["--review-dir", str(review), "--apply-speaker-review", str(decisions)]), 0)
                self.assertEqual(cli.main(args + ["--no-speaker-learning"]), 0)
            self.assertEqual(backend.transcribe.call_count, 1)
            backend._decode_audio.assert_called_once()
    def test_full_hash_detects_middle_edit_with_same_size_and_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio"
            path.write_bytes(b"a" * (4 * 1024 * 1024))
            before = source_fingerprint(path)
            with path.open("r+b") as handle:
                handle.seek(2 * 1024 * 1024)
                handle.write(b"b")
            os.utime(path, ns=(path.stat().st_atime_ns, before["mtime_ns"]))
            self.assertNotEqual(source_fingerprint(path)["sha256"], before["sha256"])

    def test_crash_recovery_restores_registry_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, new = root / "state.json", root / "new.txt"
            original.write_text("before")
            with project_lock(root):
                transaction = StateTransaction(root, [original, new])
                transaction.begin()
                original.write_text("after")
                new.write_text("partial")
            with project_lock(root):
                self.assertTrue(StateTransaction(root).rollback())
            self.assertEqual(original.read_text(), "before")
            self.assertFalse(new.exists())
            self.assertEqual(original.stat().st_mode & 0o777, 0o600)

    def test_concurrent_batch_or_review_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with project_lock(Path(directory)):
                with self.assertRaisesRegex(RuntimeError, "another transcription"):
                    with project_lock(Path(directory)):
                        pass

    def test_quality_defaults_allow_optional_third_speaker(self):
        parser = cli.build_parser()
        for flags in ([], ["--quality"]):
            with self.subTest(flags=flags):
                args = parser.parse_args(flags)
                cli._validate_args(parser, args)
                self.assertTrue(args.quality)
                self.assertEqual(args.model, "large-v3")
                self.assertEqual(args.language, "en")
                self.assertEqual(args.diarization_backend, "auto")
                self.assertTrue(args.speaker_identity and args.align)
                self.assertEqual((args.min_speakers, args.max_speakers, args.batch_size, args.tone_backend), (2, 3, 1, "off"))

    def test_quality_defaults_respect_explicit_speaker_counts(self):
        parser = cli.build_parser()
        for flags, bounds in ((["--speakers", "2"], (2, 2)),
                              (["--speakers", "1"], (1, 1)),
                              (["--min-speakers", "1", "--max-speakers", "3"], (1, 3))):
            with self.subTest(flags=flags):
                args = parser.parse_args(flags)
                cli._validate_args(parser, args)
                self.assertTrue(args.quality)
                self.assertEqual((args.min_speakers, args.max_speakers), bounds)

    def test_disabling_required_quality_stages_requires_explicit_opt_out(self):
        parser = cli.build_parser()
        for flags in (["--no-align"], ["--no-speaker-identity"],
                      ["--no-diarize"], ["--task", "translate"]):
            with self.subTest(flags=flags):
                with self.assertRaises(SystemExit):
                    cli._validate_args(parser, parser.parse_args(flags))
                legacy = parser.parse_args(["--no-quality", *flags])
                cli._validate_args(parser, legacy)
                self.assertFalse(legacy.quality)
                self.assertIsNone(legacy.min_speakers)
                self.assertIsNone(legacy.max_speakers)

    def test_doctor_honors_explicit_cuda_requirement(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--doctor", "--device", "cuda"])
        cli._validate_args(parser, args)
        with mock.patch.object(cli, "_doctor_check", return_value=True), \
             mock.patch.object(cli, "package_version", return_value=None), \
             mock.patch.object(cli, "resolve_hf_token", return_value=None), \
             mock.patch.object(cli, "_cuda_available", return_value=False), \
             mock.patch.object(cli, "resolve_runtime", return_value=SimpleNamespace(description="test")), \
             mock.patch.object(cli, "resolve_analysis_backends", return_value=("speechbrain", "off", [])):
            self.assertEqual(cli.doctor(args), 1)

    def test_raw_overlap_is_visible_without_inventing_interruption(self):
        turns = [{"start": 0., "end": 4., "speaker": "A", "words": []}]
        annotate_speaker_attribution(turns)
        annotate_raw_overlap(turns, [{"start": 0., "end": 4., "speaker": "A"}, {"start": 2., "end": 3., "speaker": "B"}])
        self.assertEqual(turns[0]["speaker_attribution"]["status"], "uncertain")
        self.assertTrue(turns[0]["acoustic_overlap"])
        self.assertNotIn("interruption_of", turns[0])

    def test_empty_srt_is_a_valid_no_speech_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.srt"
            path.touch()
            self.assertTrue(output_is_valid(path, "srt"))
