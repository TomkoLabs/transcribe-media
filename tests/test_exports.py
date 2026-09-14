import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from transcribe_media_app import cli
from transcribe_media_app.exports import render_saved_transcripts
from transcribe_media_app.schema import RESULT_SCHEMA_VERSION
from transcribe_media_app.storage import ManifestStore, atomic_write_json


def saved_results(root, count=2):
    review = root / "Review"
    manifest = ManifestStore(review / "transcription_manifest.json")
    jobs = []
    for index in range(count):
        source = f"session-{index}.wav"
        payload = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "source": {"relative_path": source}, "language": {"output": "en"},
            "processing": {"completed_utc": "synthetic", "settings_hash": "unchanged"},
            "speaker_profiles": [{"voice_id": "VOICE_0001", "label": "Adult A", "role": "adult"}],
            "speaker_review": {"pending": []},
            "turns": [{"speaker": "VOICE_0001", "text": "These exact words stay.",
                       "start": 1., "end": 3., "speaker_attribution": {"status": "uncertain"}}],
        }
        payload["segments"] = copy.deepcopy(payload["turns"])
        files = {"json": review / f"{source}.json", "txt": root / "Transcribed" / f"{source}.txt",
                 "detailed_txt": review / f"{source}.detailed.txt", "srt": review / f"{source}.srt"}
        atomic_write_json(files["json"], payload)
        for key, path in files.items():
            if key != "json":
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("Old presentation.\n")
        manifest.update(source, {"outputs": {key: str(path) for key, path in files.items()},
                                 "status": "complete", "completion_state": True, "settings_hash": "unchanged"})
        jobs.append(files)
    (review / "speaker_registry.json").write_text('{"synthetic": "must not be read or modified"}')
    return review, manifest.path, jobs


class SavedExportTests(unittest.TestCase):
    def test_cli_rebuilds_exports_without_models_media_or_changing_saved_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, manifest, jobs = saved_results(root)
            protected = [manifest, review / "speaker_registry.json", *(files["json"] for files in jobs)]
            before = {path: path.read_bytes() for path in protected}
            with (mock.patch.dict('os.environ', {'TRANSCRIBE_MEDIA_ROOT': str(root)}),
                  mock.patch.object(cli, 'resolve_runtime', side_effect=AssertionError('GPU queried')),
                  mock.patch.object(cli.WhisperXBackend, 'create', side_effect=AssertionError('ASR loaded')),
                  mock.patch.object(cli.VoiceRegistry, 'identify', side_effect=AssertionError('voices rematched'))):
                self.assertEqual(cli.main(['--render-transcripts']), 0)
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)
            for files in jobs:
                self.assertIn('Adult A [VOICE_0001]:\nThese exact words stay.', files['txt'].read_text())
                self.assertNotIn('speaker attribution uncertain', files['txt'].read_text())
                self.assertIn('[Speaker attribution: uncertain', files['detailed_txt'].read_text())
                self.assertIn('[Adult A [VOICE_0001]] These exact words stay.', files['srt'].read_text())
            self.assertFalse((review / '.state-transaction.json').exists())

    def test_late_write_failure_restores_every_export(self):
        with tempfile.TemporaryDirectory() as directory:
            review, _, jobs = saved_results(Path(directory))
            before = {path: path.read_bytes() for files in jobs for path in files.values()}
            from transcribe_media_app.renderers import write_outputs
            def fail_second(files, payload):
                write_outputs(files, payload)
                if payload['source']['relative_path'] == 'session-1.wav':
                    raise OSError('disk full')
            with mock.patch('transcribe_media_app.exports.write_outputs', side_effect=fail_second):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    render_saved_transcripts(review)
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

    def test_missing_or_invalid_json_does_not_partially_rewrite_a_batch(self):
        for contents in (None, '{broken', '{}'):
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as directory:
                review, _, jobs = saved_results(Path(directory))
                before = jobs[0]['txt'].read_bytes()
                if contents is None:
                    jobs[1]['json'].unlink()
                else:
                    jobs[1]['json'].write_text(contents)
                with self.assertRaises(ValueError):
                    render_saved_transcripts(review)
                self.assertEqual(jobs[0]['txt'].read_bytes(), before)

    def test_export_path_cannot_overwrite_another_recordings_canonical_json(self):
        with tempfile.TemporaryDirectory() as directory:
            review, manifest, jobs = saved_results(Path(directory))
            data = json.loads(manifest.read_text())
            data['sources']['session-0.wav']['outputs']['txt'] = str(jobs[1]['json'])
            atomic_write_json(manifest, data)
            before = jobs[1]['json'].read_bytes()
            with self.assertRaisesRegex(ValueError, 'paths overlap'):
                render_saved_transcripts(review)
            self.assertEqual(jobs[1]['json'].read_bytes(), before)

    def test_corrupt_manifest_is_not_reported_as_an_empty_project(self):
        with tempfile.TemporaryDirectory() as directory:
            review, manifest, _ = saved_results(Path(directory))
            manifest.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'invalid processing manifest'):
                render_saved_transcripts(review)

    def test_empty_project_needs_no_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(render_saved_transcripts(Path(directory)), {'recordings_rendered': 0, 'files_written': 0})

    def test_maintenance_actions_cannot_silently_override_each_other(self):
        parser = cli.build_parser()
        for flags in (['--render-transcripts', '--refresh-voices'], ['--render-transcripts', '--doctor'],
                      ['--review-speakers', '--configure'], ['--apply-speaker-review', 'decisions.json', '--prepare-models'],
                      ['--render-transcripts', '--dry-run'], ['--apply-speaker-review', 'decisions.json', '--dry-run']):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                cli._validate_args(parser, parser.parse_args(flags))
