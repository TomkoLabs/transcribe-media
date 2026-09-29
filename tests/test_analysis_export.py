"""Synthetic upstream review fixtures; no media inference or network."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from test_batch_review import batch_fixture
from transcribe_media_app.review import apply_review
from transcribe_media_app.analysis_export import export_analysis_ready, MARKER
from transcribe_media_app.storage import ManifestStore
from transcribe_media_app import cli


class AnalysisReadyExportTests(unittest.TestCase):
    def test_partial_export_stale_removal_unicode_and_cli_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, packets, outputs, registry, _, decisions = batch_fixture(root, 3)
            apply_review(review, decisions)
            transcript = root / "Transcribed"
            dest = root / "EXPORT_TRANSCRIBED"
            # Current rendered TXT is resolved from the manifest, retaining its relative Unicode path.
            manifest = ManifestStore(review / "transcription_manifest.json")
            key = json.loads(packets[0].read_text())["source_key"]
            moved = transcript / "nested/計画 café.txt"
            moved.parent.mkdir(parents=True)
            outputs[0]["txt"].rename(moved)
            entry = manifest.data["sources"][key]
            entry["outputs"]["txt"] = str(moved)
            manifest.update(key, entry)
            result = export_analysis_ready(review, transcript, dest)
            self.assertEqual(result["included"], 3)
            self.assertEqual(
                moved.read_bytes(), (dest / "nested/計画 café.txt").read_bytes()
            )
            old = (dest / MARKER).read_bytes()
            # One outdated/draft rendering does not block the other ready inputs.
            outputs[1]["txt"].write_text("DRAFT: SPEAKER REVIEW REQUIRED\n")
            result = export_analysis_ready(review, transcript, dest)
            self.assertEqual((result["included"], result["excluded"]), (2, 1))
            self.assertFalse(
                (dest / outputs[1]["txt"].relative_to(transcript)).exists()
            )
            self.assertTrue(Path(result["previous_generation"]).exists())
            self.assertNotEqual(old, (dest / MARKER).read_bytes())
            with (
                mock.patch.dict("os.environ", {"TRANSCRIBE_MEDIA_ROOT": str(root)}),
                mock.patch.object(
                    cli,
                    "resolve_runtime",
                    side_effect=AssertionError("inference forbidden"),
                ),
            ):
                self.assertEqual(cli.main(["--export-analysis-ready"]), 0)

    def test_operator_files_and_unowned_directory_never_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, packets, outputs, registry, _, decisions = batch_fixture(root, 1)
            apply_review(review, decisions)
            dest = root / "EXPORT_TRANSCRIBED"
            dest.mkdir()
            operator = dest / "operator.txt"
            operator.write_text("Preserve me")
            with self.assertRaises(ValueError):
                export_analysis_ready(review, root / "Transcribed", dest)
            self.assertEqual(operator.read_text(), "Preserve me")
            other = root / "managed"
            export_analysis_ready(review, root / "Transcribed", other)
            (other / "operator.txt").write_text("Still mine")
            with self.assertRaises(ValueError):
                export_analysis_ready(review, root / "Transcribed", other)
            self.assertEqual((other / "operator.txt").read_text(), "Still mine")

    def test_failed_publication_preserves_current_and_changed_source_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, packets, outputs, registry, _, decisions = batch_fixture(root, 2)
            apply_review(review, decisions)
            dest = root / "EXPORT_TRANSCRIBED"
            export_analysis_ready(review, root / "Transcribed", dest)
            before = {
                p.relative_to(dest): p.read_bytes()
                for p in dest.rglob("*")
                if p.is_file()
            }
            with (
                mock.patch(
                    "transcribe_media_app.analysis_export._exchange",
                    side_effect=OSError("synthetic disk failure"),
                ),
                self.assertRaises(OSError),
            ):
                export_analysis_ready(review, root / "Transcribed", dest)
            self.assertEqual(
                before,
                {
                    p.relative_to(dest): p.read_bytes()
                    for p in dest.rglob("*")
                    if p.is_file()
                },
            )
            packet = json.loads(packets[0].read_text())
            Path(packet["source"]).write_bytes(b"changed synthetic source")
            result = export_analysis_ready(review, root / "Transcribed", dest)
            self.assertEqual((result["included"], result["excluded"]), (1, 1))
