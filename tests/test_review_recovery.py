"""Synthetic media only; no inference or access to an operator Review tree."""
import copy
import io
import json
import os
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from test_batch_review import batch_fixture
from test_quality import fixture
from transcribe_media_app import cli, __version__
from transcribe_media_app.exports import render_saved_transcripts
from transcribe_media_app.recovery import diagnose_reviews, recover_review, inspect_packet, pending_count
from transcribe_media_app.renderers import render_txt
from transcribe_media_app.review import apply_review, refresh_reviews, review_index, restore_reviewed_choices
from transcribe_media_app.speakers import SpeakerRegistryError
from transcribe_media_app.storage import StateTransaction, atomic_write_json, source_fingerprint


def public_data(path):
    return json.loads(re.search(r'<script type="application/json" id="review-data">(.*?)</script>',
                               path.read_text(), re.S).group(1))


class RecoveryTests(unittest.TestCase):
    def test_changed_source_isolated_from_two_current_recordings_and_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, packets, outputs, registry, _, decisions = batch_fixture(root, 3)
            apply_review(review, decisions)
            before_registry = registry.path.read_bytes()
            changed = json.loads(packets[0].read_text())
            old_export = outputs[0]['txt'].read_bytes()
            Path(changed['source']).write_bytes(b'new synthetic media content')
            result = refresh_reviews(review)
            states = {item['source']: item for item in result['recordings']}
            self.assertEqual(states[changed['source_key']]['state'], 'source_content_changed')
            self.assertEqual(sum(item['state'] == 'ready' for item in states.values()), 2)
            self.assertEqual(before_registry, registry.path.read_bytes())
            self.assertTrue(outputs[0]['txt'].read_text().startswith('DRAFT:'))
            self.assertTrue(outputs[1]['txt'].read_text().startswith('ANALYSIS-READY TRANSCRIPT'))
            packet = json.loads(packets[0].read_text())
            backup = Path(packet['recovery_backup'])
            inventory = json.loads((backup / 'inventory.json').read_text())
            self.assertEqual((backup / inventory[str(outputs[0]['txt'])]).read_bytes(), old_export)
            for record in public_data(Path(result['review_index']))['recordings']:
                self.assertEqual(record['saved_status']['pending'], states[record['source']]['pending'])
                self.assertEqual(record['saved_status']['state'], states[record['source']]['state'])
                self.assertEqual(record['decisions_allowed'], states[record['source']]['decisions_allowed'])
                self.assertTrue(record['saved_status']['snapshot_id'])
                self.assertTrue(record['saved_status']['generated_utc'])
            recovery = recover_review(review, changed['source_key'])
            self.assertEqual(recovery['state'], 'source_content_changed')
            self.assertTrue(recover_review(review, changed['source_key'])['already_recovered'])
            self.assertEqual(before_registry, registry.path.read_bytes())
            self.assertNotIn('recovered_utc', json.loads(packets[1].read_text()))

    def test_metadata_recovery_preserves_work_but_rejects_old_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, path, decisions, registry, _, _, outputs = fixture(root)
            apply_review(review, decisions)
            packet = json.loads(path.read_text())
            source = Path(packet['source'])
            os.utime(source, ns=(source.stat().st_atime_ns, source.stat().st_mtime_ns + 5000000000))
            self.assertEqual(inspect_packet(path)['state'], 'metadata_equivalent')
            registry_before = registry.path.read_bytes()
            original = {p: p.read_bytes() for p in [path, *outputs.values()]}
            self.assertEqual(recover_review(review, source.name, dry_run=True)['state'], 'metadata_equivalent')
            self.assertEqual(original, {p: p.read_bytes() for p in original})
            self.assertFalse((review / 'recovery-backups').exists())
            recovered = recover_review(review, source.name)
            self.assertEqual(recovered['state'], 'ready')
            current = json.loads(path.read_text())
            self.assertEqual(current['local_result'], packet['local_result'])
            self.assertEqual(current['applied_decisions'], packet['applied_decisions'])
            self.assertEqual(registry.path.read_bytes(), registry_before)
            self.assertNotEqual(packet['review_id'], current['review_id'])
            with self.assertRaises(SpeakerRegistryError) as context:
                apply_review(review, decisions)
            message = str(context.exception)
            for text in (decisions.name, source.name, 'stored=', 'current=', 'decision_file_stale', 'NEW decisions file'):
                self.assertIn(text, message)
            self.assertNotIn('Adult A words', message)
            fresh = json.loads(decisions.read_text())
            fresh['review_id'] = current['review_id']
            atomic_write_json(decisions, fresh)
            self.assertEqual(apply_review(review, decisions)['pending'], 0)
            before = path.read_bytes()
            recover_review(review, source.name)
            self.assertEqual(before, path.read_bytes())

    def test_mtime_only_refresh_does_not_retranscribe_or_discard_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            review, path, decisions, _, _, _, _ = fixture(Path(directory))
            apply_review(review, decisions)
            packet = json.loads(path.read_text())
            source = Path(packet['source'])
            os.utime(source, ns=(1, source.stat().st_mtime_ns + 1000000))
            result = refresh_reviews(review)['recordings'][0]
            self.assertEqual(result['state'], 'metadata_equivalent')
            self.assertEqual(result['pending'], 0)
            self.assertEqual(json.loads(path.read_text())['review_id'], packet['review_id'])
            self.assertTrue(apply_review(review, decisions)['already_applied'])

    def test_explicit_move_keeps_logical_key_and_requires_full_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, path, decisions, _, _, _, _ = fixture(root)
            apply_review(review, decisions)
            source = root / 'conversation.wav'
            target = root / 'relocated.wav'
            source.rename(target)
            self.assertEqual(inspect_packet(path)['state'], 'source_unavailable')
            self.assertEqual(recover_review(review, source.name, relocated_source=target, dry_run=True)['state'], 'metadata_equivalent')
            recover_review(review, source.name, relocated_source=target)
            packet = json.loads(path.read_text())
            self.assertEqual(packet['source_key'], source.name)
            self.assertEqual(packet['source'], str(target))
            self.assertEqual(inspect_packet(path)['state'], 'ready')
            with self.assertRaisesRegex(ValueError, 'No exact review source key'):
                recover_review(review, 'conversation')
            bad = root / 'different.wav'
            bad.write_bytes(b'different')
            with self.assertRaisesRegex(ValueError, 'matching full content'):
                recover_review(review, source.name, relocated_source=bad)

    def test_relocation_refuses_path_owned_by_another_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, packets, _, _, _, decisions = batch_fixture(root)
            apply_review(review, decisions)
            first, second = [json.loads(p.read_text()) for p in packets]
            Path(second['source']).write_bytes(Path(first['source']).read_bytes())
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                recover_review(review, first['source_key'], relocated_source=second['source'])

    def test_genuine_change_rejects_decisions_and_does_not_reuse_old_choices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, path, decisions, _, local, _, outputs = fixture(root)
            apply_review(review, decisions)
            source = root / 'conversation.wav'
            source.write_bytes(b'changed content')
            with self.assertRaisesRegex(SpeakerRegistryError, 'source_content_changed'):
                apply_review(review, decisions)
            recover_review(review, source.name)
            result = copy.deepcopy(local)
            restore_reviewed_choices(review, source.name, source_fingerprint(source), local, result)
            self.assertEqual(result, local)
            render_saved_transcripts(review)
            self.assertTrue(outputs['txt'].read_text().startswith('DRAFT:'))

    def test_interrupted_recovery_restores_from_existing_transaction_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, path, decisions, _, _, _, outputs = fixture(root)
            apply_review(review, decisions)
            protected = [path, path.with_suffix('.html'), review / 'transcription_manifest.json', *outputs.values()]
            before = {p: p.read_bytes() for p in protected}
            source = root / 'conversation.wav'
            os.utime(source, ns=(1, source.stat().st_mtime_ns + 1000000))
            with mock.patch('transcribe_media_app.recovery.atomic_write_json', wraps=atomic_write_json) as write:
                def interrupt(path_arg, value):
                    if Path(path_arg) == path:
                        raise KeyboardInterrupt()
                    return atomic_write_json(path_arg, value)
                write.side_effect = interrupt
                # Simulate process death: no in-process rollback, durable journal remains.
                with mock.patch.object(StateTransaction, 'rollback'), self.assertRaises(KeyboardInterrupt):
                    recover_review(review, source.name)
            self.assertTrue((review / '.state-transaction.json').exists())
            self.assertEqual(diagnose_reviews(review)['state'], 'interrupted_transaction')
            StateTransaction(review).rollback()
            self.assertEqual(before, {p: p.read_bytes() for p in protected})
            self.assertEqual(recover_review(review, source.name)['state'], 'ready')

    def test_malformed_packet_does_not_block_other_reviews(self):
        with tempfile.TemporaryDirectory() as directory:
            review, packets, _, _, _, decisions = batch_fixture(Path(directory))
            apply_review(review, decisions)
            packets[0].write_text('{not valid JSON: synthetic private dialogue')
            result = refresh_reviews(review)
            self.assertEqual({r['state'] for r in result['recordings']}, {'malformed_saved_state', 'ready'})
            data = public_data(Path(result['review_index']))
            self.assertEqual(sum(r['decisions_allowed'] for r in data['recordings']), 1)
            self.assertNotIn('synthetic private dialogue', json.dumps(result))

    def test_pending_counts_and_freshness_match_saved_state_not_browser_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            review, path, decisions, _, _, _, outputs = fixture(Path(directory))
            apply_review(review, decisions)
            before = public_data(path.with_suffix('.html'))
            packet = json.loads(path.read_text())
            atomic_write_json(decisions, {'packet': path.name, 'review_id': packet['review_id'], 'assignments': {'A': 'unknown'}})
            applied = apply_review(review, decisions)
            after = public_data(path.with_suffix('.html'))
            self.assertEqual(after['saved_status']['pending'], applied['pending'])
            self.assertEqual(applied['pending'], 1)
            self.assertNotEqual(before['saved_status']['snapshot_id'], after['saved_status']['snapshot_id'])
            self.assertEqual(after['saved_status']['program_version'], __version__)
            render_saved_transcripts(review)
            self.assertTrue(outputs['txt'].read_text().startswith('DRAFT:'))

    def test_read_only_cli_leaves_tree_unchanged_and_never_rolls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, _, _, _, _, _, _ = fixture(root)
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with mock.patch.dict(os.environ, {'TRANSCRIBE_MEDIA_ROOT': str(root)}), redirect_stdout(io.StringIO()), mock.patch.object(StateTransaction, 'rollback', side_effect=AssertionError('must not mutate')):
                self.assertEqual(cli.main(['--diagnose-reviews']), 0)
                self.assertEqual(cli.main(['--recover-review', 'conversation.wav', '--dry-run']), 0)
            self.assertEqual(before, {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_only_source_processes_one_exact_key_with_mocked_backend(self):
        from test_core import FakeBackend, FlowTests
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'Video Source'
            source.mkdir()
            (source / 'selected.wav').write_bytes(b'synthetic selected')
            (source / 'unrelated.wav').write_bytes(b'synthetic unrelated')
            with (mock.patch.object(cli, 'resolve_runtime', return_value=FlowTests._runtime()),
                  mock.patch.object(cli.WhisperXBackend, 'create', return_value=(FakeBackend(), FlowTests._runtime())),
                  mock.patch.object(cli, 'ffprobe_duration', return_value=1.0),
                  redirect_stdout(io.StringIO())):
                self.assertEqual(cli.main(FlowTests()._args(root, '--only-source', 'selected.wav')), 0)
            self.assertTrue((root / 'Review' / 'selected.wav.json').exists())
            self.assertFalse((root / 'Review' / 'unrelated.wav.json').exists())

    def test_fresh_partial_batch_applies_while_unrelated_changed_source_stays_blocked(self):
        from transcribe_media_app.review import batch_digest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, packets, outputs, _, batch, decisions = batch_fixture(root)
            apply_review(review, decisions)
            stale = json.loads(packets[0].read_text())
            Path(stale['source']).write_bytes(b'changed source')
            current = json.loads(packets[1].read_text())
            batch['reviews'] = [{'format_version': 2, 'packet': packets[1].name,
                                 'review_id': current['review_id'], **current['applied_decisions']}]
            batch['new_profiles'] = {}
            batch['batch_id'] = batch_digest(batch['reviews'])
            atomic_write_json(decisions, batch)
            applied = apply_review(review, decisions)
            self.assertEqual(applied['reviews_applied'], 1)
            self.assertTrue(outputs[0]['txt'].read_text().startswith('DRAFT:'))
            self.assertTrue(outputs[1]['txt'].read_text().startswith('ANALYSIS-READY'))

    def test_source_change_during_metadata_recovery_cannot_rebind_old_words(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, path, decisions, _, _, _, outputs = fixture(root)
            apply_review(review, decisions)
            source = root / 'conversation.wav'
            os.utime(source, ns=(1, source.stat().st_mtime_ns + 1000000))
            current = source_fingerprint(source)
            changed = {**current, 'sha256': '0' * 64}
            before = {p: p.read_bytes() for p in [path, *outputs.values()]}
            with mock.patch('transcribe_media_app.recovery.source_fingerprint', side_effect=[current, changed]):
                with self.assertRaisesRegex(RuntimeError, 'changed during recovery'):
                    recover_review(review, 'conversation.wav')
            self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_backup_failure_prevents_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, path, decisions, _, _, _, outputs = fixture(root)
            apply_review(review, decisions)
            before = {p: p.read_bytes() for p in [path, *outputs.values()]}
            (root / 'conversation.wav').write_bytes(b'new content')
            with mock.patch('transcribe_media_app.recovery._backup', side_effect=OSError('synthetic disk full')):
                with self.assertRaises(OSError):
                    recover_review(review, 'conversation.wav')
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            self.assertFalse((review / '.state-transaction.json').exists())

    def test_contract_headings_and_opaque_reversed_ids(self):
        for husband, wife in [('VOICE_0001', 'VOICE_0002'), ('VOICE_0002', 'VOICE_0001')]:
            payload = {'source': {'relative_path': 'synthetic.wav'}, 'language': {'output': 'en'},
                       'speaker_profiles': [{'voice_id': husband, 'label': 'Husband', 'role': 'adult'}, {'voice_id': wife, 'label': 'Wife', 'role': 'adult'}],
                       'turns': [{'speaker': husband, 'start': 0, 'end': 1, 'text': 'Synthetic first turn.'}, {'speaker': wife, 'start': 2, 'end': 3, 'text': 'Synthetic second turn.'}],
                       'speaker_review': {'pending': [], 'pending_turns': []}}
            ready = render_txt(payload)
            self.assertTrue(ready.startswith('ANALYSIS-READY TRANSCRIPT\n'))
            self.assertIn(f'Husband [{husband}]:\nSynthetic first turn.', ready)
            self.assertIn(f'Wife [{wife}]:\nSynthetic second turn.', ready)
            payload['speaker_review']['pending_turns'] = ['turn-1']
            self.assertEqual(pending_count(payload), 1)
            self.assertTrue(render_txt(payload).startswith('DRAFT:'))
            payload['speaker_review'] = {'pending': [], 'source_status': 'source_content_changed'}
            self.assertTrue(render_txt(payload).startswith('DRAFT:'))


if __name__ == '__main__':
    unittest.main()
