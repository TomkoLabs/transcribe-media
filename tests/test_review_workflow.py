import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_quality import fixture, evidence
from transcribe_media_app.review import apply_review, packet_digest, review_index, save_review, merge_project_profiles
from transcribe_media_app.speakers import VoiceRegistry
from transcribe_media_app.storage import atomic_write_json, ManifestStore
from transcribe_media_app.renderers import write_outputs


class SmoothReviewTests(unittest.TestCase):
    def test_no_reference_audio_still_saves_new_people_and_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            review, path, decisions, registry, _, _, outputs = fixture(Path(directory))
            packet = json.loads(path.read_text())
            packet['evidence'] = {}
            packet['review_id'] = packet_digest(packet)
            atomic_write_json(path, packet)
            decision = json.loads(decisions.read_text())
            decision['review_id'] = packet['review_id']
            atomic_write_json(decisions, decision)
            result = apply_review(review, decisions)
            self.assertEqual(result['pending'], 0)
            self.assertEqual(result['references_saved'], 0)
            self.assertEqual(len(result['profiles_needing_audio']), 3)
            self.assertEqual(registry.validate()['profile_count'], 3)
            self.assertFalse(outputs['txt'].read_text().startswith('DRAFT:'))
            report = VoiceRegistry(registry.path, reviewed=True, learn=False).identify(
                source_key='future', source_fingerprint='future', local_speakers=['A'], evidence={'A': evidence([1., 0., 0.])})
            self.assertTrue(report['matches'][0]['status'].startswith('unresolved'))

    def test_all_reference_checkboxes_off_does_not_block_initial_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            review, _, decisions, registry, *_ = fixture(Path(directory))
            data = json.loads(decisions.read_text())
            data['exclude_windows'] = [f'{local}:{i}' for local in 'ABC' for i in range(3)]
            atomic_write_json(decisions, data)
            self.assertEqual(apply_review(review, decisions)['references_saved'], 0)
            self.assertTrue(all(p['training_status'] == 'untrained' for p in registry.profiles_for_review()))

    def test_pauses_between_asr_turns_do_not_discard_reviewed_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            review, path, decisions, registry, *_ = fixture(Path(directory))
            packet = json.loads(path.read_text())
            segment = packet['local_result']['segments'][0]
            segment['start'], segment['end'] = 1., 2.
            segment['words'][0].update(start=1., end=2.)
            packet['review_id'] = packet_digest(packet)
            atomic_write_json(path, packet)
            data = json.loads(decisions.read_text()); data['review_id'] = packet['review_id']
            atomic_write_json(decisions, data)
            result = apply_review(review, decisions)
            adult = next(p for p in registry.profiles_for_review() if p['label'] == 'Adult A')
            self.assertEqual(adult['verified_windows'], 1)
            self.assertEqual(result['pending'], 0)

    def test_modified_old_draft_reuses_created_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            review, _, decisions, registry, *_ = fixture(Path(directory))
            created = apply_review(review, decisions)['created_profiles']
            data = json.loads(decisions.read_text()); data['exclude_windows'] = ['A:0']
            atomic_write_json(decisions, data)
            result = apply_review(review, decisions)
            self.assertEqual(result['created_profiles'], {})
            self.assertEqual(registry.validate()['profile_count'], 3)
            self.assertEqual({p['voice_id'] for p in registry.profiles_for_review()}, set(created.values()))

    def test_rebuild_and_merge_preserve_draft_person_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            review, path, decisions, registry, local, samples, outputs = fixture(Path(directory))
            created = apply_review(review, decisions)['created_profiles']
            packet = json.loads(path.read_text())
            save_review(review, Path(packet['source']), packet['source_key'], packet['fingerprint'],
                        local, samples, packet['payload'], outputs, registry, [])
            self.assertEqual(json.loads(path.read_text())['new_profile_ids'], created)
            merge_project_profiles(review, created['new:Adult B'], created['new:Adult A'])
            self.assertEqual(json.loads(path.read_text())['new_profile_ids']['new:Adult B'], created['new:Adult A'])
            data = json.loads(decisions.read_text()); data['exclude_windows'] = ['A:0']
            atomic_write_json(decisions, data)
            self.assertEqual(apply_review(review, decisions)['created_profiles'], {})
            self.assertEqual(registry.validate()['profile_count'], 2)

    def test_older_successful_release_receipt_recovers_draft_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            review, path, decisions, registry, *_ = fixture(Path(directory))
            created = apply_review(review, decisions)['created_profiles']
            packet = json.loads(path.read_text()); del packet['new_profile_ids']
            saved = json.loads(registry.path.read_text())
            saved['applied_reviews'][packet['receipts'][-1]] = created
            atomic_write_json(registry.path, saved); atomic_write_json(path, packet)
            data = json.loads(decisions.read_text()); data['exclude_windows'] = ['A:0']
            atomic_write_json(decisions, data)
            self.assertEqual(apply_review(review, decisions)['created_profiles'], {})
            self.assertEqual(registry.validate()['profile_count'], 3)

    def test_reapplying_an_older_snapshot_restores_its_reference_choices(self):
        with tempfile.TemporaryDirectory() as directory:
            review, _, decisions, registry, *_ = fixture(Path(directory))
            original = json.loads(decisions.read_text()); original['format_version'] = 2
            atomic_write_json(decisions, original)
            apply_review(review, decisions)
            atomic_write_json(decisions, {**original, 'exclude_windows': ['A:0', 'A:1', 'A:2']})
            apply_review(review, decisions)
            self.assertEqual(next(p for p in registry.profiles_for_review() if p['label']=='Adult A')['verified_windows'], 0)
            atomic_write_json(decisions, original)
            self.assertEqual(apply_review(review, decisions)['references_saved'], 9)
            self.assertTrue(apply_review(review, decisions)['already_applied'])

    def test_automatic_assignment_does_not_become_a_training_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            review, path, decisions, registry, _, _, outputs = fixture(Path(directory))
            created = apply_review(review, decisions)['created_profiles']
            packet = json.loads(path.read_text())
            # Seed another recording condition so clearing this source's human
            # choices still permits automatic matching, without self-training.
            saved = json.loads(registry.path.read_text())
            for profile in saved['profiles'].values():
                for observation in profile['observations']:
                    observation['source_fingerprint'] = 'another-recording'
            atomic_write_json(registry.path, saved)
            atomic_write_json(decisions, {'format_version': 2, 'packet': path.name,
                'review_id': packet['review_id'], 'assignments': {'C': created['new:Child']}})
            result = apply_review(review, decisions)
            self.assertEqual(result['references_saved'], 3)
            payload = json.loads(outputs['json'].read_text())
            adult_match = next(m for m in payload['speaker_identity']['matches'] if m['local_speaker']=='A')
            self.assertEqual(adult_match['status'], 'matched')
            self.assertEqual(adult_match['speaker'], created['new:Adult A'])

    def test_complete_draft_can_remove_old_turn_correction(self):
        with tempfile.TemporaryDirectory() as directory:
            review, path, decisions, _, _, _, outputs = fixture(Path(directory))
            packet = json.loads(path.read_text())
            from transcribe_media_app.analysis import build_turns, normalize_segments
            turn_id = build_turns(normalize_segments(packet['local_result']))[0]['id']
            data = json.loads(decisions.read_text()); data['turn_overrides'] = {turn_id: 'new:Child'}
            atomic_write_json(decisions, data)
            apply_review(review, decisions)
            packet = json.loads(path.read_text())
            data = {**packet['applied_decisions'], 'review_id': packet['review_id'], 'packet': path.name,
                    'format_version': 2, 'turn_overrides': {}}
            atomic_write_json(decisions, data)
            apply_review(review, decisions)
            payload = json.loads(outputs['json'].read_text())
            adult_id = next(p['voice_id'] for p in payload['speaker_profiles'] if p['label'] == 'Adult A')
            self.assertEqual(payload['turns'][0]['speaker'], adult_id)

    def test_profile_rename_updates_other_transcripts_and_archive_stops_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            review, path, decisions, registry, _, samples, outputs = fixture(Path(directory))
            created = apply_review(review, decisions)['created_profiles']; voice = created['new:Adult A']
            second = {'json': review / 'second.json', 'txt': review / 'second.txt'}
            write_outputs(second, json.loads(outputs['json'].read_text()))
            manifest = ManifestStore(review / 'transcription_manifest.json')
            manifest.update('second.wav', {'outputs': {k: str(v) for k,v in second.items()}})
            packet = json.loads(path.read_text())
            atomic_write_json(decisions, {'packet': path.name, 'review_id': packet['review_id'],
                'profile_updates': {voice: {'label': 'Alex', 'role': 'adult', 'archived': True}}})
            apply_review(review, decisions)
            for output in (outputs['json'], second['json']):
                profile = next(p for p in json.loads(output.read_text())['speaker_profiles'] if p['voice_id'] == voice)
                self.assertEqual(profile['label'], 'Alex')
            report = VoiceRegistry(registry.path, reviewed=True, learn=False).identify(source_key='future',
                source_fingerprint='future', local_speakers=['A'], evidence={'A': samples['A']})
            self.assertTrue(report['matches'][0]['status'].startswith('unresolved'))
            self.assertEqual(registry.validate()['profile_count'], 3)

    def test_profile_only_edit_failure_rolls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            review, path, decisions, registry, _, _, outputs = fixture(Path(directory))
            voice = apply_review(review, decisions)['created_profiles']['new:Adult A']
            before = {p: p.read_bytes() for p in (registry.path, outputs['json'], path)}
            packet = json.loads(path.read_text())
            atomic_write_json(decisions, {'packet': path.name, 'review_id': packet['review_id'],
                'profile_updates': {voice: {'label': 'Alex', 'role': 'adult'}}})
            original_write = write_outputs
            count = 0
            def fail_later(*args):
                nonlocal count
                count += 1
                if count == 2: raise OSError('full disk')
                original_write(*args)
            with mock.patch('transcribe_media_app.review.write_outputs', side_effect=fail_later):
                with self.assertRaises(OSError): apply_review(review, decisions)
            for p, content in before.items(): self.assertEqual(p.read_bytes(), content)

    def test_review_command_regenerates_existing_pages_without_asr(self):
        with tempfile.TemporaryDirectory() as directory:
            review, path, *_ = fixture(Path(directory))
            path.with_suffix('.html').write_text('old UI')
            with mock.patch('transcribe_media_app.cli.WhisperXBackend.create', side_effect=AssertionError('ASR loaded')):
                review_index(review)
            page = path.with_suffix('.html').read_text()
            self.assertIn('Import saved review', page)
            self.assertIn('class ReviewDraft', page)
            self.assertIn('Needs attention', page)

    def test_wrong_recording_import_still_rejected_without_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            review, _, decisions, registry, *_ = fixture(Path(directory))
            original = registry.path.read_bytes()
            data = json.loads(decisions.read_text()); data['review_id'] = 'wrong'
            atomic_write_json(decisions, data)
            with self.assertRaisesRegex(RuntimeError, 'stale'):
                apply_review(review, decisions)
            self.assertEqual(registry.path.read_bytes(), original)


class ShortReferenceTests(unittest.TestCase):
    def profile(self, durations, vectors=None):
        vectors = vectors or [[1.,0.]] * len(durations)
        return {'centroid': [1.,0.], 'observations': [
            {'verified': True, 'source_fingerprint': 'session', 'observation_id': str(i),
             'start': i*3., 'clean_seconds': duration, 'embedding': vectors[i]}
            for i,duration in enumerate(durations)]}

    def test_many_consistent_short_clips_support_automatic_matching(self):
        profile = self.profile([1.6]*5)
        self.assertGreater(VoiceRegistry.profile_similarity(profile, [1.,0.], True), .99)

    def test_too_few_short_clips_or_too_little_speech_stays_untrained(self):
        for durations in ([2.]*4, [1.]*5, [.4]*30):
            self.assertEqual(VoiceRegistry.profile_similarity(self.profile(durations), [1.,0.], True), -1.)

    def test_short_clips_of_different_people_do_not_establish_a_match(self):
        profile = self.profile([1.6]*5, [[1.,0.]]*4+[[0.,1.]])
        self.assertEqual(VoiceRegistry.profile_similarity(profile, [1.,0.], True), -1.)

    def test_unverified_short_clips_never_count_as_support(self):
        profile = self.profile([1.6]*10)
        for obs in profile['observations']: obs['verified'] = False
        self.assertEqual(VoiceRegistry.profile_similarity(profile, [1.,0.], True), -1.)
