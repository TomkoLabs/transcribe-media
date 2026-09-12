import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_quality import fixture
from transcribe_media_app.review import (apply_review, batch_digest, merge_project_profiles,
    refresh_reviews, review_index, save_review)
from transcribe_media_app.storage import atomic_write_json, source_fingerprint, ManifestStore
from transcribe_media_app.renderers import write_outputs
from transcribe_media_app import cli


def batch_fixture(root, count=2):
    review, first, single, registry, local, samples, first_outputs = fixture(root)
    seed = json.loads(first.read_text())
    packets = [first]
    outputs = [first_outputs]
    for index in range(1, count):
        source = root / f'recording-{index}.wav'
        source.write_bytes(f'synthetic source {index}'.encode())
        fingerprint = source_fingerprint(source)
        payload = copy.deepcopy(seed['payload'])
        payload['source'] = {'relative_path': source.name, 'fingerprint': fingerprint}
        files = {'json': review / f'recording-{index}.json', 'txt': root / 'Transcribed' / f'recording-{index}.txt'}
        path = save_review(review, source, source.name, fingerprint, local, samples, payload, files, registry, [])
        ManifestStore(review / 'transcription_manifest.json').update(source.name,
            {'source_fingerprint': fingerprint, 'outputs': {key: str(value) for key, value in files.items()}})
        packets.append(path); outputs.append(files)
    original = json.loads(single.read_text())
    reviews = []
    for path in packets:
        packet = json.loads(path.read_text())
        reviews.append({'format_version': 2, 'packet': path.name, 'review_id': packet['review_id'],
                        'assignments': copy.deepcopy(original['assignments'])})
    batch = {'kind': 'speaker_review_batch', 'format_version': 3, 'batch_id': batch_digest(reviews),
             'new_profiles': original['new_profiles'], 'profile_updates': {}, 'reviews': reviews}
    target = root / 'batch.decisions.json'
    atomic_write_json(target, batch)
    return review, packets, outputs, registry, batch, target


class BatchReviewTests(unittest.TestCase):
    def test_shared_people_created_once_and_named_in_all_transcripts(self):
        with tempfile.TemporaryDirectory() as directory:
            review, packets, outputs, registry, _, target = batch_fixture(Path(directory))
            result = apply_review(review, target)
            self.assertEqual(result['pending'], 0)
            self.assertEqual(len(result['created_profiles']), 3)
            self.assertEqual(registry.validate()['profile_count'], 3)
            self.assertTrue(all(p['verified_sessions'] == 2 for p in registry.profiles_for_review()))
            for files in outputs:
                self.assertIn('Adult A [VOICE_', files['txt'].read_text())
                self.assertNotIn('DRAFT:', files['txt'].read_text())
            self.assertTrue(apply_review(review, target)['already_applied'])
            self.assertEqual(registry.validate()['profile_count'], 3)
            self.assertIn('speaker_review_batch', review_index(review).read_text())

    def test_final_rematch_uses_later_reviewed_recording_and_includes_other_cached_files(self):
        with tempfile.TemporaryDirectory() as directory:
            review, _, outputs, _, batch, target = batch_fixture(Path(directory), 3)
            batch['reviews'] = batch['reviews'][:2]
            batch['reviews'][0]['assignments'] = {}
            batch['batch_id'] = batch_digest(batch['reviews'])
            atomic_write_json(target, batch)
            result = apply_review(review, target)
            self.assertEqual(len(result['recordings_refreshed']), 3)
            self.assertEqual(result['pending'], 0)
            for i in (0, 2):
                payload = json.loads(outputs[i]['json'].read_text())
                self.assertTrue(all(m['status'] == 'matched' for m in payload['speaker_identity']['matches']))
            self.assertEqual(result['references_saved'], 9)  # Automatic matches did not verify themselves.

    def test_revised_batch_and_reverting_snapshot_keep_canonical_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            review, _, _, registry, batch, target = batch_fixture(Path(directory))
            ids = apply_review(review, target)['created_profiles']
            revised = copy.deepcopy(batch)
            for entry in revised['reviews']: entry['exclude_windows'] = ['A:0', 'A:1', 'A:2']
            atomic_write_json(target, revised)
            self.assertEqual(apply_review(review, target)['created_profiles'], {})
            self.assertEqual(next(p for p in registry.profiles_for_review() if p['voice_id']==ids['new:Adult A'])['verified_windows'], 0)
            atomic_write_json(target, batch)
            apply_review(review, target)
            self.assertEqual(registry.validate()['profile_count'], 3)
            self.assertEqual(next(p for p in registry.profiles_for_review() if p['voice_id']==ids['new:Adult A'])['verified_windows'], 6)

    def test_failed_later_recording_rolls_back_all_outputs_and_enrollment(self):
        with tempfile.TemporaryDirectory() as directory:
            review, packets, outputs, registry, _, target = batch_fixture(Path(directory))
            before = {path: path.read_bytes() for path in [registry.path, *packets]}
            with mock.patch('transcribe_media_app.review.refresh_reviews', side_effect=OSError('failed final write')):
                with self.assertRaises(OSError): apply_review(review, target)
            for path, content in before.items(): self.assertEqual(path.read_bytes(), content)
            self.assertFalse((review / 'speaker-review-batches.json').exists())
            self.assertFalse(any(files['txt'].exists() for files in outputs))
            self.assertFalse((review / '.state-transaction.json').exists())

    def test_stale_second_recording_fails_before_creating_people(self):
        with tempfile.TemporaryDirectory() as directory:
            review, _, _, registry, batch, target = batch_fixture(Path(directory))
            batch['reviews'][1]['review_id'] = 'stale'
            batch['batch_id'] = batch_digest(batch['reviews']); atomic_write_json(target, batch)
            before = registry.path.read_bytes()
            with self.assertRaisesRegex(RuntimeError, 'stale'): apply_review(review, target)
            self.assertEqual(registry.path.read_bytes(), before)

    def test_explicit_unknown_excludes_training_and_survives_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            review, _, outputs, registry, batch, target = batch_fixture(Path(directory))
            for entry in batch['reviews']: entry['assignments']['C'] = 'ignore'
            atomic_write_json(target, batch)
            self.assertEqual(apply_review(review, target)['pending'], 0)
            self.assertEqual(registry.validate()['profile_count'], 2)
            refresh_reviews(review)
            for files in outputs:
                self.assertIn('UNKNOWN', files['txt'].read_text())
                self.assertIn('Child speaks here', files['txt'].read_text())
                self.assertNotIn('DRAFT:', files['txt'].read_text())
                payload=json.loads(files['json'].read_text())
                self.assertEqual(payload['turns'][-1]['speaker'], 'SPEAKER_UNKNOWN')

    def test_all_unknown_batch_needs_no_voice_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            review, _, _, registry, batch, target = batch_fixture(Path(directory))
            for entry in batch['reviews']: entry['assignments'] = dict.fromkeys('ABC', 'ignore')
            atomic_write_json(target, batch)
            self.assertEqual(apply_review(review, target)['pending'], 0)
            self.assertEqual(registry.validate()['profile_count'], 0)

    def test_relative_decisions_folder_resolves_from_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            review, _, _, _, batch, _ = batch_fixture(root)
            target=root/'speaker-decisions'/'batch.decisions.json'; atomic_write_json(target, batch)
            with mock.patch.dict('os.environ', {'TRANSCRIBE_MEDIA_ROOT': str(root)}):
                self.assertEqual(cli.main(['--apply-speaker-review', 'speaker-decisions/batch.decisions.json']), 0)

    def test_revised_batch_after_profile_merge_resolves_the_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            review, _, _, registry, batch, target = batch_fixture(Path(directory))
            created=apply_review(review,target)['created_profiles']
            merge_project_profiles(review,created['new:Adult B'],created['new:Adult A'])
            batch['reviews'][0]['exclude_windows']=['A:0'];atomic_write_json(target,batch)
            self.assertEqual(apply_review(review,target)['created_profiles'],{})
            self.assertEqual(registry.validate()['profile_count'],2)

    def test_subtitles_show_labels_without_changing_json_identity_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            review, _, outputs, _, _, target=batch_fixture(root)
            voice=apply_review(review,target)['created_profiles']['new:Adult A']
            payload=json.loads(outputs[0]['json'].read_text())
            self.assertEqual(payload['turns'][0]['speaker'],voice)
            files={'srt':root/'sample.srt','vtt':root/'sample.vtt','detailed_txt':root/'detailed.txt'}
            write_outputs(files,payload)
            for path in files.values():self.assertIn(f'Adult A [{voice}]',path.read_text())


if __name__ == '__main__': unittest.main()
