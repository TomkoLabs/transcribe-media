"""A learned person can recur in clusters without erasing genuine exceptions."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_quality import fixture, evidence
from transcribe_media_app.analysis import build_turns, normalize_segments
from transcribe_media_app.evidence import identity_evidence_summary
from transcribe_media_app.speakers import VoiceRegistry, apply_speaker_identities
from transcribe_media_app.timing import pending_review


def legacy_disputed_sample(vector, start=0.):
    sample=evidence(vector,start)
    sample['windows']=[{**sample['windows'][0], 'start':start+i*4, 'end':start+(i+1)*4,
                       'model_disagreement':i>=4, 'retained':i<4} for i in range(6)]
    sample.update(suspected_mixed_speakers=True,clean_seconds=16.,window_count=4)
    return sample


class FocusedMatchingTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)
        review,_,decisions,registry,*_=fixture(self.root)
        from transcribe_media_app.review import apply_review
        self.ids=apply_review(review,decisions)['created_profiles']
        self.registry=VoiceRegistry(registry.path,reviewed=True,learn=False)

    def identify(self,samples,**kwargs):
        return self.registry.identify(source_key='new.wav',source_fingerprint='new-recording',
                                      local_speakers=samples,evidence=samples,**kwargs)

    def test_older_mixed_flag_from_diarizer_disagreement_does_not_block_verified_consensus(self):
        sample=legacy_disputed_sample([1.,0.,0.])
        before=copy.deepcopy(sample);registry_before=self.registry.path.read_bytes()
        match=self.identify({'A':sample})['matches'][0]
        self.assertEqual(match['status'],'matched')
        self.assertEqual(match['speaker'],self.ids['new:Adult A'])
        self.assertEqual(match['matching_diagnostics']['all_sample_support'],6)
        self.assertEqual(match['matching_diagnostics']['blockers'],[])
        self.assertEqual(sample,before)
        self.assertEqual(self.registry.path.read_bytes(),registry_before)

    def test_disputed_samples_of_another_voice_still_block_automatic_group_assignment(self):
        for vector in ([0.,0.,1.],[-1.,0.,0.]):
            sample=legacy_disputed_sample([1.,0.,0.])
            for window in sample['windows'][4:]:window['embedding']=vector
            match=self.identify({'A':sample})['matches'][0]
            self.assertTrue(match['status'].startswith('unresolved'))
            self.assertIn('unresolved_disputed_voice_evidence',match['matching_diagnostics']['blockers'])

    def test_actual_acoustic_mix_and_opaque_legacy_flags_still_block(self):
        sample=legacy_disputed_sample([1.,0.,0.])
        for window in sample['windows'][4:]:
            window.update(model_disagreement=False,embedding=[0.,0.,1.])
        self.assertTrue(identity_evidence_summary(sample)['mixed_voice_evidence'])
        match=self.identify({'A':sample})['matches'][0]
        self.assertIn('mixed_voice_evidence',match['matching_diagnostics']['blockers'])
        opaque=evidence([1.,0.,0.]);opaque['suspected_mixed_speakers']=True
        self.assertTrue(self.identify({'A':opaque})['matches'][0]['status'].startswith('unresolved'))

    def test_one_known_person_can_match_multiple_nonoverlapping_clusters(self):
        samples={'A':legacy_disputed_sample([1.,0.,0.]),'B':legacy_disputed_sample([0.,1.,0.],30.),
                 'C':legacy_disputed_sample([0.,1.,0.],60.)}
        report=self.identify(samples)
        self.assertTrue(all(m['status']=='matched' for m in report['matches']))
        self.assertEqual(report['active_speaker_count'],2)
        self.assertEqual(report['speaker_groups_after_reconciliation'],3)
        self.assertEqual(report['merged_local_clusters'],0)
        self.assertEqual(report['matches'][1]['speaker'],report['matches'][2]['speaker'])
        conflicts=self.identify(samples,incompatible_pairs=[frozenset(('B','C'))])
        self.assertEqual(sum(m['status']=='matched' for m in conflicts['matches']),2)
        self.assertIn('overlap_with_same_profile',next(m for m in conflicts['matches'] if m['status']!='matched')['matching_diagnostics']['blockers'])

    def test_only_local_exceptions_queue_when_known_people_recur_in_clusters(self):
        samples={'A':legacy_disputed_sample([1.,0.,0.]),'B':legacy_disputed_sample([0.,1.,0.],30.),
                 'C':legacy_disputed_sample([0.,1.,0.],60.)}
        result={'segments':[]}
        for local,item in samples.items():
            for window in item['windows']:
                start=window['start']+.2
                words=[{'word':text,'start':start+i*.4,'end':start+i*.4+.3,'speaker':local,
                        'timing':{'issues':['weak_alignment']}} for i,text in enumerate(('Some','real','words'))]
                result['segments'].append({'speaker':local,'start':start,'end':start+1.1,
                                           'text':'Some real words','words':words})
        apply_speaker_identities(result,self.identify(samples))
        result['turns']=build_turns(normalize_segments(result))
        queued=pending_review(result,samples,self.registry)
        self.assertEqual(len(result['turns']),18)
        self.assertEqual(len(queued['pending_turns']),6)  # Disputed intervals, not every timing note.
        self.assertTrue(all(t['review_timing']['status']=='uncertain' for t in result['turns']))
        for turn in result['turns']:
            for word in turn['words']:word['speaker_identity']={'status':'human_verified'}
        self.assertEqual(pending_review(result,samples,self.registry)['pending'],[])

    def test_known_voice_roster_is_respected_by_batched_matching(self):
        self.registry=VoiceRegistry(self.registry.path,reviewed=True,learn=False,known_voices=[self.ids['new:Adult A']])
        match=self.identify({'B':legacy_disputed_sample([0.,1.,0.])})['matches'][0]
        self.assertTrue(match['status'].startswith('unresolved'))
        self.assertNotEqual(match['speaker'],self.ids['new:Adult B'])

    def test_unidentified_speech_stays_pending_until_explicitly_excluded(self):
        payload={'turns':[{'id':'unknown','start':0.,'end':2.,'speaker':'SPEAKER_UNKNOWN','words':[
            {'text':'Hello','start':.2,'end':.6,'speaker':'SPEAKER_UNKNOWN'}]}]}
        self.assertEqual(pending_review(payload)['pending'],['SPEAKER_UNKNOWN'])
        payload['turns'][0]['words'][0]['speaker_identity']={'status':'human_excluded'}
        self.assertEqual(pending_review(payload)['pending'],[])

    def test_secondary_labels_for_the_same_verified_person_are_not_identity_conflicts(self):
        samples={'A':legacy_disputed_sample([1.,0.,0.]),'A2':legacy_disputed_sample([1.,0.,0.],30.)}
        report=self.identify(samples)
        result={'segments':[{'speaker':'A','start':16.2,'end':17.4,'text':'The same adult',
            'words':[{'word':text,'speaker':'A','sortformer_speaker':'A2','start':16.2+i*.4,'end':16.5+i*.4}
                     for i,text in enumerate(('The','same','adult'))]}]}
        apply_speaker_identities(result,report)
        result['turns']=build_turns(normalize_segments(result))
        result['turns'][0]['speaker_attribution']={'status':'uncertain','reasons':['sortformer_majority_disagrees']}
        before=copy.deepcopy(samples)
        self.assertEqual(pending_review(result,samples)['pending'],[])
        self.assertEqual(samples,before)
        result['segments'][0]['words'][0]['sortformer_ambiguous']=True
        result['turns'][0]['words'][0]['sortformer_ambiguous']=True
        self.assertIn('A',pending_review(result,samples)['pending'])
        result['segments'][0]['words'][0]['sortformer_ambiguous']=False
        result['turns'][0]['words'][0]['sortformer_ambiguous']=False
        next(m for m in report['matches'] if m['local_speaker']=='A2')['speaker']=self.ids['new:Child']
        self.assertIn('A',pending_review(result,samples)['pending'])

    def test_refresh_preserves_existing_decisions_review_ids_and_reference_library(self):
        from transcribe_media_app.review import refresh_reviews, review_index
        review=self.root/'Review'
        path=next((review/'speaker-reviews').glob('*.json'))
        packet=json.loads(path.read_text());registry=self.registry.path.read_bytes()
        refreshed=refresh_reviews(review)
        after=json.loads(path.read_text())
        for key in ('review_id','local_result','evidence','applied_decisions','new_profile_ids'):
            self.assertEqual(after[key],packet[key])
        self.assertEqual(self.registry.path.read_bytes(),registry)
        self.assertEqual(refreshed['recordings'][0]['pending'],0)
        self.assertIn('identity_review_reasons',review_index(review).read_text())

    def test_batched_reference_scores_match_original_consensus_calculation(self):
        rng=np.random.default_rng(19)
        profiles={}
        for person in range(3):
            observations=[]
            for session in range(3):
                for index in range(6):
                    vector=rng.normal(size=192);vector/=np.linalg.norm(vector)
                    observations.append({'source_fingerprint':str(session),'verified':True,
                                         'clean_seconds':4.,'embedding':vector.tolist()})
            profiles[str(person)]={'observations':observations}
        profiles['archived']={**profiles['0'],'archived':True}
        queries=rng.normal(size=(11,192)).tolist()
        ranked=VoiceRegistry._reviewed_rankings(profiles,queries)
        for query in queries:
            actual=dict(ranked[tuple(query)])
            for key,profile in profiles.items():
                self.assertAlmostEqual(actual[key],VoiceRegistry.profile_similarity(profile,query,True),places=12)
