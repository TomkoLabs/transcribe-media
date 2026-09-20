import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from test_quality import fixture
from transcribe_media_app.backends import Emotion2VecToneEstimator
from transcribe_media_app.emotion import gate_tone, preserve_tone, estimate_clean_tone
from transcribe_media_app.timing import annotate_word_timing, turn_timing, pending_review
from transcribe_media_app.review import apply_review, refresh_reviews, review_data, apply_turn_choices, _resolve_choices
from transcribe_media_app.context import refresh_context
from transcribe_media_app.schema import SAMPLE_RATE
from transcribe_media_app.storage import atomic_write_json

MODEL = 'emotion2vec/emotion2vec_plus_large'
ADULTS = [{'voice_id': 'VOICE_0001', 'role': 'adult', 'label': 'Adult A'}]


def spoken_turn(start=0., end=6.):
    return {'id':'turn-0','start':start,'end':end,'speaker':'VOICE_0001','text':'One two three four five six.',
            'words':[{'start':i+.1,'end':i+.9,'text':'word','speaker':'VOICE_0001'} for i in np.arange(start,end,1.)]}


def scored_tone(start=0., end=6., label='angry'):
    scores = [{'label':label,'probability':.96},{'label':'neutral','probability':.04}]
    return {'kind':'approximate_model_estimate','model':MODEL,'model_version':'test','scores':scores,
            'windows':[{'start':start,'end':end,'scores':scores,'confirmation_scores':copy.deepcopy(scores),'audio_eligible':True}]}


class TimingTests(unittest.TestCase):
    def test_dual_timing_disagreement_keeps_words_and_timestamps_and_source_anchor(self):
        result={'segments':[{'words':[{'word':'Hello!','start':8.,'end':8.4,'score':.9}]}],
                'asr_diagnostics':[{'start':3.,'end':5.,'text':'Hello!',
                    'asr_words':[{'word':' Hello','start':3.2,'end':3.6,'probability':.95}]}]}
        annotate_word_timing(result)
        word=result['segments'][0]['words'][0]
        self.assertEqual((word['word'],word['start'],word['end']),('Hello!',8.,8.4))
        self.assertIn('whisper_alignment_disagreement',word['timing']['issues'])
        timing=turn_timing({'start':8.,'end':8.4,'words':[word]})
        self.assertEqual(timing['source_context']['start'],3.)

    def test_missing_alignment_is_never_fabricated(self):
        result={'segments':[{'words':[{'word':'2014','start':None,'end':None}]}]}
        annotate_word_timing(result)
        word=result['segments'][0]['words'][0]
        self.assertIsNone(word['start']); self.assertEqual(word['timing']['status'],'uncertain')

    def test_untimed_words_accept_only_explicit_whole_soundbite_choices_without_guessed_times(self):
        local={'segments':[{'start':2.,'end':5.,'text':'In 2014','speaker':'A',
                          'words':[{'word':'In','start':2.,'end':2.2}, {'word':'2014'}]}]}
        packet={'local_result':local}
        unresolved=copy.deepcopy(local)
        apply_turn_choices(unresolved,_resolve_choices(packet,{'assignments':{'A':'VOICE_0001'}}),'test')
        self.assertNotIn('speaker_identity',unresolved['segments'][0]['words'][1])
        reviewed=copy.deepcopy(local)
        choices=_resolve_choices(packet,{'turn_overrides':{'turn-000000':'VOICE_0001'}})
        apply_turn_choices(reviewed,choices,'test')
        word=reviewed['segments'][0]['words'][1]
        self.assertEqual(word['speaker'],'VOICE_0001')
        self.assertNotIn('start',word);self.assertNotIn('end',word)
        self.assertEqual(word['speaker_identity']['status'],'human_verified')

    def test_old_cache_segment_anchors_are_available_without_new_asr(self):
        result={'segments':[{'words':[{'word':'Hello','start':3.2,'end':3.6,'score':.9}]}],
                'asr_diagnostics':[{'start':3.,'end':5.,'text':'Hello there.'}]}
        annotate_word_timing(result)
        word=result['segments'][0]['words'][0]
        self.assertEqual(word['timing']['source_segment']['text'],'Hello there.')
        self.assertIsNone(word['timing']['whisper_word'])

    def test_timing_notes_do_not_revoke_a_confident_identity(self):
        turn=spoken_turn();turn['words'][0]['timing']={'issues':['weak_alignment']}
        payload={'turns':[turn], 'speaker_identity':{'matches':[{'local_speaker':'A','speaker':'VOICE_0001','status':'matched'}]}}
        self.assertEqual(pending_review(payload)['pending'],[])
        self.assertIn('check_word_timing',turn['review_reasons'])
        self.assertEqual(turn['identity_review_reasons'],[])
        for word in turn['words']:word['speaker_identity']={'status':'human_verified'}
        self.assertEqual(pending_review(payload)['pending'],[])

    def test_review_suggestions_use_verified_references_without_registry_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            review,path,decisions,registry,*_=fixture(Path(directory))
            ids=apply_review(review,decisions)['created_profiles']
            before=registry.path.read_bytes()
            data=review_data(path,json.loads(path.read_text()))
            self.assertEqual(data['windows']['A'][0]['suggested_speaker'],ids['new:Adult A'])
            self.assertNotIn('embedding',data['windows']['A'][0])
            self.assertEqual(registry.path.read_bytes(),before)

    def test_voice_outlier_stays_in_backend_queue_despite_a_strong_group_match(self):
        turn=spoken_turn()
        payload={'turns':[turn], 'speaker_identity':{'matches':[{'local_speaker':'A','speaker':'VOICE_0001','status':'matched'}]}}
        evidence={'A':{'windows':[{'start':2.,'end':3.,'retained':False}]}}
        self.assertEqual(pending_review(payload,evidence)['pending'],['A'])
        self.assertIn('voice_reference_outlier',turn['review_reasons'])
        for word in turn['words']:word['speaker_identity']={'status':'human_verified'}
        self.assertEqual(pending_review(payload,evidence)['pending'],[])


class EmotionTests(unittest.TestCase):
    def test_only_confirmed_clean_strong_adult_vocal_emotion_is_shown(self):
        turn=spoken_turn();turn['tone']=scored_tone()
        gate_tone(turn,ADULTS);self.assertEqual(turn['tone']['display']['label'],'angry')
        for change in ('child','quiet','disagreement','neutral','weak_timing','overlap'):
            sample=copy.deepcopy(turn);profiles=copy.deepcopy(ADULTS);window=sample['tone']['windows'][0]
            if change=='child':profiles[0]['role']='child'
            if change=='quiet':window['audio_eligible']=False
            if change=='disagreement':window['confirmation_scores'][0]['label']='sad'
            if change=='neutral':window['scores'][0]['label']='neutral'
            if change=='weak_timing':sample['words'][0]['timing']={'issues':['weak_alignment']}
            if change=='overlap':sample['acoustic_overlap']=[{'start':2.,'end':3.}]
            gate_tone(sample,profiles)
            with self.subTest(change=change):self.assertIsNone(sample['tone']['display']['label'])

    def test_clean_parts_of_an_overlapped_turn_are_analyzed_without_joining_audio(self):
        turn=spoken_turn(end=12.);turn['acoustic_overlap']=[{'start':5.,'end':7.}]
        estimator=mock.Mock()
        def estimate(audio,turns):
            for i,fragment in enumerate(turns):
                fragment['tone']=scored_tone(fragment['start'],fragment['end'],'angry' if i==0 else 'sad')
        estimator.estimate.side_effect=estimate
        fragments=estimate_clean_tone(estimator,np.ones(12*SAMPLE_RATE)*.1,[turn])
        self.assertEqual([(t['start'],t['end']) for t in fragments],[(0.,4.85),(7.15,12.)])
        gate_tone(turn,ADULTS)
        self.assertIn('angry',turn['tone']['display']['label'])
        self.assertIn('sad',turn['tone']['display']['label'])
        self.assertTrue(turn['tone']['temporal_variation'])
        self.assertAlmostEqual(turn['tone']['scores'][0]['probability'],.48)
        for event in turn['tone']['display']['events']:
            self.assertTrue(event['end']<=5. or event['start']>=7.)

    def test_split_speaker_turns_only_reuse_wholly_contained_emotion_windows(self):
        original=spoken_turn(end=12.);original['tone']=scored_tone(0.,6.)
        original['tone']['windows'] += scored_tone(6.,12.,'sad')['windows']
        new=[spoken_turn(0.,4.),spoken_turn(4.,12.)]
        preserve_tone([original],new,ADULTS)
        self.assertNotIn('tone',new[0])
        self.assertEqual(new[1]['tone']['windows'][0]['start'],6.)
        self.assertIn('sad',new[1]['tone']['display']['label'])

    def test_estimator_checks_a_second_audio_view_and_measures_audio_eligibility(self):
        estimator=Emotion2VecToneEstimator.__new__(Emotion2VecToneEstimator)
        estimator.version='synthetic';estimator.classifier=mock.Mock()
        estimator.classifier.generate.return_value=[{'labels':['angry','neutral'],'scores':[.96,.04]}]
        turn=spoken_turn()
        audio=np.sin(np.arange(6*SAMPLE_RATE)*.1).astype(np.float32)*.1
        estimator.estimate(audio,[turn]);gate_tone(turn,ADULTS)
        self.assertEqual(estimator.classifier.generate.call_count,2)
        self.assertEqual(turn['tone']['display']['label'],'angry')
        self.assertLess(len(estimator.classifier.generate.call_args.kwargs['input']),len(audio))

    def test_context_refresh_preserves_decisions_and_works_without_asr(self):
        with tempfile.TemporaryDirectory() as directory:
            review,path,decisions,registry,*_=fixture(Path(directory))
            apply_review(review,decisions)
            packet=json.loads(path.read_text());before=copy.deepcopy(packet);registry_bytes=registry.path.read_bytes()
            estimator=mock.Mock();estimator.name=MODEL;estimator.version='test'
            def estimate(audio,turns):
                for turn in turns:turn['tone']=scored_tone(turn['start'],turn['end'])
            estimator.estimate.side_effect=estimate
            result=refresh_context(review,estimator,None)
            self.assertEqual(result['recordings_updated'],1)
            after=json.loads(path.read_text())
            for key in ('review_id','local_result','applied_decisions','evidence'):
                self.assertEqual(after[key],before[key])
            self.assertEqual(registry.path.read_bytes(),registry_bytes)
            self.assertEqual([(t['text'],t['speaker'],t['start'],t['end']) for t in after['payload']['turns']],
                             [(t['text'],t['speaker'],t['start'],t['end']) for t in before['payload']['turns']])
            refresh_reviews(review)
            self.assertTrue(any(t.get('tone',{}).get('windows') for t in json.loads(path.read_text())['payload']['turns']))

    def test_failed_context_inference_preserves_prior_context_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            review,path,decisions,registry,*_=fixture(Path(directory));apply_review(review,decisions)
            packet=json.loads(path.read_text());paths=[path,registry.path,*map(Path,packet['outputs'].values())]
            before={p:p.read_bytes() for p in paths}
            estimator=mock.Mock();estimator.estimate.side_effect=RuntimeError('failed inference')
            with self.assertRaisesRegex(RuntimeError,'failed inference'):refresh_context(review,estimator,None)
            for path,contents in before.items():self.assertEqual(path.read_bytes(),contents)
