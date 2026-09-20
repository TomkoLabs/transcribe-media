"""Conservative vocal-emotion presentation and source-time annotation reuse."""
from __future__ import annotations

import copy
import math

EMOTION_POLICY_VERSION = '1.0'
EMOTIONS = {'angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised'}


def summarize_windows(tone):
    totals, duration = {}, 0.
    for window in tone.get('windows', []):
        weight = window['end'] - window['start']
        duration += weight
        for score in window['scores']:
            totals[score['label']] = totals.get(score['label'], 0.) + score['probability'] * weight
    if duration:
        tone['scores'] = sorted(({'label': k, 'probability': v / duration} for k, v in totals.items()),
                                key=lambda s: s['probability'], reverse=True)
        tone['classification_status'] = 'unclassified' if tone['scores'][0]['label'] == 'unknown' else 'classified'
        tone['temporal_variation'] = len({w['scores'][0]['label'] for w in tone['windows'] if w['scores']}) > 1


def strong_label(scores):
    ranked = sorted((s for s in scores or [] if isinstance(s, dict) and
                     isinstance(s.get('probability'), (int, float)) and math.isfinite(s['probability'])),
                    key=lambda s: s['probability'], reverse=True)
    if not ranked:
        return None
    top = ranked[0]
    runner = ranked[1]['probability'] if len(ranked) > 1 else 0.
    return top.get('label') if top.get('label') in EMOTIONS and .85 <= top['probability'] <= 1. and top['probability'] - runner >= .30 else None


def gate_tone(turn, profiles=()):
    tone = turn.get('tone')
    if not isinstance(tone, dict):
        return
    reasons, events = [], []
    profile = next((p for p in profiles if p['voice_id'] == turn.get('speaker')), {})
    if profile.get('role') != 'adult':
        reasons.append('adult_voice_not_confirmed')
    if tone.get('model') != 'emotion2vec/emotion2vec_plus_large':
        reasons.append('estimator_not_validated_for_display')
    overlap = [*turn.get('acoustic_overlap', []), *turn.get('overlaps', [])]
    for window in tone.get('windows', []):
        first, last = window['start'], window['end']
        label = strong_label(window.get('scores'))
        if (not window.get('audio_eligible') or last-first<2.5 or not label or
                label != strong_label(window.get('confirmation_scores')) or
                first < turn['start'] or last > turn['end']):
            continue
        if any(e['start'] < last and e['end'] > first for e in overlap):
            continue
        words = [w for w in turn.get('words', []) if isinstance(w.get('start'), (int,float))
                 and isinstance(w.get('end'), (int,float)) and w['start']<last and w['end']>first]
        if not words or any((w.get('timing') or {}).get('issues') for w in words):
            continue
        events.append({'start': first, 'end': last, 'label': label})
    if not events:
        reasons.append('no_clean_consistent_strong_emotion')
    labels = list(dict.fromkeys(e['label'] for e in events))
    label = labels[0] if len(labels)==1 else 'varied ('+' -> '.join(labels[:3])+')' if labels else None
    coverage = sum(e['end']-e['start'] for e in events) / max(.001, turn['end']-turn['start'])
    if label and coverage < .8:
        label += ' (part of this passage)'
    tone['display'] = {'policy': EMOTION_POLICY_VERSION, 'label': label if not reasons else None,
                       'events': events if not reasons else [], 'suppressed_reasons': reasons,
                       'score_kind': 'uncalibrated_model_score'}


def estimate_clean_tone(estimator, audio, turns):
    """Analyze contiguous clean spans, so a brief overlap does not erase a turn."""
    fragments, owners = [], []
    for turn in turns:
        turn['tone'] = None
        spans = [(turn['start'], turn['end'])]
        for event in [*turn.get('acoustic_overlap', []), *turn.get('overlaps', [])]:
            first, last = event['start']-.15, event['end']+.15
            spans = [piece for start,end in spans for piece in
                     ([(start,end)] if last<=start or first>=end else
                      ([(start,first)] if first>start else [])+([(last,end)] if last<end else []))]
        for first,last in spans:
            if last-first < 2.5:
                continue
            fragments.append({**turn, 'start': first, 'end': last, 'tone': None,
                              'words': [w for w in turn.get('words', []) if isinstance(w.get('start'),(int,float))
                                        and isinstance(w.get('end'),(int,float)) and w['start']<last and w['end']>first]})
            owners.append(turn)
    estimator.estimate(audio, fragments)
    for fragment, turn in zip(fragments, owners):
        tone = fragment.get('tone')
        if not tone:
            continue
        if tone.get('kind') == 'unavailable':
            turn['tone'] = tone
        elif not turn.get('tone'):
            turn['tone'] = copy.deepcopy(tone)
        elif turn['tone'].get('kind') != 'unavailable':
            turn['tone']['windows'].extend(copy.deepcopy(tone.get('windows', [])))
    for turn in turns:
        if (turn.get('tone') or {}).get('kind') == 'approximate_model_estimate':
            summarize_windows(turn['tone'])
    return fragments


def gate_turns(turns, profiles=()):
    for turn in turns:
        gate_tone(turn, profiles)


def preserve_tone(old_turns, new_turns, profiles=()):
    """Reuse only windows wholly within the new speaker turn; never stretch them."""
    windows = []
    for old in old_turns:
        tone = old.get('tone') or {}
        for window in tone.get('windows', []):
            windows.append((window, tone))
    for turn in new_turns:
        contained = [(w, t) for w, t in windows if turn['start'] <= w['start'] < w['end'] <= turn['end']]
        if not contained or len({t.get('model') for _, t in contained}) != 1:
            continue
        tone = copy.deepcopy(contained[0][1])
        tone['windows'] = [copy.deepcopy(w) for w, _ in contained]
        summarize_windows(tone)
        tone['reused_source_windows'] = True
        turn['tone'] = tone
    gate_turns(new_turns, profiles)
