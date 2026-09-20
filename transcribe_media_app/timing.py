"""Auditable word/audio links. Timing doubts never rewrite recognized words."""
from __future__ import annotations

import difflib
import math
import re

TIMING_VERSION = '1.0'


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def token(text):
    return ''.join(re.findall(r'\w+', str(text).casefold()))


def annotate_word_timing(result):
    """Compare CTC alignment with independent Whisper times when available.

    Older caches keep their original segment anchors. No guessed timestamp is
    substituted for an unaligned word, and the source-bound review IDs stay valid.
    """
    diagnostics = result.get('asr_diagnostics', [])
    source_words = [(word, segment) for segment in diagnostics for word in segment.get('asr_words', [])]
    words = [word for segment in result.get('segments', []) for word in segment.get('words', [])]
    pairs = {}
    matcher = difflib.SequenceMatcher(None, [token(w.get('word', w.get('text', ''))) for w in words],
                                     [token(w.get('word', '')) for w, _ in source_words], autojunk=False)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            pairs[block.a + offset] = source_words[block.b + offset]
    for index, word in enumerate(words):
        start, end = word.get('start'), word.get('end')
        issues = []
        valid = finite(start) and finite(end) and 0 <= start < end
        if not valid:
            issues.append('missing_or_zero_word_timing')
        elif end - start > max(1.5, len(token(word.get('word', word.get('text', '')))) * .18):
            issues.append('stretched_word_timing')
        score = word.get('score', word.get('confidence'))
        if finite(score) and score < .35:
            issues.append('weak_alignment')
        original, segment = pairs.get(index, (None, None))
        if segment is None and valid:
            candidates = [s for s in diagnostics if finite(s.get('start')) and finite(s.get('end'))
                          and s['start'] < end and s['end'] > start]
            if candidates:
                segment = max(candidates, key=lambda s: min(end, s['end']) - max(start, s['start']))
        if original and valid and finite(original.get('start')) and finite(original.get('end')):
            if max(abs(start - original['start']), abs(end - original['end'])) > .75:
                issues.append('whisper_alignment_disagreement')
        if segment and ((finite(segment.get('avg_logprob')) and segment['avg_logprob'] < -1.)
                        or (finite(segment.get('no_speech_prob')) and segment['no_speech_prob'] > .6)):
            issues.append('weak_asr_support')
        word['timing'] = {'version': TIMING_VERSION, 'status': 'uncertain' if issues else 'aligned',
                          'issues': issues, 'alignment_score': score,
                          'whisper_word': {k: original[k] for k in ('start', 'end', 'probability') if k in original} if original else None,
                          'source_segment': {k: segment[k] for k in ('start', 'end', 'text')} if segment else None}
    return result


def turn_timing(turn, diagnostics=()):
    words = turn.get('words', [])
    issues = sorted({issue for word in words for issue in (word.get('timing') or {}).get('issues', [])})
    if not words or any(not finite(w.get('start')) or not finite(w.get('end')) for w in words):
        issues.append('incomplete_word_timing')
    anchors = [w['timing']['source_segment'] for w in words if (w.get('timing') or {}).get('source_segment')]
    if not anchors:
        anchors = [s for s in diagnostics if finite(s.get('start')) and finite(s.get('end'))
                   and s['start'] < turn['end'] and s['end'] > turn['start']]
    # Prefer the actual recognizer's sentence anchor, rather than widening an
    # unreliable aligned word forever. Multiple sentences remain bounded.
    context = None
    if anchors:
        start, end = min(s['start'] for s in anchors), max(s['end'] for s in anchors)
        if 0 <= start < end and end - start <= 90:
            context = {'start': start, 'end': end,
                       'text': ' '.join(dict.fromkeys(s.get('text', '').strip() for s in anchors))}
    return {'version': TIMING_VERSION, 'status': 'uncertain' if issues else 'aligned',
            'issues': sorted(set(issues)), 'source_context': context}


def review_reasons(turn, windows=()):
    reasons = list((turn.get('speaker_attribution') or {}).get('reasons', []))
    # Identity is handled separately by the matcher; these are local exceptions.
    reasons = [r for r in reasons if r != 'unresolved_voice_identity']
    if turn.get('speaker') == 'SPEAKER_UNKNOWN' and 'unknown_speaker' not in reasons:
        reasons.append('unknown_speaker')
    if (turn.get('speaker_attribution') or {}).get('status') == 'uncertain' and not reasons:
        if not (turn.get('speaker_attribution') or {}).get('reasons'):
            reasons.append('speaker_timing_disputed')
    if turn.get('acoustic_overlap') or turn.get('overlaps'):
        reasons.append('overlapping_voices')
    if (turn.get('review_timing') or {}).get('status') == 'uncertain':
        reasons.append('check_word_timing')
    if any(w.get('retained') is False and not w.get('model_disagreement') and w['start'] < turn['end'] and w['end'] > turn['start'] for w in windows):
        reasons.append('voice_reference_outlier')
    if any(w.get('model_disagreement') and not w.get('identity_agreement') and w['start'] < turn['end'] and w['end'] > turn['start'] for w in windows):
        reasons.append('diarizer_disagreement')
    return sorted(set(reasons))


def matched_aliases(matches):
    return {m['local_speaker']: m['speaker'] for m in matches if m['status'] == 'matched'}


def identity_windows(evidence, result, matches):
    """Resolve label-only disagreements after independent known-person matches.

    This changes review hints, never the immutable samples or training eligibility.
    """
    aliases = matched_aliases(matches)
    words = [w for s in result.get('segments', []) for w in s.get('words', [])
             if finite(w.get('start')) and finite(w.get('end'))]
    output = {}
    for local, item in evidence.items():
        output[local] = []
        for window in item.get('windows', []):
            relevant = [w for w in words if (w.get('local_speaker') or w.get('speaker')) == local
                        and w['start'] < window['end'] and w['end'] > window['start']]
            agreed = bool(window.get('model_disagreement') and relevant and aliases.get(local)) and all(
                not w.get('sortformer_ambiguous') and aliases.get(w.get('sortformer_speaker')) == aliases[local]
                for w in relevant)
            output[local].append({**window, 'identity_agreement': agreed})
    return output


def voice_hints(turn, windows, matched_speakers, matches=()):
    relevant = [w for w in windows if w['start'] < turn['end'] and w['end'] > turn['start']]
    suggestions = {w.get('suggested_speaker') for w in relevant} - {None}
    reasons = review_reasons(turn, relevant)
    aliases = matched_aliases(matches)
    differences = [w for w in turn.get('words', []) if w.get('sortformer_speaker')
                   and w['sortformer_speaker'] != w.get('local_speaker', turn.get('speaker'))]
    if differences and all(not w.get('sortformer_ambiguous') and aliases.get(w.get('local_speaker'))
                           and aliases.get(w['sortformer_speaker']) == aliases[w['local_speaker']] for w in differences):
        reasons = [r for r in reasons if r != 'sortformer_majority_disagrees']
    if len(suggestions) > 1 or (matched_speakers and suggestions - set(matched_speakers)):
        reasons.append('conflicting_voice_references')
    return {'review_reasons': reasons, 'identity_review_reasons': [r for r in reasons if r != 'check_word_timing'],
            'suggested_speakers': sorted(suggestions)}


def pending_review(payload, evidence=None, registry=None):
    matches = (payload.get('speaker_identity') or {}).get('matches', [])
    pending = {m['local_speaker'] for m in matches if m['status'].startswith('unresolved')}
    pending_turns = []
    candidates = registry.review_window_candidates(evidence) if registry and evidence else {}
    annotated_windows = identity_windows(evidence or {}, payload, matches)
    for turn in payload.get('turns', []):
        words = turn.get('words') or [turn]
        reviewed = all((w.get('speaker_identity') or {}).get('status') in ('human_verified', 'human_excluded') for w in words)
        turn['review_timing'] = turn_timing(turn, payload.get('asr_diagnostics', []))
        locals_ = {w.get('local_speaker') for w in words if w.get('local_speaker')}
        if not locals_:
            locals_ = {m['local_speaker'] for m in matches if m['speaker'] == turn.get('speaker')}
        if not locals_ and turn.get('speaker') == 'SPEAKER_UNKNOWN':
            locals_ = {'SPEAKER_UNKNOWN'}
        windows = [{**w, **candidates.get(local, [{}] * len(windows))[i]}
                   for local, windows in annotated_windows.items() if local in locals_
                   for i, w in enumerate(windows)]
        turn.update(voice_hints(turn, windows, [m['speaker'] for m in matches
                                              if m['local_speaker'] in locals_ and m['status'] == 'matched'], matches))
        if not reviewed and turn['identity_review_reasons']:
            pending_turns.append(turn['id'])
            pending.update(locals_)
    return {'pending': sorted(pending), 'pending_turns': pending_turns}
