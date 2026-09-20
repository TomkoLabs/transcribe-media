"""Refresh acoustic and vocal-emotion context without changing words or identities."""
from __future__ import annotations

import copy
import json
import wave
from pathlib import Path

from .emotion import gate_turns, estimate_clean_tone
from .storage import ManifestStore, StateTransaction, atomic_write_json, source_fingerprint, utc_now
from .timing import annotate_word_timing, pending_review


def refresh_context(review_dir, estimator, analyzer):
    import numpy as np
    from .review import packet_digest, render_review, review_index
    from .renderers import write_outputs
    from .schema import SAMPLE_RATE
    review_dir = Path(review_dir)
    from .speakers import VoiceRegistry
    registry = VoiceRegistry(review_dir / 'speaker_registry.json')
    manifest = ManifestStore(review_dir / 'transcription_manifest.json')
    paths = sorted((review_dir / 'speaker-reviews').glob('*.json'))
    jobs = []
    for path in paths:
        packet = json.loads(path.read_text(encoding='utf-8'))
        if packet_digest(packet) != packet.get('review_id'):
            raise ValueError(f'review evidence changed: {path.name}')
        if source_fingerprint(Path(packet['source'])) != packet['fingerprint']:
            raise ValueError(f'source recording changed: {packet["source_key"]}')
        entry = manifest.get(packet['source_key'])
        if not entry or entry.get('source_fingerprint') != packet['fingerprint']:
            raise ValueError('recording review and manifest do not agree')
        files = {key: Path(value) for key, value in entry['outputs'].items()}
        payload = json.loads(files['json'].read_text(encoding='utf-8'))
        jobs.append((path, packet, entry, files, payload))
    transaction = StateTransaction(review_dir, [manifest.path, review_dir / 'speaker-reviews/index.html',
        *paths, *(path.with_suffix('.html') for path in paths),
        *(file for _, _, _, files, _ in jobs for file in files.values())])
    transaction.begin()
    try:
        displayed = 0
        for path, packet, entry, files, original in jobs:
            payload = copy.deepcopy(original)
            with wave.open(str(path.with_suffix('.wav')), 'rb') as handle:
                if handle.getframerate() != SAMPLE_RATE or handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                    raise ValueError(f'unsupported review audio: {path.name}; regenerate review audio by processing the source')
                audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype='<i2').astype(np.float32) / 32768.
            annotate_word_timing(payload)
            annotate_word_timing({'segments': [{'words': t.get('words', [])} for t in payload['turns']],
                                  'asr_diagnostics': payload.get('asr_diagnostics', [])})
            for turn in payload['turns']:
                turn['tone'] = None
            if analyzer is not None:
                analyzer.analyze(audio, payload['turns'])
            eligible = estimate_clean_tone(estimator, audio, payload['turns'])
            failures = [t for t in eligible if (t.get('tone') or {}).get('kind') == 'unavailable']
            if failures:
                raise RuntimeError(f'vocal-emotion inference failed for {packet["source_key"]}; previous outputs are preserved')
            gate_turns(payload['turns'], payload.get('speaker_profiles', []))
            displayed += sum(bool((t.get('tone') or {}).get('display', {}).get('label')) for t in payload['turns'])
            payload['processing']['provenance']['context_refresh'] = {
                'at': utc_now(), 'tone_model': estimator.name, 'tone_version': estimator.version,
                'words_and_speakers_preserved': True}
            payload['speaker_review'] = {**payload.get('speaker_review', {}), **pending_review(payload, packet['evidence'], registry)}
            packet['payload'], packet['pending'] = payload, payload['speaker_review']['pending']
            entry['speaker_review_pending'] = len(packet['pending'])
            if not entry.get('retry_recommended'):
                entry['status'] = 'awaiting_review' if packet['pending'] else 'complete'
            write_outputs(files, payload)
            atomic_write_json(path, packet)
            render_review(path, packet)
            manifest.update(packet['source_key'], entry)
        index = review_index(review_dir)
        transaction.commit()
    except BaseException:
        transaction.rollback()
        raise
    return {'recordings_updated': len(jobs), 'vocal_emotion_labels': displayed, 'review_index': str(index)}
