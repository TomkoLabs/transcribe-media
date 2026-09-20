"""Source-bound, model-free diagnosis and single-recording recovery.

Mutating callers hold project_lock and recover the existing transaction journal.
"""
from __future__ import annotations

import copy
import fcntl
from contextlib import contextmanager
import json
import os
import shlex
import shutil
import tempfile
from pathlib import Path

from . import __version__
from .storage import (ManifestStore, StateTransaction, atomic_write_json,
                      source_fingerprint, stable_hash, utc_now)


def same_content(first, second):
    return (isinstance(first, dict) and isinstance(second, dict)
            and first.get('hash_algorithm') == second.get('hash_algorithm') == 'sha256-full-v1'
            and isinstance(first.get('sha256'), str) and len(first['sha256']) == 64
            and first['sha256'] == second.get('sha256') and first.get('size') == second.get('size'))


def pending_count(payload):
    state = payload.get('speaker_review') or {}
    # Groups and turns overlap: report the larger queue, never add them together.
    return max(len(state.get('pending', [])), len(state.get('pending_turns', [])))


def is_ready(payload):
    state = payload.get('speaker_review') or {}
    return not pending_count(payload) and state.get('source_status', 'current') in ('current', 'metadata_equivalent')


def short_fingerprint(value):
    if not isinstance(value, dict):
        return 'unavailable'
    return {key: value.get(key) for key in ('size', 'mtime_ns', 'hash_algorithm')} | {
        'sha256': str(value.get('sha256', 'unavailable'))[:16]}


def inspect_packet(path, packet=None, *, source=None, manifest=None):
    from .review import packet_digest
    result = {'source': Path(path).name, 'state': 'malformed_saved_state', 'pending': None,
              'decisions_allowed': False, 'action': 'Restore this packet and its manifest from a matching backup; do not edit evidence.'}
    try:
        packet = packet if packet is not None else json.loads(Path(path).read_text(encoding='utf-8'))
        result.update(source=packet['source_key'], stored_fingerprint=short_fingerprint(packet['fingerprint']))
        if packet.get('version') != 1 or packet_digest(packet) != packet.get('review_id'):
            return result
        if (not isinstance(packet.get('local_result'), dict) or not isinstance(packet.get('evidence'), dict)
                or not isinstance(packet.get('payload'), dict) or not isinstance(packet['payload'].get('processing'), dict)
                or packet['payload'].get('source', {}).get('fingerprint', packet['fingerprint']) != packet['fingerprint']):
            return result
        result['pending'] = pending_count(packet['payload'])
        if 'speaker_review' not in packet['payload']:
            result['pending'] = len(packet['pending'])
        if manifest is not None:
            entry = manifest.get(packet['source_key'])
            if not entry or entry.get('source_fingerprint') != packet['fingerprint']:
                return result
        target = Path(source or packet['source']).expanduser()
        action = './transcribe-media --recover-review ' + shlex.quote(packet['source_key'])
        result['action'] = action + ' --dry-run'
        try:
            current = source_fingerprint(target)
        except (OSError, RuntimeError):
            result.update(state='source_unavailable', action=result['action'] + '; restore access or supply --relocated-source /exact/path')
            return result
        result['current_fingerprint'] = short_fingerprint(current)
        if packet['fingerprint'].get('hash_algorithm') != 'sha256-full-v1':
            result.update(state='recovery_required', action='Full content provenance unavailable; restore a matching modern backup before recovery.')
            return result
        if not same_content(packet['fingerprint'], current):
            result.update(state='source_content_changed', action=action + '; then ./transcribe-media --only-source ' + shlex.quote(packet['source_key']))
            return result
        if packet.get('reprocess_required'):
            result.update(state='recovery_required', action='./transcribe-media --only-source ' + shlex.quote(packet['source_key']))
            return result
        equivalent = (current != packet['fingerprint'] or str(target.resolve()) != packet['source']
                      or packet['payload'].get('speaker_review', {}).get('source_status', 'current') not in ('current', 'metadata_equivalent'))
        result.update(state='metadata_equivalent' if equivalent else 'pending_review' if result['pending'] else 'ready',
                      decisions_allowed=True, action=action if equivalent else 'Export and apply a fresh review, then render transcripts.' if result['pending'] else 'Ready for downstream transfer.')
        return result
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError):
        # Never echo an exception containing saved transcript/decision values.
        return result


def snapshot(path, packet, *, manifest=None):
    status = inspect_packet(path, packet, manifest=manifest)
    return {**status, 'generated_utc': utc_now(), 'program_version': __version__,
            'snapshot_id': stable_hash([status, packet.get('review_id'), packet.get('applied_decisions')])[:16]}


@contextmanager
def diagnosis_lock(review_dir):
    """Use an existing lock without creating files; first-use diagnosis is a snapshot."""
    try:
        handle = (Path(review_dir) / '.batch.lock').open('rb')
    except FileNotFoundError:
        yield
        return
    with handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Review state is in use; retry read-only diagnosis after the active command finishes.') from exc
        yield


def diagnose_reviews(review_dir):
    root = Path(review_dir)
    if (root / '.state-transaction.json').exists():
        return {'state': 'interrupted_transaction', 'action': 'Run a mutating maintenance command to roll back the journal first.', 'recordings': []}
    manifest = ManifestStore(root / 'transcription_manifest.json')
    return {'recordings': [inspect_packet(path, manifest=manifest)
                           for path in sorted((root / 'speaker-reviews').glob('*.json'))]}


def _backup(review_dir, paths):
    root = Path(review_dir) / 'recovery-backups'
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    backup = Path(tempfile.mkdtemp(prefix=utc_now().replace(':', '-') + '-', dir=root))
    inventory = {}
    for index, path in enumerate(dict.fromkeys(Path(p) for p in paths)):
        if not path.exists():
            continue
        destination = backup / f'{index}-{path.name}'
        with path.open('rb') as src, destination.open('xb') as dst:
            os.fchmod(dst.fileno(), 0o600)
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        inventory[str(path)] = destination.name
    atomic_write_json(backup / 'inventory.json', inventory)
    for directory in (root, root.parent):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return backup


def recover_review(review_dir, source_key, *, relocated_source=None, dry_run=False, quarantine=False, managed=False):
    """Exact logical key; relocation keeps that identity and existing output paths."""
    from .review import packet_path, packet_digest, render_review
    from .renderers import write_outputs
    root = Path(review_dir)
    path = packet_path(root, source_key)
    if not path.is_file():
        raise ValueError('No exact review source key; use --diagnose-reviews and copy its source value. No basename/fuzzy matching is performed.')
    manifest = ManifestStore(root / 'transcription_manifest.json')
    packet = json.loads(path.read_text(encoding='utf-8'))
    if packet.get('source_key') != source_key:
        raise ValueError('Review source key is inconsistent; restore a matching backup.')
    status = inspect_packet(path, packet, source=relocated_source, manifest=manifest)
    if status['state'] not in ('metadata_equivalent', 'source_content_changed', 'source_unavailable'):
        return status
    if relocated_source:
        target = Path(relocated_source).expanduser().resolve()
        for other in (root / 'speaker-reviews').glob('*.json'):
            if other != path:
                try:
                    if Path(json.loads(other.read_text())['source']).resolve() == target:
                        raise ValueError('Relocated path already belongs to another review; recovery is ambiguous.')
                except (KeyError, json.JSONDecodeError):
                    continue
        if status['state'] != 'metadata_equivalent':
            raise ValueError('Relocation requires matching full content hashes; restore the original source path for changed-content reprocessing.')
    if dry_run:
        return status
    if status['state'] == 'source_unavailable' and not quarantine:
        return status
    entry = manifest.get(source_key)
    blocked = not status['decisions_allowed']
    if blocked and packet.get('reprocess_required') and not quarantine:
        return {**status, 'backup': packet.get('recovery_backup'), 'already_recovered': True}
    if quarantine and packet['payload'].get('speaker_review', {}).get('source_status') == status['state']:
        return status
    outputs = {key: Path(value) for key, value in entry['outputs'].items()}
    paths = [manifest.path, path, path.with_suffix('.html'), *outputs.values()]
    backup = _backup(root, [*paths, path.with_suffix('.wav'), root / 'speaker_registry.json', root / 'speaker-review-batches.json'])
    transaction = StateTransaction(root, paths)
    if not managed:
        transaction.begin()
    try:
        packet = copy.deepcopy(packet)
        payload = packet['payload']
        if blocked:
            payload.setdefault('speaker_review', {})['source_status'] = status['state']
            packet['reprocess_required'] = not quarantine or packet.get('reprocess_required', False)
            entry.update(status='recovery_required', completion_state=False)
        else:
            packet.setdefault('prior_review_fingerprints', {})[packet['review_id']] = packet['fingerprint']
            packet['source'] = str(Path(relocated_source or packet['source']).resolve())
            current = source_fingerprint(Path(packet['source']))
            if not same_content(packet['fingerprint'], current):
                raise RuntimeError('Source changed during recovery; no repaired state was published. Keep media fixed and diagnose again.')
            packet['fingerprint'] = current
            packet['review_id'] = packet_digest(packet)
            payload['source']['fingerprint'] = packet['fingerprint']
            if 'path' in payload['source']:
                payload['source']['path'] = packet['source']
            payload.setdefault('speaker_review', {})['source_status'] = 'current'
            entry.update(source_path=packet['source'], source_fingerprint=packet['fingerprint'])
            if not entry.get('retry_recommended'):
                entry.update(status='awaiting_review' if pending_count(payload) else 'complete', completion_state=True)
        packet['recovery_backup'] = str(backup)
        packet['recovered_utc'] = utc_now()
        write_outputs(outputs, payload)
        atomic_write_json(path, packet)
        manifest.update(source_key, entry)
        render_review(path, packet)
        if not managed:
            transaction.commit()
    except BaseException:
        if not managed:
            transaction.rollback()
        raise
    return {**inspect_packet(path, packet, manifest=manifest), 'backup': str(backup), 'previous_state': status['state']}
