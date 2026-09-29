# Source-bound review recovery (1.16.0.dev1)

## Diagnosis: facts and limits

1.15.1 compared whole fingerprint dictionaries: full-media SHA-256, byte size,
mtime in nanoseconds, `sample_sha256` (a compatibility alias of the full hash),
and `hash_algorithm`. Path is separate, but the review digest also binds the
absolute source path, fingerprint, local transcription, evidence, processing
settings hash and embedding model. The manifest must agree with the packet.
An mtime-only change could therefore reject identical media. Refresh raised at
the first changed recording, preventing the remaining recordings and index from
being refreshed. Reinstalling code does not repair those saved artifacts.

The browser previously showed attention calculated from editable local drafts,
while CLI pending counts came from saved review results. Downloading or editing
JSON does not apply it. Old HTML embeds an old snapshot; localStorage may restore
unapplied edits. These are proven ways counts can differ. Which occurred in any
particular operator session, and whether its source bytes changed, cannot be
established without that runtime's diagnosis. Development tests do not establish
real recording accuracy or Kraken recovery.

1.16.0.dev1 trusts equivalence only when **both** fingerprints declare
`sha256-full-v1`, full hashes match, and sizes match. It hashes every source byte;
mtime and explicitly supplied location can change without ASR. A source modified
while being hashed is unavailable for this operation. Legacy/unknown hash
provenance is blocked, not guessed equivalent. Digest or manifest disagreement
requires restoration from a matching backup. A global registry failure still
requires registry repair; per-source isolation cannot replace shared lost state.

## Short safe procedure

Stop other transcribe/review commands first. Keep media fixed during maintenance.
Use your actual custom directory options on **every** command if not using the
standard project layout. Commands below assume the existing standard layout.
No reinstall, model preparation, model download, or inference is needed for
upgrade, diagnosis, metadata recovery, review, or rerendering.

```bash
cd '/mnt/EVO870/PYTHON_PROJECTS/transcribe-media'
umask 077
backup="../transcribe-review-backup-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -m 700 "$backup"
# Review contains manifest, registry, packets, cached audio, and batch receipts.
# Include the project's exported decisions explicitly; also retain any unapplied
# decisions still in your browser's chosen downloads folder at their exact paths.
cp -a -- Review Transcribed speaker-decisions "$backup/"
chmod -R go-rwx "$backup"
git pull --ff-only origin main
./transcribe-media --version
./transcribe-media --diagnose-reviews
```

Use the exact `source` key printed by diagnosis (no basename/fuzzy matching):

```bash
./transcribe-media --recover-review 'synthetic-session.wav' --dry-run
./transcribe-media --recover-review 'synthetic-session.wav'
```

- `metadata_equivalent`: retains words, timing, diarization, decisions and registry
  learning. Rebinds the saved fingerprint/path and generates a new review ID.
  Export new decisions; old exports correctly become obsolete after this repair.
  Ordinary refresh also tolerates matching-content mtime changes without rebasing
  review IDs. Explicit recovery updates the recorded metadata.
- If identical media moved, provide its exact path:
  `./transcribe-media --recover-review 'synthetic-session.wav' --relocated-source '/exact/new/location.wav' --dry-run`
  then repeat without `--dry-run`. A path owned by another review is rejected.
  The original logical source key and output names stay unchanged. Use
  `--only-source 'synthetic-session.wav'` for future targeted processing at that
  relocated path. Do not discover that moved file as a new logical recording in
  a broad transcription run; this maintenance feature does not rename projects.
- `source_content_changed`: old words/times/diarization/decisions are historical
  only. Recovery saves a private backup and marks active exports DRAFT. It arms
  only this source for fresh transcription; reviewed references in the registry
  remain historical evidence tied to the old content hash, never relabeled as
  evidence from the replacement media. Explicitly run:
  `./transcribe-media --only-source 'synthetic-session.wav'`
  This is the **only inference step** and runs on the operator's prepared machine
  with its existing models/settings. Supply the same language/backend options
  used for that source. Do not add `--overwrite` or reset the registry.
- `source_unavailable`: restore access or supply an explicitly matching relocation.
- `malformed_saved_state` / `recovery_required`: follow the printed action. Do not
  edit fingerprints or guess that a missing hash means unchanged content.

Recovery itself never transcribes. Timestamped private backups are under
`Review/recovery-backups/`; `inventory.json` maps exact original paths to backup
files, including the manifest, registry, packet, cached audio and existing
exports. No media is removed. Successful repeat recovery is a no-op. A durable
before-image journal precedes active-file writes; the next mutating command rolls
back an interrupted transaction before doing new work. Read-only diagnosis
reports a pending journal and does not roll it back. A failed backup prevents
mutation. Retained backups can be restored by the operator to the exact paths in
that inventory while commands are stopped; never mix manifests/registries from
unrelated snapshots.

After recovery (or while an unrelated source remains blocked):

```bash
./transcribe-media --refresh-voices
./transcribe-media --review-speakers
```

Open the **newly generated** `Review/speaker-reviews/index.html`. Check its Saved
state, pending count, snapshot ID, generation time, version, and source action.
Close old tabs and reopen/hard-refresh after each CLI change. This is an offline
snapshot, not a live connection. Blocked records stay visible but are omitted
from batch exports; current recordings remain reviewable. The browser draft's
attention count is explicitly separate from authoritative saved pending items.

Select the desired recording, resolve its remaining soundbites, assign canonical
Husband/Wife labels, and download **new** JSON. Move that exact file into
`speaker-decisions/`; use the browser's copied apply command, for example:

```bash
./transcribe-media --apply-speaker-review 'speaker-decisions/ACTUAL_NEW_FILENAME.decisions.json'
./transcribe-media --render-transcripts
./transcribe-media --diagnose-reviews
```

A stale decision error identifies the file, logical source, shortened stored and
current fingerprint, mismatch classification and next action without printing
speech or speaker decisions. An older pre-recovery decision's original fingerprint
is retained when rebasing metadata. Never keep reapplying an obsolete download.

Verify a specific output using its exact actual filename:

```bash
head -n 1 -- 'Transcribed/synthetic-session.wav.txt'
```

It must read `ANALYSIS-READY TRANSCRIPT` to transfer. To classify **all** TXT files
without printing dialogue and copy only READY files, use this explicit script.
Set the real Couples Insight configured TXT inbox first; it is deployment-specific.
Run it only after the diagnosis/refresh/render sequence above, with no concurrent
media changes:

```bash
export COUPLES_TXT_INBOX='/exact/configured/couples-insight/txt-inbox'
python3 - <<'PY'
import os
import shutil
from pathlib import Path
root = Path('Transcribed')
target = Path(os.environ['COUPLES_TXT_INBOX'])
if not target.is_dir():
    raise SystemExit('Set COUPLES_TXT_INBOX to the existing configured TXT inbox.')
for source in sorted(root.rglob('*.txt')):
    with source.open(encoding='utf-8') as handle:
        first = handle.readline().rstrip('\r\n')
    ready = first == 'ANALYSIS-READY TRANSCRIPT'
    print('READY' if ready else 'DRAFT / not eligible', source.relative_to(root))
    if ready:
        destination = target / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if destination.exists():
            print('Already present; compare/import through the normal operator workflow:', destination)
            continue
        with source.open('rb') as src, destination.open('xb') as dst:
            os.fchmod(dst.fileno(), 0o600)
            shutil.copyfileobj(src, dst)
PY
```

The downstream watcher handles the copied finalized TXT through its normal
identity gate. Do not transfer JSON, media, private registry files, or drafts to
work around that gate. See [TXT contract](TXT_CONTRACT.md).

## Optional separate clean-room acceptance

Keep the existing working state and backups. Choose a new, empty runtime directory
outside source repositories; use the existing launcher/environment and installed
models. On the operator machine only:

```bash
umask 077
acceptance="/mnt/EVO870/couples-acceptance-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -m 700 "$acceptance"
./transcribe-media '/exact/existing/media-directory' \
  --review-dir "$acceptance/Review" --transcript-dir "$acceptance/Transcribed"
```

Repeat both output-directory options for review, apply, refresh and render; use an absolute decisions filename. The media directory is positional.

This optional run performs real transcription and requires operator authorization,
available GPU/runtime, existing models and time. It was not run during development.
Do not merge the fresh registry into the existing one. Compare words and speaker
identities by listening, not by assuming matching VOICE numbers.

## Release acceptance still required

This is a development release; no stable tag. Kraken must demonstrate real
metadata-only recovery without inference; targeted changed-content transcription;
unrelated recording preservation; fresh review download/apply; zero saved pending
items and READY output for the intended finalized recordings; deliberately DRAFT
output for unresolved recordings; canonical identities by listening; and successful
normal downstream import. Follow CONTRIBUTING's existing hardware acceptance
requirements before a stable release. Model accuracy and the exact cause of the
operator's historical CLI/UI mismatch remain unverified here.

Development evidence: [validation results](REVIEW_RECOVERY_VALIDATION.md).
