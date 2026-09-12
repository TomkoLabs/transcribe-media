# Quality workflow: stable people across recordings

Quality mode is enabled by default starting with version 1.12.0. Version 1.11.0
introduced the reviewed-reference matcher and offline review workflow.
The recommended scenario is two adults with an occasional child. For a recording
or excerpt with only one speaker, explicitly use `--speakers 1` (or allow a range
with `--min-speakers 1 --max-speakers 3`). The software does not infer a
person's age, relationship, emotion or identity from their transcript.

## Start here

Follow the [README quick start](README.md#quick-start-debian--rtx-3080), including
Community-1 access and the Hugging Face read token. The installer prepares the
default models. After adding recordings, run:

```bash
./transcribe-media
./transcribe-media --review-speakers
```

On the recommended CUDA installation with a token, automatic selection uses
Community-1 plus Sortformer as a second opinion. Community-1 alone is available
with `--diarization-backend pyannote`. Token-free SpeechBrain clustering is a
weaker baseline for speaker changes and overlap. Check the actual backend in
the run output and JSON.
The included development sample is not a labeled evaluation corpus.

Quality mode keeps large-v3 unless you explicitly choose another model. It uses
the full faster-whisper sequential decoder with temperature fallback for low
log-probability/repetitive output, beam size 5, and Silero VAD with a lower onset
threshold and short-speech padding. Forced word alignment follows. These choices
exercise decoding controls the batched path did not fully use; they are not a
measured claim of lower WER on your recordings. ASR diagnostic scores remain in
JSON. Audio enhancement and text rewriting are not applied.

The preset allows **2–3 speakers**, uses ASR batch size 1, enables the reviewed
reference workflow, and disables inferred vocal tone by default. Measured timing
and acoustic observations remain available. Use `--tone-backend emotion2vec`
explicitly if that optional estimate is wanted. Do not use `--speakers 2` when
a child might speak: an exact count can force three people into two labels.
An exact count should be supplied only when known for every file in the batch.

## Confirm initial profiles

Open `Review/speaker-reviews/index.html` in your own browser. This aggregates all
recordings with shared people, a recording selector, progress counts, local
playback, reference clips, timestamps, all turns and candidate VOICE IDs. Individual
recording HTML files remain available. There are no external scripts or services.
The pages use sibling WAVs, so keep the review folder together. On a headless GX10,
copy this review folder to your desktop for listening, then apply the exported
decisions on the machine holding the original project and recordings.

1. Start in **Needs attention**, which leaves uncertain groups unassigned.
   **Already assigned** contains confident automatic matches and your selections.
   Listen to clips from several parts of each local speaker group. A diarizer's
   group can contain mistakes; a good first clip does not verify the whole group.
2. Select an existing VOICE ID if it is the same person. Otherwise choose
   **Adult A**, **Adult B**, **Child**, or add another label under **People & profile labels**.
   The role is your annotation; it is not an automated child detector.
3. Each clip card displays its timestamp, excerpt and assigned person. If a group
   mixes people, use **Change only this clip** for an exception, or expand
   **All soundbites** and correct a whole turn or part of it. Multiple time
   corrections can be added and removed in the page. Existing word timestamps
   determine assignment by word midpoint; corrections do not rewrite words.
4. Uncheck clips with the wrong speaker, overlap, noise or unsuitable content.
   An acoustic outlier or model-disputed clip is initially unchecked but can be explicitly selected
   after assigning it to its actual speaker. A window spanning different
   reviewed identities is excluded from reference training.
5. Click **Save to project** and select the project folder. Supported browsers
   write into `speaker-decisions/`. Alternatively, **Download JSON** and move the
   file there yourself. Click **Copy apply command** and run it from the project:

```bash
./transcribe-media --apply-speaker-review "speaker-decisions/batch-REPLACE_WITH_EXPORTED_ID.decisions.json"
```

Applying a review assigns durable IDs, stores the selected references as
human-verified evidence, and immediately regenerates TXT, JSON and requested
subtitles. It does not rerun ASR or change recognized words. Refresh rematches
cached recordings using verified references, preserving existing human choices.
An aggregated export applies all explicit reviews and then refreshes every cached
recording, using the final shared references. No separate `--refresh-voices` is
needed. A single-recording export updates that recording; use `--refresh-voices`
afterward to propagate improved profiles. Strong matches are applied automatically;
unresolved matches stay in the queue. There is no automatic equivalence between
the same local SPEAKER number in different recordings. Choose the same shared
person only after listening; processing order need not determine enrollment.
Refresh and review application leave old per-turn tone/acoustic estimates out
when rebuilding turns; a normal processing run recomputes them.

New identities are **never automatically enrolled in quality mode**. Until reviewed,
the transcript is labeled `DRAFT: SPEAKER REVIEW REQUIRED`, and the manifest records
`awaiting_review`. That is a completed processing job awaiting a person, not an
error that should repeatedly run ASR. A weak or unknown voice stays unresolved
instead of being forced onto an adult profile. Human approval identifies who
spoke; it does not certify ASR wording or eliminate acoustic overlap uncertainty.

A new person can be saved **with no usable reference audio**. Their transcript
label is valid independently of whether the voice profile can identify them
automatically later. Profiles display **untrained** (no verified clips),
**collecting** (some clips, insufficient support), or **ready** (a reference
condition meets the support rules). Ready is not an accuracy guarantee.

TXT turns and subtitles display the person's label with the stable ID, for example
`Adult A [VOICE_0001]`. JSON retains the canonical ID and profile catalog. Filenames
stay tied to their source recording. `SPEAKER_01` is a local detection label; a
confirmed person receives a durable `VOICE_…` ID.

**Unknown / review later** excludes a passage from training and leaves it pending.
**UNKNOWN — exclude from voice learning** also excludes training, preserves the
words under `UNKNOWN`, and records a completed human decision without repeatedly
queuing it for identity review. Both survive refresh. This is an abstention about
identity, not proof that the voice belongs to someone outside your named profiles.
To keep a known person's label but skip training, simply uncheck the reference clip.

Automatic matching needs either two verified clips of at least 2.5 seconds, or
at least five consistent clips of 0.8–under 2.5 seconds totaling at least eight
seconds, **from the same recording**. The short-clip route compares every clip
with the set's centroid, then requires agreement with two disjoint embedding
pools. It does not stitch audio. Query-evidence, score, margin, child-role and
agreement checks still apply. Shorter speech can be labeled, but cannot serve
as a reference. Nonverbal vocalizations are not guaranteed to be transcribed.

### Save drafts and manage people

**Import saved review** restores an exported decisions JSON before it has been
applied. It checks the recording and validates all choices before changing the
page. A local browser draft is also saved when browser storage is available;
export is the portable backup. **Reset draft** restores the last applied review.
In the aggregated page, reset covers the whole batch. Import validates every
included recording before changing any draft. Older single-recording files can
be imported into the batch page; other recordings keep their choices.
The page stays fully offline: saving/importing decisions cannot directly write transcript or
registry files. Applying through the copied command keeps the workflow to one
HTML page and one command, without installing or securing a local web service.
Folder saving requires the browser's File System Access API and your explicit
folder selection; unsupported browsers use the download-and-move fallback.
The copied command always uses `speaker-decisions/` relative to the project root.
The script also resolves that folder against its installed project when invoked
from another working directory. No Downloads location is assumed.
[Browser folder access](https://developer.mozilla.org/en-US/docs/Web/API/Window/showDirectoryPicker)

**People & profile labels** lets you add people, edit labels and adult/child roles,
remove unused draft people, and archive existing profiles. Removing a draft person
leaves their draft assignments unknown; archiving keeps historical IDs and labels
while excluding that person from future automatic matching. Applied label edits
update registered transcript outputs. Existing IDs are preserved; archive rather
than deleting a person referenced by previous recordings. If using `--known-voices`,
remove an archived ID from that roster before applying or processing.

Automatic assignments are not exported as human approval and do not train
themselves. To add references for an already confident person, listen and use
**Confirm this person & learn**. An uncertain group's explicit assignment approves
its selected suitable references; clip/turn corrections take precedence. A window
crossing different or unresolved identities cannot train a profile.

## How the references improve

Reference extraction keeps real contiguous waveform windows, trims boundaries,
excludes detected overlap, rejects severe clipping and
very low energy, and checks acoustic consistency. It does not splice disjoint
replies into artificial speech. These checks are useful filters, not proof that
a clip has no background noise, music or speaker contamination. Diarizer-disputed
clips are excluded from automatic evidence but retained, initially unchecked,
for deliberate human review when otherwise usable. Pauses between transcribed
turns no longer disqualify the surrounding reviewed speech window.

Verified clips remain separate from automatic observations and are never evicted
by the automatic observation limit. Quality matching uses only verified clips.
Automatic matches can be logged as strong observations but cannot silently become
trusted reference material or replace confirmed anchors. This limits the feedback
loop in which an early wrong match makes later wrong matches look stronger.

Each person's verified recordings provide separate condition prototypes. A match
needs support from a condition centroid and two individual clips (or the two
short-clip pools described above), plus a margin
over competing people and agreement across query windows. Confirming another
room/microphone/session can add useful coverage without averaging that condition
away. The number of reference conditions can also affect false-match rates, so
the thresholds remain conservative heuristics requiring evaluation. Child-labeled
profiles use a stricter acceptance threshold. Pitch is not used to infer age.
Learning means accumulating verified speaker embeddings; it does not fine-tune
the speech-recognition or speaker-embedding neural networks.

The candidate list displays **cosine similarity, not probability**. A similarity
of 0.82 does not mean an 82% chance of identity. Diagnostic centroid scores shown
for comparison are distinct from the stricter condition-and-window checks used
for automatic acceptance. Older automatically generated profiles can appear as
candidates, but quality mode requires reviewed clips before trusting them.

Begin with several clean utterances from different sessions per person. Roughly
30–60 seconds across three recordings is a useful collection target to evaluate,
not a guarantee. Include different microphones, ordinary expressive speech and
the child when present. Review should become less frequent as the reference
coverage improves; growth, illness, new microphones, overlap and unfamiliar voices
can still require confirmation.

```bash
./transcribe-media --evaluate-voices
```

This writes `Review/voice-evaluation.json`. It excludes the held-out recording
from every profile and reports incorrect accepted matches, abstentions and
coverage. It cannot evaluate cross-recording accuracy from one recording. It
uses selected reference clips and is not a substitute for a separate labeled
conversation test set, WER, overlap-aware DER, or speaker-attributed word error.
It deliberately does not manufacture confidence percentages or automatically
lower thresholds based on a small sample.

Once reference coverage is satisfactory:

```bash
./transcribe-media --no-speaker-learning
```

The verified-reference matcher remains enabled by default. Add
`--known-voices VOICE_0001,VOICE_0002,VOICE_0003` only when that roster is correct
for the whole batch; include the child's ID when applicable. Freezing disables
automatic updates and enrollment. An explicit `--apply-speaker-review` still
applies the human reference changes you request.

## Correct duplicates and preserve review history

After listening and confirming that two profiles are the same person:

```bash
./transcribe-media --merge-voices VOICE_0004 VOICE_0001
```

The first ID is the duplicate; the second is retained. The operation moves its
references, keeps an alias/audit record, and updates registered transcript outputs
and review choices. It does not infer that similar voices should be merged.
Use the canonical ID in future rosters. Keep a backup of the whole ignored
`Review` directory, including the registry, manifest, review packets and audio.

Repeated imports are idempotent. Revised reference selections replace the
reviewed references from that source; unchecking old clips removes them from
training. An ID with no remaining references is retained but cannot automatically
match until it has evidence again. Human decisions survive ordinary voice-policy
reruns when the source and local transcript boundaries are unchanged. If those
boundaries or source change, the run preserves the old outputs and asks for a
fresh review rather than silently applying obsolete labels. To explicitly start
that recording's review again, archive its HTML/JSON/WAV review packet first.

New exports are complete snapshots: removing a turn/time override in the page
and applying the export removes that override. Older exports remain supported
as incremental edits. Reapplying a revised draft reuses its previously created
people, including after merges. Reapplying the most recent file is a no-op;
applying an older snapshot after a newer edit restores that snapshot's choices.

After upgrading, `./transcribe-media --review-speakers` rebuilds existing HTML
pages with the current UI and profile catalog without loading models. Existing
failed decisions exports can be retried directly when their source is unchanged.
Newly extracted short/disputed reference windows require processing the recording
again; the review update cannot recover embeddings absent from an old packet.

Processing, review application and merging use one project lock and a recoverable
journal for the registry, manifest and transcript outputs. After interruption,
the next command restores the prior state before proceeding. Files are written
privately and atomically. Whole-content SHA-256 hashes bind decisions to the
recording, including edits in the middle of a file. Quality ASR/alignment caches
are independent of voice matching policy; `--overwrite` forces fresh ASR.

## Hardware and models

No larger GPU is required by this quality preset. It retains the existing
RTX 3080 policy: ASR on CUDA, other models on CPU for constrained VRAM, with the
same-model compute fallback. The full decoding fallback may take longer. A GX10's
larger memory allows more analysis work on CUDA under the existing runtime policy.
GPU hardware validation remains pending; the ARM build path and required target
machine checks are documented in the README and contributing guide.

Whisper large has 1.55 billion parameters; speaker encoders are much smaller.
The original Whisper project's memory table is not an exact memory measurement
for this CTranslate2 pipeline. [Whisper models](https://github.com/openai/whisper#available-models-and-languages)

Ollama is not used. These are specialized speech, diarization and speaker-embedding
models. Running a remote text-model endpoint would not provide voice matching
and would add a transfer step. Running this same Python pipeline directly on the
GX10 is the supported arrangement; a separate authenticated speech inference
service could be future work if processing must be split between machines.
[Ollama's chat API](https://docs.ollama.com/api/chat)

Alternative speech/speaker models, automatic denoising and mono overlap separation
are not part of this release. They need matched recording-level comparisons,
dependency/hardware validation and, for an identity encoder change, re-embedding
reviewed source audio while retaining canonical IDs. Additional GPU memory alone
does not establish more accurate identity matching. Preserve channels at capture
time when separate microphones are available; this version still downmixes to
mono and does not separately transcribe simultaneous speakers.

## Validation performed

The automated suite covers first enrollment, frozen matching, changed
recording conditions, partial-turn correction, reference removal, duplicate merges,
cache reuse, source changes, concurrent access and recovery after write failure.
An end-to-end local sample completed on CPU with the full decoding fallback,
alignment, token-free diarization, reference extraction and review generation.
A subsequent run reused the ASR/alignment cache. The fallback produced three
candidate groups, requiring review; this does not prove correct speaker separation.
The review page's selection, clip playback and export controls were exercised
using synthetic text and silent audio. GPU execution and independently measured WER/DER/
cross-recording identity improvements remain unvalidated.

## TL;DR

Run `./transcribe-media`, confirm the first profiles in the local review page, apply the
exported decisions, then use `--refresh-voices` to reduce the remaining queue.
Collect verified clips across sessions and inspect held-out evaluation results.
Use `--no-speaker-learning` once the references are satisfactory. Keep
uncertain child/overlap speech unresolved until reviewed; never force two speakers
when three may be present. The quality workflow does not require a GPU larger
than the existing 3080 target, and it does not use Ollama.
