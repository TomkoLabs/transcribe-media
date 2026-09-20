# Quality workflow: stable people across recordings

Quality mode is enabled by default starting with version 1.12.0. Version 1.11.0
introduced the reviewed-reference matcher and offline review workflow.
The recommended scenario is two adults with an occasional child. For a recording
or excerpt with only one speaker, explicitly use `--speakers 1` (or allow a range
with `--min-speakers 1 --max-speakers 3`). The software does not infer a
person's age, relationship or identity from their transcript. Optional text context
includes conservative estimates of vocal expression from audio, described below.

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

Quality mode keeps large-v3 unless you explicitly choose another model. It uses
the full faster-whisper sequential decoder with temperature fallback for low
log-probability/repetitive output, beam size 5, and Silero VAD with a lower onset
threshold and short-speech padding. Forced word alignment follows. These choices
exercise decoding controls the batched path did not fully use; they are not a
measured claim of lower WER on your recordings. ASR diagnostic scores remain in
JSON. Audio enhancement and text rewriting are not applied.

The preset allows **2–3 speakers**, uses ASR batch size 1, enables the reviewed
reference workflow, and enables selective emotion2vec+ Large vocal-emotion
estimates. Measured timing and acoustic observations remain available. Use
`--no-tone` to opt out. Do not use `--speakers 2` when
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

1. Start in **Needs attention**, which shows unassigned or flagged soundbites.
   Use **Next soundbite needing attention** to move through them and across files.
   Confident, unflagged turns are already assigned; **All voices** also shows them.
   **Already assigned** contains fully resolved groups.
   Listen to clips from several parts of each local speaker group. A diarizer's
   group can contain mistakes; a good first clip does not verify the whole group.
2. Select an existing VOICE ID if it is the same person. Otherwise choose
   **Adult A**, **Adult B**, **Child**, or add another label under **People & profile labels**.
   The role is your annotation; it is not an automated child detector.
3. Each clip card displays its timestamp, excerpt and assigned person. If a group
   mixes people, use **Change only this clip** for an exception, or expand
   **All voices** and correct a whole soundbite or part of it.
   A group default applies only to unflagged speech; unreviewed timing/overlap/voice
   exceptions remain unknown until you choose individually. Existing manual
   decisions, including older group decisions, are retained. Multiple time
   corrections can be added and removed in the page. Existing word timestamps
   determine assignment by word midpoint; corrections do not rewrite words.
4. Uncheck clips with the wrong speaker, overlap, noise or unsuitable content.
   An acoustic outlier or model-disputed clip is initially unchecked but can be explicitly selected
   after assigning it to its actual speaker. A window spanning different
   reviewed identities is excluded from reference training.
5. Click **Download JSON**, move the downloaded file into `speaker-decisions/`
   without renaming it, then **Copy apply command** and run it from the project.
   **Apply command & optional folder saving** offers direct folder saving in
   compatible browsers. If permission is blocked, use the download workflow:

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
Refresh and review application preserve emotion windows wholly contained within
the rebuilt speaker turns and recheck adult roles. Windows crossing a corrected
speaker boundary are dropped. Use `--refresh-context` to recompute all acoustic
and emotion context afterward without changing text, speaker decisions or profiles.

New identities are **never automatically enrolled in quality mode**. Until reviewed,
the transcript is labeled `DRAFT: SPEAKER REVIEW REQUIRED`, and the manifest records
`awaiting_review`. That is a completed processing job awaiting a person, not an
error that should repeatedly run ASR. A weak or unknown voice stays unresolved
instead of being forced onto an adult profile. Human approval identifies who
spoke; it does not certify ASR wording or eliminate acoustic overlap uncertainty.

Readable transcripts use plain speaker headings without repeating attribution
warnings on individual paragraphs. Adjacent speech by the same identified person
can share a paragraph even if model confidence differs. Speaker changes, long
pauses and separate UNKNOWN turns remain distinct. Detailed TXT/JSON and the
review page retain attribution diagnostics; the primary TXT keeps one draft
notice when speaker assignments still need review.

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

### Listen around uncertain speech

**Listen** includes two seconds before and after the target, or four seconds
when timing, overlap or reference evidence is uncertain. Choose **5s each side**
for more context, or **Exact selection only** to hear the original interval.
Playback stops after the surrounding context. The clock and status distinguish
**Before target**, **Target** and **After target**; the selected card's outline
turns green during the target. Nearby voices are context only: the selected
speaker, correction interval and training audio do not expand with playback.

Under **All soundbites → Correct part of this soundbite**, use **Preview selection**
to audition the start/end values. Listen and pause near a speaker change, then
use **Start at playhead** or **End at playhead**, or type seconds. Choose a person
and click **Set time correction**. Saved corrections also have a Listen button.
Marking or previewing alone does not apply a correction. Boundaries must remain
inside the soundbite; word midpoints determine which words change speaker.
Wider playback helps locate speech, but does not repair inaccurate word alignment.
Timing notes alone do not revoke a confident identity or require another speaker choice.

Groups become **Chosen by you** when all their spoken words have explicit choices;
small unassigned pauses or trimmed clip edges do not hold the group pending.
Reference-clip progress is shown separately: reviewing every sampled clip may
still leave other words unassigned. When all samples have the same person, use
**Assign remaining speech to …** after checking for exceptions. This is an
explicit default for unflagged speech; flagged soundbites remain for individual
review, and sample labels are never silently propagated.

### Word/audio links

New quality transcriptions save Whisper's word boundaries and original ASR sentence
ranges alongside forced-alignment times in the existing detailed JSON. Alignment
remains authoritative for speaker corrections; disagreements over 0.75 seconds,
weak alignment scores, missing/zero timings, stretched words and weak ASR support
are flagged for listening. These are review heuristics, not measured confidence.

Click a word to hear its aligned position. **Listen to source sentence** plays the
recognizer's original sentence range, which can help locate words when alignment
drifted. The alternate playback never expands a correction or voice-training clip.
Reference-clip excerpts include only fully contained aligned words; a sentence
extending outside the clip is no longer presented as that clip's text. Old caches
can use their saved sentence anchors without fresh transcription; independent
Whisper word comparisons require new ASR. No duplicate timing sidecar is needed.
An unaligned word has no playable word button. Assign its whole soundbite explicitly
after listening to the source sentence; group defaults and partial time corrections
cannot place that word. The assignment preserves the missing timestamp rather
than inventing one.

This improves traceability rather than guaranteeing every word is audible: ASR
can hallucinate, and neither timing system resolves all overlapping/faint speech.
Larger alignment models are not an established fix for these cases.
[WhisperX alignment and limitations](https://github.com/m-bain/whisperX)

### Selective vocal-emotion context

Quality mode uses the pinned **emotion2vec+ Large** checkpoint already supported
by the installer (about 300 million parameters). It analyzes real contiguous
speech windows, up to 12 seconds, excluding detected overlap with a boundary guard.
A brief overlap need not suppress clean speech elsewhere in the same turn.

Primary text uses **[Vocal tone estimate: …]** only when an adult profile is
assigned and both the original window and a central crop agree on a non-neutral
emotion with model score at least 0.85 and margin at least 0.30. Each eligible
window needs at least 2.5 seconds, sufficient waveform energy and aligned speech
coverage, no severe clipping, and no flagged word timing. Partial coverage is
labelled as part of the passage. Child/unspecified roles, faint or mixed speech,
neutral/unknown output and inconsistent scores produce no emotion label.

These thresholds are conservative engineering filters, **not calibrated emotion
probabilities or validated clinical accuracy**. Loudness alone cannot establish
anger. Emotion datasets and recording conditions differ; even strong agreement
can be wrong. Absence of a label means abstention, not proof of neutrality.
The model uses audio, not transcript-based speculation about intent or diagnoses.
Raw scores, accepted windows and suppression reasons stay in detailed JSON.
[Official model card](https://huggingface.co/emotion2vec/emotion2vec_plus_large)

Initial recordings without adult profiles retain model windows so applying adult
speaker labels can reveal eligible estimates. Speaker review and voice refresh
reuse only windows that remain wholly within the resulting speaker turn. Use:

```bash
./transcribe-media --prepare-models   # online once if weights are missing
./transcribe-media --refresh-context
```

Context refresh reads saved review WAVs, preserves words/times/identity decisions,
updates outputs and regenerates review pages. A failed refresh rolls back its
changes. It does not rerun ASR, train profiles or change review IDs. Existing saved
decisions remain usable when the original source/evidence are unchanged. A full
reinstall is unnecessary for 1.15.0 because the model dependencies were already
included. Model-free UI regeneration is still `--review-speakers`.

Emotion computation defaults to CPU on the 3080 setup, adding time and system RAM
use while leaving the ASR GPU budget unchanged. Quality mode reports a degraded
run if the estimator fails; it does not substitute a weaker emotion model.

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
Direct folder saving is optional. It requires the browser's File System Access
API and your explicit folder selection; browser restrictions can still block it.
The page hides the option when unsupported or denied and explains the
download-and-move workflow. Cancelling folder selection leaves your draft intact.
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

## Why a well-trained voice might still need review

The profile library stores verified acoustic references; it does not fine-tune the
speaker model. More reviewed recording conditions can improve matching, but the
number of reviewed videos alone does not guarantee a small queue. A new room,
faint/overlapping speech or an actual mixed cluster can still require listening.

Version 1.15.1 corrects three sources of unnecessary review:

- A diarizer disagreement is distinct from an acoustic outlier. Older packets
  combined both into a group-wide mixed flag. The matcher now derives these
  separately without changing the saved evidence. A disputed group is accepted
  only with the existing score/margin checks, at least two comparable samples,
  80% comparable-window agreement, and 80% verified-reference agreement across
  all samples, including disputed ones. The disputed intervals remain reviewable.
- Several non-overlapping local clusters can independently match the same known
  person. They remain separate detected clusters; there is no forced enrollment
  or merging into an adult. Simultaneous clusters cannot automatically receive
  the same person. If independently accepted clusters name the same person,
  a secondary-diarizer label switch between them is not an identity conflict.
  Ambiguous secondary evidence still requires review.
- Word-timing/text warnings remain visible with playback controls, but do not
  revoke a confident speaker identity by themselves. Genuine speaker conflicts,
  unknown voices and unresolved overlapping speech remain in **Needs attention**.
  Use **All voices** to inspect timing/text notes too.

The page now shows the verified-reference scores used for matching and why a group
was withheld. The older overall-centroid resemblance (for example, 0.936) was a
suggestion, not necessarily the score that passed the reference-consensus rules.
Neither score is a probability. **0 / 58 reference clips approved from this
recording** does not mean references from earlier videos disappeared.

Run `./transcribe-media --refresh-voices` after this update. It uses existing saved
embeddings, preserves reviewed decisions and the registry, and regenerates all
outputs/pages without ASR or new models. `--refresh-context` changes acoustic/emotion
context only; it does not invoke voice rematching. Existing source-bound decisions
remain compatible. The reference comparison is now batched to avoid repeatedly
rebuilding every session's reference centroid as the library grows; scoring and
acceptance thresholds are retained.

No measured reduction in review count or speaker-error rate is claimed for your
recordings. Regression fixtures verify that false group-wide blockers are removed
while actual mixed evidence, ambiguous matches, child-role thresholds, known-voice
rosters, manual exclusions and overlap conflicts remain protected.

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

For presentation changes alone, run `./transcribe-media --render-transcripts`.
It regenerates the registered TXT/detailed TXT/subtitle exports from saved JSON,
without rematching speakers, running ASR, learning references or changing JSON
and processing state. Original media and model weights are not needed. It also
preserves existing acoustic/tone annotations. Export files are replaced, so keep
any hand-edited TXT copies separately. Missing/invalid canonical JSON stops the
operation, and write failures roll back all exports. Use `--refresh-voices`
instead when you intend to change automatic matches using improved references.

Processing, review application and merging use one project lock and a recoverable
journal for the registry, manifest and transcript outputs. After interruption,
the next command restores the prior state before proceeding. Files are written
privately and atomically. Whole-content SHA-256 hashes bind decisions to the
recording, including edits in the middle of a file. Quality ASR/alignment caches
are independent of voice matching policy; `--overwrite` forces fresh ASR.

## Hardware and models

On an RTX 3080, ASR runs on CUDA while smaller analysis models run on CPU to keep
VRAM available. Alignment can retry on CPU. The same large-v3 model can retry
initialization with `int8_float16` if float16 runs out of memory; no smaller model
is silently selected. Automatic-device processing can fall back to CPU, with the
selected runtime reported; use `--device cuda` if you need it to fail instead.
There is no larger-model quality tier that requires a GPU upgrade.

| Problem | Next step |
| --- | --- |
| `401`, `403`, or gated-model download error | Accept Community-1 access with the token's account; run `./transcribe-media --configure`, then `./transcribe-media --prepare-models` online |
| Missing weights/offline cache error | Run `./transcribe-media --prepare-models` online, using the same optional language/model/backend flags as the intended run |
| CUDA error or unexpected CPU use | Run `nvidia-smi`, then `./transcribe-media --doctor --device cuda`; repair the driver/environment before retrying |
| Too many uncertain speakers | Review clean clips from several recording conditions; use `--refresh-voices`; do not lower matching thresholds as the first fix |
| Complete processing but draft transcript | Open the review page and resolve the remaining speaker assignments |

**ASUS GX10 / GB10:** use the same installer on its DGX OS. It selects ARM64
PyTorch CUDA 12.9 wheels and builds pinned CTranslate2 with CUDA locally. This
needs a working driver and CUDA toolkit **12.8 or newer** at `/usr/local/cuda`
(or set `TRANSCRIBE_CUDA_ROOT`). Apt installs the build prerequisites.
`TRANSCRIBE_BUILD_JOBS` controls compilation parallelism (default 8). The installer
does not install the driver/toolkit. Other ARM GPU architectures are not covered
by this GB10 build path.

The current release has CPU integration tests and GX10 preflight tests.
**RTX 3080 and GX10 end-to-end hardware acceptance remains outstanding**;
installer GPU inference checks are required on the target machine. No measured
WER/DER or cross-session identity accuracy is claimed. More VRAM alone does not
make profile matching more accurate. Ollama is not used; run this specialized
speech pipeline directly on the GX10 if using that machine.


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
The review page's selection, clip playback, alternate sentence playback, word
playback, exception protection and batch navigation were exercised using synthetic
text/audio. The installed emotion2vec model also passed a CPU inference smoke
test with non-speech abstention; this does not measure emotion accuracy. GPU execution and independently measured WER/DER/
cross-recording identity improvements remain unvalidated.

## TL;DR

Run `./transcribe-media` on all recordings, review uncertain voices in the batch
page, download the decisions into `speaker-decisions/`, and run the copied apply
command. Batch application also refreshes cached speaker matches.
Collect verified clips across sessions and inspect held-out evaluation results.
Use `--no-speaker-learning` once the references are satisfactory. Keep
uncertain child/overlap speech unresolved until reviewed; never force two speakers
when three may be present. The quality workflow does not require a GPU larger
than the existing 3080 target, and it does not use Ollama.

## Recovering saved review state

See [review recovery](docs/REVIEW_RECOVERY.md) for source-fingerprint diagnosis,
private backups and targeted recovery. Generated pages show saved pending state
and a snapshot time separately from unapplied browser edits. See the
[reviewed TXT contract](docs/TXT_CONTRACT.md) before copying files downstream.
