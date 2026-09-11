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

Open `Review/speaker-reviews/index.html` in your own browser. Each recording has
a page with local playback, clean reference clips, timestamps, all transcribed
turns, and candidate VOICE IDs. There are no external scripts or hosted services.
The page uses a sibling WAV, so keep the HTML and WAV together. On a headless GX10,
copy this review folder to your desktop for listening, then apply the exported
decisions on the machine holding the original project and recordings.

1. Listen to clips from several parts of each local speaker group. A diarizer's
   group can contain mistakes; a good first clip does not verify the whole group.
2. Select an existing VOICE ID if it is the same person. Otherwise choose
   **Create Adult A**, **Create Adult B**, **Create Child**, or add another label.
   The role is your annotation; it is not an automated child detector.
3. If a group mixes people, expand its turns and override the affected turn's
   identity. Narrow its start/end seconds to correct only part of a turn.
   Existing word timestamps determine assignment by word midpoint. Multiple
   disjoint ranges can also be supplied in the decision JSON.
4. Uncheck clips with the wrong speaker, overlap, noise or unsuitable content.
   An acoustic outlier is initially unchecked but can be explicitly selected
   after assigning it to its actual speaker. A window spanning different
   reviewed identities is excluded from reference training.
5. Export decisions, then apply the downloaded JSON:

```bash
./transcribe-media --apply-speaker-review /path/to/downloaded.decisions.json
./transcribe-media --refresh-voices
./transcribe-media --review-speakers
```

Applying a review assigns durable IDs, stores the selected references as
human-verified evidence, and immediately regenerates TXT, JSON and requested
subtitles. It does not rerun ASR or change recognized words. Refresh rematches
cached recordings using verified references, preserving existing human choices.
Strong matches are applied automatically; unresolved matches stay in the queue.
Refresh and review application leave old per-turn tone/acoustic estimates out
when rebuilding turns; a normal processing run recomputes them.

New identities are **never automatically enrolled in quality mode**. Until reviewed,
the transcript is labeled `DRAFT: SPEAKER REVIEW REQUIRED`, and the manifest records
`awaiting_review`. That is a completed processing job awaiting a person, not an
error that should repeatedly run ASR. A weak or unknown voice stays unresolved
instead of being forced onto an adult profile. Human approval identifies who
spoke; it does not certify ASR wording or eliminate acoustic overlap uncertainty.

The new profile must have at least one usable selected clip. Very short/noisy
utterances may need an existing profile or remain unknown. Automatic identity
matching requires stronger support: at least two reference clips of 2.5 seconds
or more from one verified recording condition, adequate query evidence, a strong
score and margin, and agreement across the query's windows. A short child reply
can be manually labeled without being trusted as an automatic identification
reference. Nonverbal vocalizations are not guaranteed to be transcribed.

## How the references improve

Reference extraction keeps real contiguous waveform windows, trims boundaries,
excludes detected overlap and diarizer disagreement, rejects severe clipping and
very low energy, and checks acoustic consistency. It does not splice disjoint
replies into artificial speech. These checks are useful filters, not proof that
a clip has no background noise, music or speaker contamination.

Verified clips remain separate from automatic observations and are never evicted
by the automatic observation limit. Quality matching uses only verified clips.
Automatic matches can be logged as strong observations but cannot silently become
trusted reference material or replace confirmed anchors. This limits the feedback
loop in which an early wrong match makes later wrong matches look stronger.

Each person's verified recordings provide separate condition prototypes. A match
needs support from a condition centroid and two individual clips, plus a margin
over competing people and agreement across query windows. Confirming another
room/microphone/session can add useful coverage without averaging that condition
away. The number of reference conditions can also affect false-match rates, so
the thresholds remain conservative heuristics requiring evaluation. Child-labeled
profiles use a stricter acceptance threshold. Pitch is not used to infer age.

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
