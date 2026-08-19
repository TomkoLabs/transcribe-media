# transcribe-media

## TL;DR

`transcribe-media` is a private, local-first Linux tool that turns a folder of
audio/video recordings into concise analysis-ready transcripts plus detailed,
timestamped review evidence with anonymous speaker labels, measured acoustic
observations, and approximate vocal-tone estimates.

It uses accuracy-oriented models by default: WhisperX/Whisper `large-v3` for
speech recognition and alignment; pyannote Community-1 plus NVIDIA Sortformer
v2.1 on CUDA installations when configured (or a SpeechBrain fallback) for
speaker separation; ECAPA voice embeddings for confidence-gated label
correction and cross-recording matching; and emotion2vec+ Large for tone
estimation. NVIDIA CUDA is used automatically when available; CPU processing is
fully supported.

One intended use is preparing objective, reviewable source material for later
analysis by a qualified therapist, including couples-therapy review. The output
is machine-generated evidence to verify against the recording—not a diagnosis
or a factual reading of emotion, intent, honesty, or identity.

The normal workflow is deliberately short:

```bash
git clone https://github.com/TomkoLabs/transcribe-media
cd transcribe-media
./install.sh
```

Put recordings in `./Video Source/`, then run this for a known two-person
conversation:

```bash
./transcribe-media --min-speakers 2 --max-speakers 2
```

Concise TXT transcripts appear in `./Transcribed/`. Timestamped detailed TXT,
JSON evidence, processing state, and the private voice registry appear in
`./Review/`.

## What it does

For each supported media file, the pipeline:

```text
FFmpeg decode -> transcription -> word alignment -> diarization/model consensus
              -> confidence-gated speaker refinement
              -> persistent anonymous voice matching
              -> acoustic/tone analysis -> TXT + JSON
```

The source file is opened read-only and is never renamed, moved, modified, or
uploaded to a transcription service. Normal processing happens locally after
the dependencies and model weights have been downloaded.

Example analysis-ready TXT output:

```text
VOICE_0002:
I mean... I—I don't know, you just always...
[Context: elevated volume; overlapping speech]
[Vocal tone estimate: sad]
```

The concise transcript preserves ASR wording and conversational order without
per-turn timestamps or repeated model metadata. Adjacent turns from the same
speaker are joined into bounded paragraphs. Only objective interaction markers
and strong, stable, non-neutral model-based vocal-tone signals are retained;
weak, neutral, unclassified, temporally inconsistent, and heuristic tone
results stay out of this downstream-oriented view.

Default artifacts for `Video Source/clip.mp4` are:

```text
Transcribed/clip.mp4.txt
Review/clip.mp4.detailed.txt
Review/clip.mp4.json
Review/transcription_manifest.json
Review/speaker_registry.json
```

The detailed TXT retains timestamps, all acoustic flags, and complete tone
percentages while naming each model once in its header. JSON includes the
source fingerprint, model/runtime provenance, aligned words, speaker turns,
local and persistent speaker IDs, explicit attribution uncertainty, acoustic
measurements, tone scores, confidence values where available, and processing
limitations. Optional subtitles can be requested with:

```bash
./transcribe-media --review-formats json,srt,vtt
```

## Persistent voices across recordings

Persistent anonymous voice matching is automatic; there is no manual enrollment
step. The first suitable recording creates profiles such as `VOICE_0001` from
clean, non-overlapping recognized speech. Later recordings compare their
speaker clusters with those profiles and reuse the same IDs when the match is
confident. Strong new evidence can improve a profile, and a sufficiently
different new speaker can receive a new ID.

The matching is deliberately conservative. Short, noisy, overlapping,
non-verbal, or ambiguous speech can remain `SPEAKER_XX`/`SPEAKER_UNKNOWN`
instead of being forced onto the wrong person. Persistent IDs are probabilistic
voice-similarity labels—not names, real-world identity, or authentication—and
can still be wrong when microphones, health, age, or recording conditions
change. Existing transcripts are not retroactively rewritten.

The registry is stored at `Review/speaker_registry.json`. It contains biometric
voice embeddings, is created with restrictive permissions, and is ignored by
Git. Preserve the registry and manifest together to retain IDs across future
runs; protect them as private data. A number such as `VOICE_0006` is a durable
profile number, not proof that six people occur in that recording. Each TXT
header lists the active voices for that file.

To intentionally rebuild all learned IDs from the media currently in
`Video Source`, run:

```bash
./transcribe-media --reset-speaker-registry --min-speakers 2 --max-speakers 2
```

The previous registry and manifest are archived under
`Review/speaker-registry-backups/` before processing. Use reset only when you
intend to start the project voice history again; do not use it for normal runs.

## Speaker-count guidance

Providing the expected count improves diarization by removing a major source of
over- or under-splitting:

```bash
# Exactly two speakers
./transcribe-media --min-speakers 2 --max-speakers 2

# Usually two, but sometimes a meaningful third speaker
./transcribe-media --min-speakers 2 --max-speakers 3
```

If the number is unknown, omit both options. `--speakers 2` is a shorter
equivalent for an exact count. Count everyone whose speech should receive a
separate label, including a child who speaks meaningfully.

For the best overlap-aware separation, configure a Hugging Face read token for
pyannote Community-1 after accepting its model terms:

```bash
./transcribe-media --configure
```

Without a token, the installed SpeechBrain ECAPA fallback remains fully local
and functional. On an NVIDIA installation, Sortformer can operate without
pyannote, but the two-model ensemble is preferred when the token is available.

The default CUDA installation includes pinned NeMo `2.7.3` and NVIDIA's
four-speaker Sortformer v2.1. In automatic mode, Community-1 remains the primary
timeline and Sortformer supplies an independent second opinion mapped into the
same recording-local labels. Disagreements are retained in JSON instead of
being silently forced. Use `--diarization-backend pyannote` to compare the
single-model result. Sortformer is primarily English and supports at most four
speakers.

After diarization, the default acoustic refinement pass rechecks short,
non-overlapping label changes against clean voice prototypes from the complete
recording. It changes a label only when the neighboring speaker is supported by
a strong similarity margin; ambiguous turns and genuine overlap are preserved.
For English, an extra narrow rule also checks label changes at near-zero-gap
sentence seams, including a mislabeled continuation lasting several seconds.
It also embeds a complete aligned utterance when individual words form a
fragmented A-B-A label pattern, allowing pieces too short to classify alone to
be judged together.
JSON retains the original diarization speaker, the corrected assignment, and
the scores used for the decision. Disable this conservative pass when comparing
raw diarization behavior:

```bash
./transcribe-media --no-speaker-refinement --min-speakers 2 --max-speakers 2
```

## Installation and operation

The supported target is Debian-family Linux. Installation needs internet access
and enough disk space for the Python environment and model weights:

```bash
./install.sh
```

The installer creates a managed Python 3.11 environment, installs/checks FFmpeg,
selects compatible CPU or NVIDIA CUDA packages, installs pinned ML dependencies,
creates the working directories, and validates the models. Useful diagnostics:

```bash
./transcribe-media --doctor
./transcribe-media --prepare-models
```

English is the primary validated use case. Automatic language detection remains
the default for general use; add `--language en` when every recording is known
to be English. Other languages can be transcribed in place or translated to
English with `--task translate`.

Common operations:

```bash
./transcribe-media --recursive
./transcribe-media --language en
./transcribe-media --task translate
./transcribe-media --overwrite
./transcribe-media --dry-run
./transcribe-media --no-speaker-identity
./transcribe-media --no-acoustic --no-tone
```

Run `./transcribe-media --help` for all advanced model, device, directory, and
speaker-matching controls.

Normal reruns are incremental. A completed file is skipped only when its source
fingerprint, program version, settings, models, and required outputs still
match. Outputs and the manifest are written atomically; failures and
interruptions are retried later. One bad media file does not stop the remaining
batch.

On GPUs below 16 GiB, including a 10 GiB RTX 3080, large-v3 transcription and
alignment stay CUDA-accelerated while speaker/tone analysis runs on CPU to avoid
VRAM contention. This does not substitute smaller models. CUDA out-of-memory
errors are retried with smaller ASR batches.

## Privacy and limitations

- `Video Source`, `Transcribed`, and `Review` are ignored by Git. The tracked
  `Video Source/.gitkeep` only preserves the empty input directory.
- Do not publish recordings, transcripts, or voice registries without the
  necessary rights, consent, and privacy review.
- Speech recognition, alignment, diarization, persistent matching, and tone are
  estimates. Noise, music, crosstalk, short turns, and domain/language mismatch
  reduce accuracy.
- Keep speaker labels for downstream structure, but treat them as probabilistic
  attribution and verify consequential quotations against the recording. Text
  alone is not reliable enough for an LLM to reconstruct who spoke each turn.
- Condensed TXT marks evidence conflicts as `speaker attribution uncertain`
  rather than presenting every boundary as equally reliable. The assigned ID
  remains visible, and the detailed JSON retains the reasons and model evidence.
- `Observed` annotations are measured timing/acoustic features. `Tone approx`
  is uncertain vocal-presentation classification; `unclassified` is a valid
  model abstention, not a processing failure.
- Generated material requires human review before clinical, legal, employment,
  safety, or other consequential use.
- Sortformer model weights are downloaded separately under the NVIDIA Open
  Model License; this repository does not redistribute them.

## Development

```bash
python -m unittest discover -s tests -v
bash -n install.sh transcribe-media
python -m compileall -q transcribe_media.py transcribe_media_app tests
```

## License

The project source code is licensed under the [MIT License](LICENSE).
Third-party applications, Python packages, and downloaded model checkpoints are
not relicensed by this project and remain subject to their own licenses, access
conditions, and acceptable-use terms. No model weights, source recordings,
transcripts, or voice registries are included in the repository.
