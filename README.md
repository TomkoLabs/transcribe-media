# transcribe-media

Local transcription with persistent speaker identities and an offline review
page. Optimized for **two adults, with an occasional child**, with quality over
speed. **Quality mode is the default:** Whisper large-v3, word alignment,
speaker diarization, conservative matching against reviewed voice references,
and selective vocal-emotion estimates.
Audio stays on the machine running the script.

## Quick start: Debian + RTX 3080

### 1. Prepare once

- Debian 12/13, x86-64, with a working NVIDIA driver: `nvidia-smi` must list your
  GPU. The installer supplies Python/CUDA libraries, but does not install the
  driver. No separate CUDA toolkit is needed on this PC setup.
- RTX 3080 with 10 or 12 GB VRAM is the target. Budget roughly 32 GB system RAM
  and 30 GB free disk, plus recordings and outputs; these are planning estimates.
- Accept [pyannote Community-1 access](https://huggingface.co/pyannote/speaker-diarization-community-1)
  and create a [Hugging Face read token](https://huggingface.co/settings/tokens).
  Paste the token into the installer's hidden prompt. Model downloads need
  internet access; your recordings are processed locally.

### 2. Install

```bash
sudo apt-get update
sudo apt-get install -y git

git clone https://github.com/TomkoLabs/transcribe-media.git
cd transcribe-media
./install.sh
```

Use your repository URL if using a fork/mirror; skip cloning for an existing
checkout. Run the installer as your normal user. It requests `sudo` for system
packages, installs FFmpeg and its own Python environment, downloads the models,
and checks local inference. No separate `pip`, environment activation or model
flags are needed. If GPU validation fails, fix the reported error before running.

### 3. Transcribe

Put all your audio/video files in **`Video Source/`**, then run:

```bash
./transcribe-media
```

Read transcripts in **`Transcribed/`**. Add more files and run the same command
again; completed, unchanged files are skipped. No best first recording is needed.
The default allows **2–3 speakers**. Use `--speakers 2` only if exactly two people
speak in every file; use `--min-speakers 1 --max-speakers 3` for batches that also
contain solo recordings.

### 4. Review and apply once

Open **`Review/speaker-reviews/index.html`** in your browser. No server is needed.

1. Start with **Needs attention**, then **Next soundbite needing attention**.
   Confident, unflagged speech is already assigned. Choose the person for each
   queued soundbite; verified-reference matches and reasons explain remaining
   uncertainty. Timing/text notes alone do not require choosing the speaker again.
   **People & profile labels** manages shared names and adult/child roles across recordings.
2. A **Default person for this group** assigns unflagged speech and preserves
   exceptions for individual review. Click words to listen at their aligned times;
   **Listen to source sentence** provides the recognizer's original audio range.
   Expand **Voice learning reference clips** to approve clean training samples.
   Use **UNKNOWN — exclude from voice learning** for intentionally anonymous speech.
3. **Download JSON**, then move that file into **`speaker-decisions/`** inside
   this project. Keep its filename. Download location depends on your browser.
4. **Copy apply command** and run it from the project folder. For example:

```bash
./transcribe-media --apply-speaker-review "speaker-decisions/batch-REPLACE_WITH_EXPORTED_ID.decisions.json"
```

Use the actual filename in the copied command. **One batch apply updates all
reviewed transcripts, learns from approved clips, and rematches cached recordings.**
It preserves manual choices and recognized words; no retranscription or separate
refresh is needed. Reopen the page to see remaining uncertainty.

Turns show **`Adult A [VOICE_0001]:`**, without repeated uncertainty notes.
Transcript filenames stay unchanged. Unresolved voices keep one draft notice
at the top; detailed diagnostics stay in `Review/`. **Import saved review** resumes an
exported file; local browser drafts also save when available. Downloading alone
does not update transcripts. Optional direct folder saving is under **Apply
command & optional folder saving**; use the download steps if the browser blocks it.

## Update an existing installation

For **1.16.0.dev1**, first back up the runtime directories as described in the
[review recovery runbook](docs/REVIEW_RECOVERY.md), then:

```bash
git pull --ff-only origin main
./transcribe-media --diagnose-reviews
./transcribe-media --refresh-voices
```

Refresh isolates changed/unavailable recordings and regenerates current reviews
for the others. Full-content-equivalent metadata changes need no ASR. Use
`--recover-review 'exact-source-key' --dry-run` to inspect targeted recovery before
applying it. Real content changes require targeted `--only-source` transcription
on the operator machine. Recovery preserves private backups and the registry.
No reinstall, model download or broad retranscription is needed for this update.
Read [the finalized TXT contract](docs/TXT_CONTRACT.md) before downstream transfer.
Reopen the generated index and check Saved state and snapshot time; browser drafts
are unapplied edits, not the CLI's saved pending count.

`--refresh-context` updates acoustic/emotion notes; it does **not** rematch voices.
If you have not yet enabled the emotion context introduced in 1.15.0, run
`./transcribe-media --prepare-models` online, then `./transcribe-media --refresh-context`.
For just the current UI, use `./transcribe-media --review-speakers`.

Apply pending exports before a normal processing run: version/settings changes
can regenerate recording evidence and review IDs. Such runs normally reuse the
ASR cache but still rerun speaker analysis. Do not use `--overwrite` just to upgrade;
it runs fresh ASR and can invalidate old time-based decisions. Back up before updating.

## Useful commands

| Need | Command |
| --- | --- |
| Rematch cached transcripts after other profile improvements; wording stays unchanged | `./transcribe-media --refresh-voices` |
| Disable vocal-emotion estimates for new processing | `./transcribe-media --no-tone` |
| Freeze automatic profile learning; explicit reviews still apply | `./transcribe-media --no-speaker-learning` |
| Include source subfolders | `./transcribe-media --recursive` |
| Change language | `./transcribe-media --language fr` or `--language auto` |
| Diagnose GPU setup | `./transcribe-media --doctor --device cuda` |

## Files and quality notes

- **`Video Source/`**: originals, never modified by processing.
- **`Transcribed/`**: readable transcripts.
- **`Review/`**: detailed JSON, voice registry, caches and review HTML/audio.
- **`speaker-decisions/`**: exported manual decisions.

**Back up all four together.** They contain private material and are ignored by
Git. Keep originals at the same location while reviewing, and keep review HTML
beside its WAV files when copying to another computer. Custom source folders
still use this checkout's output directories and shared registry by default.

Collect clean reviewed clips from several sessions for each person. The reference
clip counter in a review counts that recording only; earlier references are retained.
The same known person can match several non-overlapping detected groups. Similarity
scores are not probabilities, and adult/child roles are supplied by you. Noise,
short replies and overlapping speech can still confuse the models. Verify
consequential words and speaker assignments against the recording before use
in therapist analysis; the script provides no therapeutic interpretation.

Vocal-emotion labels appear only for adult profiles and sufficiently clear,
consistent speech. Silence means the model abstained, not that the speaker was
neutral. These are estimates of vocal expression, not psychological assessments.

The 3080 remains the quality target. Emotion analysis runs on CPU by default and
adds processing time; this update does not require more GPU memory.
GX10 has a separate ARM/CUDA installation path. End-to-end GPU acceptance and
measured accuracy benchmarks remain outstanding; installer checks must pass on
your machine. Ollama is not used. See [hardware and troubleshooting](QUALITY_GUIDE.md#hardware-and-models).

See [QUALITY_GUIDE.md](QUALITY_GUIDE.md) for review details, profile repair, backups,
advanced options and model limits; [CONTRIBUTING.md](CONTRIBUTING.md) for checks;
and [CHANGELOG.md](CHANGELOG.md) for releases. Source is [MIT licensed](LICENSE);
third-party models and packages retain their own licenses and access terms.
