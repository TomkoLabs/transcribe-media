# transcribe-media

Local transcription with persistent speaker identities and an offline review
page. Optimized for **two adults, with an occasional child**, with quality over
speed. **Quality mode is the default:** Whisper large-v3, word alignment,
speaker diarization and conservative matching against reviewed voice references.
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

1. Start with **Needs attention**. Confident matches are already assigned.
   Listen, then select Adult A, Adult B, Child, or the same existing person across
   recordings. **People & profile labels** manages the shared names.
2. Correct exceptions using **Change only this clip** or **All soundbites**.
   Listen includes surrounding audio, with extra context for uncertain clips.
   Partial corrections have preview and playhead controls. Uncheck noisy/mixed
   training clips; choose **UNKNOWN — exclude from voice learning** for passages
   you want to leave anonymous without further identity review.
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

```bash
git pull --ff-only
./transcribe-media --render-transcripts
./transcribe-media --review-speakers
```

This updates saved text/subtitles and review pages without running models.
Words, speaker assignments and profiles stay unchanged. Reopen the regenerated HTML.
No reinstall is needed for this update.
Run `./install.sh` again when a future update changes model/dependency requirements.
Back up first: version/settings changes can cause normal processing to rerun.

## Useful commands

| Need | Command |
| --- | --- |
| Rematch cached transcripts after other profile improvements; wording stays unchanged | `./transcribe-media --refresh-voices` |
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

Collect clean reviewed clips from several sessions for each person. Similarity
scores are not probabilities, and adult/child roles are supplied by you. Noise,
short replies and overlapping speech can still confuse the models. Verify
consequential words and speaker assignments against the recording before use
in therapist analysis; the script provides no therapeutic interpretation.

The 3080 is the intended quality target; this update needs no larger model or GPU.
GX10 has a separate ARM/CUDA installation path. End-to-end GPU acceptance and
measured accuracy benchmarks remain outstanding; installer checks must pass on
your machine. Ollama is not used. See [hardware and troubleshooting](QUALITY_GUIDE.md#hardware-and-models).

See [QUALITY_GUIDE.md](QUALITY_GUIDE.md) for review details, profile repair, backups,
advanced options and model limits; [CONTRIBUTING.md](CONTRIBUTING.md) for checks;
and [CHANGELOG.md](CHANGELOG.md) for releases. Source is [MIT licensed](LICENSE);
third-party models and packages retain their own licenses and access terms.
