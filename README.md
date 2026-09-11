# transcribe-media

Private, local transcription with persistent speaker identities and an offline
voice-review page. Designed for **two adults, with an occasional child**, with
quality preferred over speed. Audio stays on the machine running the script.

**Quality mode is the default.** After installation, the everyday command is:

```bash
./transcribe-media
```

It uses Whisper **large-v3**, word alignment, speaker diarization, and conservative
matching against human-reviewed voice references. You do not need model, GPU,
batch-size, speaker-count, or quality flags for the baseline case.

## Quick start: Debian + RTX 3080

### 1. Prepare the machine and model access

Use Debian 12/13 on an x86-64 PC with a working NVIDIA driver. Run `nvidia-smi`;
it must list your RTX 3080. The installer supplies the Python/CUDA libraries but
**does not install the GPU driver**. A separate CUDA toolkit is unnecessary on
this x86-64 path. See [NVIDIA's driver compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
if the runtime reports an insufficient driver.

Plan for **32 GB system RAM and 30 GB free disk** as a practical starting budget,
plus recordings, model caches, transcripts and review audio. These are planning
estimates, not measured minimums. An RTX 3080 with 10 or 12 GB VRAM is the target.

For the recommended speaker models, sign into Hugging Face and:

1. Accept the access conditions for [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
2. Create a [read access token](https://huggingface.co/settings/tokens). Paste it
   into the installer's hidden prompt when asked. Do not put it in source files.

This enables downloading the diarizer for local use; it does not send your
recordings to Hugging Face.

### 2. Install once

Clone this repository using its GitHub **Code → HTTPS** URL, then enter its
`transcribe-media` directory. If you already have the checkout, skip cloning.
The intended GitHub location is shown below; use your own URL for a fork/mirror.

```bash
sudo apt-get update
sudo apt-get install -y git

git clone https://github.com/TomkoLabs/transcribe-media.git
cd transcribe-media
./install.sh
```

Run the installer as your normal user; it requests `sudo` for system packages.
It installs FFmpeg, creates its own Python 3.11 environment, installs the pinned
ML stack, prompts for the token, downloads the default models, and runs local
inference checks. Internet access is needed for setup. Allow time for large
downloads; no separate `pip`, virtual-environment activation, model preparation,
or CUDA flags are needed.

If NVIDIA is detected but CUDA installation or validation fails, installation
stops with an error. Fix that error before processing. If no working driver is
found, the installer explicitly warns that it is installing CPU mode.

### 3. Add recordings and run

Put audio/video files into **`Video Source/`**, then run:

```bash
./transcribe-media
```

Read transcripts in **`Transcribed/`**. The terminal prints progress and a link
path to **`Review/speaker-reviews/index.html`**. Processing is incremental: add
more recordings and run the same command again. Completed, unchanged files are
skipped; files awaiting human review do not repeatedly rerun transcription.

### 4. Confirm the initial voices

Open **`Review/speaker-reviews/index.html`** in your own browser, then open a
recording's review page. No web server is required.

- Listen to several reference clips for each local speaker group. The page shows
  timestamps, transcript turns, likely existing profiles and similarity scores.
- On the first recording, choose **Create Adult A**, **Create Adult B**, and
  **Create Child** if applicable. On later recordings, choose the existing
  person's `VOICE_…` ID. Avoid creating another profile for the same person.
- Uncheck unsuitable reference clips. If a group contains two people, expand its
  turns and correct those turns or time ranges before accepting the references.
- Click **Export decisions**, then apply the downloaded JSON:

```bash
./transcribe-media --apply-speaker-review "$HOME/Downloads/REPLACE_WITH_EXPORTED_FILENAME.decisions.json"
```

Use the actual exported filename. The command saves the verified profiles and
updates that recording's transcript immediately, without rerunning ASR. If you
already processed other recordings, apply the improved profiles to them with:

```bash
./transcribe-media --refresh-voices
```

Reopen the review index for remaining uncertain voices. New recordings use the
verified profiles automatically. The first recordings need more review; coverage
across microphones, rooms and sessions matters more than repeatedly confirming
the same clip. Scores are **similarities, not calibrated probabilities**.

Unresolved voices stay marked **`DRAFT: SPEAKER REVIEW REQUIRED`**. An occasional
child is allowed as a third speaker, but age is not inferred automatically.
Human review of speaker identity does not certify every recognized word.

## Defaults and the few useful options

| Need | Command |
| --- | --- |
| Normal quality run; two adults and possibly a child | `./transcribe-media` |
| Exactly two people speak in every recording | `./transcribe-media --speakers 2` |
| One-person recordings or excerpts | `./transcribe-media --speakers 1` |
| Include subdirectories | `./transcribe-media --recursive` |
| A different spoken language | `./transcribe-media --language fr` (or `--language auto`) |
| Freeze automatic profile learning after sufficient review | `./transcribe-media --no-speaker-learning` |
| Check installation and require working CUDA | `./transcribe-media --doctor --device cuda` |

Do not use `--speakers 2` when a child might speak. The default range is **2–3**;
if some files contain only one speaker, use `--min-speakers 1 --max-speakers 3`.
Freezing preserves conservative matching and leaves uncertain voices for review.
Explicit human review imports can still update a frozen registry.

With the recommended NVIDIA installation and token, automatic backend selection
uses **Community-1 plus Sortformer** as a second opinion. Without NeMo it uses
Community-1 alone. Without a token it can use Sortformer, or the weaker SpeechBrain
window-clustering fallback. The selected backend appears in the run output and
JSON. Configure the token to follow the recommended quality setup.

Inferred vocal tone is off by default. Measured timing/acoustic observations
remain available. All advanced options are under `./transcribe-media --help`;
review, duplicate-profile repair, evaluation, and model tradeoffs are explained
in [QUALITY_GUIDE.md](QUALITY_GUIDE.md).

## Outputs, reruns and backups

| Location | Contents |
| --- | --- |
| `Video Source/` | Your original media; never modified by processing |
| `Transcribed/` | Readable TXT transcripts with speaker turns and uncertainty |
| `Review/` | Detailed JSON, manifest, persistent voice registry, ASR cache and review pages/audio |

JSON is always generated; add `--review-formats json,srt,vtt` for subtitles.
Default paths belong to the checkout, even when invoking the installed
`transcribe-media` command from another directory. Passing another source folder
alone still uses this checkout's output directories and voice registry.

**Back up `Review/` together with the original recordings and `Transcribed/`.**
The registry, manifest and review packets belong together. Do not delete just the
registry to fix duplicate people; use the merge workflow in the quality guide.
Review application validates the original source, so keep its location/content
unchanged while reviewing. Review audio adds approximately 115 MB per hour.
Keep each review HTML beside its WAV if copying the review folder to a desktop.

Updates use `git pull --ff-only` followed by `./install.sh`. Ignored recordings
and voice state are preserved. Program/model changes can require reprocessing;
back up first. Versions before 1.12 required `--quality`; it is now automatic.
Old automatic profiles remain candidates but need human-reviewed reference clips
before the quality matcher trusts them. `--no-quality` explicitly restores the
legacy batched transcription and automatic-enrollment behavior; it is also
required for translation or disabling alignment/speaker identity.

## Hardware and troubleshooting

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

## Privacy, limitations and development

Normal processing uses local model caches. Setup/model preparation downloads
weights; the launcher disables supported telemetry. Review pages contain local
scripts and audio, with no hosted service. Treat recordings, review audio,
transcripts, embeddings and decision exports as private data; default working
directories and decision exports are ignored by Git.

Short replies, noise and simultaneous speech remain difficult. This version
labels speakers but does not extract separate audio tracks for overlapping
voices; it downmixes input to mono. Verify consequential quotations and speaker
attribution against the recording before downstream therapist analysis. The
software performs no therapeutic interpretation or diagnosis.

See [CONTRIBUTING.md](CONTRIBUTING.md) for tests and release validation and
[CHANGELOG.md](CHANGELOG.md) for changes. Source is [MIT licensed](LICENSE);
third-party packages and model weights retain their own licenses/access terms.
