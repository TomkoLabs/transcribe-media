# Development and releases

Use Python 3.11 on Linux. The standard installer supplies the full environment;
run the checks with:

```bash
sudo apt-get install -y shellcheck nodejs
./scripts/check.sh
```

For tests alone, no model downloads, Hugging Face token, GPU or therapy recordings
are required. In a clean Python 3.11 environment:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
python -m pip install -r requirements-test.txt
TRANSCRIBE_TEST_PYTHON=python ./scripts/check.sh
```

The OS also needs FFmpeg, ShellCheck and Node.js 18 or newer. Node is used only
for review-state tests; normal installation and transcription do not need it.
Tests use temporary directories,
synthetic audio/embeddings and mocked model interfaces. They exercise CLI
processing, first enrollment, speaker correction/merge/rematch, incremental
reruns, cache integrity, state recovery, CUDA selection, and installer failures.
The offline UI embeds `review_state.js`, `review_io.js` and `review_page.html` from the Python
package. Run `node tests/test_review_state.cjs` for draft/import/clip-range tests.
Use synthetic recordings for browser checks of filters, playback, labels and export.
Batch tests cover shared enrollment, final rematching, UNKNOWN exclusions,
alias recovery, relative decision paths and rollback of the whole batch. Browser
folder-saving logic is tested with fake file handles; manually check the native
folder picker in a supporting browser before claiming a specific browser is supported.
The GitHub Actions workflow runs the same checks in Debian 12 and 13 containers.
GitLab CI is retained for the existing mirror.

## Before a release

Update `transcribe_media_app/__init__.py`, `CHANGELOG.md` and any changed behavior
in the README/quality guide. Run `./scripts/check.sh` and inspect `git diff --check`.
The shared version is included in cache/output fingerprints, so version changes
can invalidate prior processing results.

From a fresh Debian checkout on an RTX 3080, follow the README exactly: accept
Community-1 access, run `./install.sh`, add a consented test recording, and run
`./transcribe-media` with no mode/backend options. Installation must finish its
CUDA/model self-tests. Confirm that the run reports quality mode, CUDA ASR,
CPU analysis and Community-1/Sortformer with the recommended token setup.

Use at least two recordings of the same people from different sessions. Include
a short third-speaker turn in one recording. Review the first profiles, import
the decisions, refresh other recordings, and check stable IDs against listening.
Also check unchanged reruns skip processing, review imports do not rerun ASR,
and freezing learning leaves the registry unchanged during normal processing.
Check important words and speaker turns against the audio; self-tests do not
measure accuracy. Record the driver, hardware, package versions, runtime/backend
and any fallback in release notes without including private material.

For GX10 support claims, independently repeat installation and real inference
on GB10/DGX OS with a supported CUDA toolkit. The ARM CTranslate2 source build
and preflight checks cannot establish hardware compatibility by themselves.
Until those runs are recorded, describe both GPU platforms as targets with
hardware acceptance pending.

## Publishing on GitHub

This checkout may point at the existing GitLab origin. Inspect `git remote -v`;
add a GitHub remote for the repository you own if needed, without replacing a
working mirror unintentionally. The README clone URL targets
`TomkoLabs/transcribe-media`; adjust it if publishing elsewhere.

Commit only source, documentation, scripts, dependency lists and tests. Keep
recordings, transcripts, review folders, voice embeddings, exported decisions,
model caches and tokens out of commits and release attachments. `.gitignore`
protects the default paths; custom output paths must be handled separately.
Do not blindly add unrelated local notes to the release.

Push the reviewed commit to the chosen GitHub repository and verify the **CI**
workflow passes. Create an annotated version tag on that exact commit and publish
release notes from the changelog, stating any hardware validation still pending.
Use GitHub's source archives; do not archive the working directory containing
private media. Release publication is a maintainer action, not part of running
`install.sh` or the check script.
