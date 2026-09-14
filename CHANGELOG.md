# Changelog

## 1.14.2 — 2026-09-12

- Removed repeated `[speaker attribution uncertain]` tags from primary transcripts
  and shortened the introduction. Attribution diagnostics stay in detailed review
  files, with one draft notice in the primary TXT for unresolved identities.
- Confidence changes no longer split adjacent speech by the same person into
  separate paragraphs. Speaker changes, long pauses and separate UNKNOWN turns
  remain distinct. Pauses use actual timestamps even after speaker review removes
  old acoustic annotations; recognized words and identity decisions are preserved.
- Added `--render-transcripts` to update existing text/subtitle exports from
  canonical JSON without inference, voice rematching or profile changes. It
  validates saved results and rolls back the whole export batch on write failure.
- Maintenance commands now reject conflicting actions and transcription-only
  dry runs, avoiding silently ignored flags.

Run `--render-transcripts` after updating to clean existing transcripts. Normal
quality defaults, model choices and hardware requirements are unchanged. Tests
cover rendering and recovery; target-GPU acceptance and measured accuracy remain
outside this presentation update.

## 1.14.1 — 2026-09-12

- Listening adds 2 seconds of context on either side, or 4 seconds for uncertain
  speech, with 5-second and exact-selection options. A live clock and target
  highlight distinguish the selected speech from neighboring voices. Correction
  and training intervals stay unchanged.
- Partial corrections have preview, start/end-at-playhead controls and playback
  for saved ranges. Guidance explains uncertain timing and word-based assignment.
- Group completion follows reviewed spoken words, so silent gaps between clips
  no longer keep fully assigned speech pending. Reference progress is separate;
  applying a sample's person to remaining speech still requires explicit approval.
- Download JSON is the primary save path, with clear project-folder/apply steps.
  Optional direct saving detects unavailable APIs and handles browser denial or
  cancellation without losing the draft.
- Split the review template into maintainable source assets while still embedding
  everything into one offline page. Shortened the README and moved detailed
  hardware/troubleshooting instructions into the quality guide.

Regenerate existing pages with `./transcribe-media --review-speakers` and reopen
HTML. No reinstall, retranscription, new model or additional GPU requirement.
Synthetic workflow/browser checks do not establish model accuracy or target-GPU
performance; hardware acceptance remains pending.

## 1.14.0 — 2026-09-11

- Aggregated offline review across all processed recordings, with one shared
  person catalog, recording navigation, saved drafts and one batch decisions JSON.
- Batch application creates shared profiles once, applies manual corrections,
  then rematches all cached transcripts using the final references. The entire
  update rolls back on failure, and repeated/revised imports preserve identity IDs.
- Replaced the assumed Downloads path with project-relative `speaker-decisions/`.
  Added optional browser folder saving, explicit download fallback instructions,
  and useful errors for misplaced decision files.
- Readable transcript turns and subtitles now show profile labels alongside
  durable VOICE IDs. JSON IDs and source-based transcript filenames stay stable.
- Added a confirmed UNKNOWN exclusion that preserves words without voice training
  or repeated identity-review requests. The existing unresolved choice remains.

No new models, inference hardware requirements, server or runtime dependencies.
Regenerate existing pages with `--review-speakers`; batch apply includes the final
cached voice refresh, without retranscription. Synthetic tests validate workflow
and recovery, not measured identification accuracy or GX10 speed.

## 1.13.0 — 2026-09-11

- Rebuilt offline speaker review with distinct clip cards, current identities,
  confident preselection, an uncertainty-first queue, and clip/turn/time corrections.
- Added saved-review import, local drafts, profile label/role editing, draft-person
  removal, archiving, and a copyable apply command. No web service is required.
- Human labels now save even without usable reference audio. Separated transcript
  correction from training readiness and removed an overly strict ASR coverage
  check that rejected reviewed speech windows containing pauses.
- Consistent sets of short, verified clips can support matching. Disputed model
  windows remain manual-only evidence; automatic matches cannot verify themselves.
- Preserved draft-to-person mappings on reapply/rerun/merge, supported restoring
  older decision snapshots, and added transactional propagation of profile edits.
- `--review-speakers` refreshes existing pages and the profile catalog without ASR.
  Old decision exports remain supported when their source and boundaries match.

This update changes review and reference handling, with no larger model, Ollama
service or additional GPU requirement. Existing 3080/GX10 hardware targets and
accuracy-validation limits remain unchanged. Model-free regression tests and a
synthetic browser workflow validate behavior, not measured transcription or
speaker-identification accuracy.

## 1.12.0 — 2026-09-11

Quality is now the default for `./transcribe-media`, including sequential
large-v3 decoding, word alignment, a 2–3 speaker range, reviewed voice references,
and no inferred vocal tone. `--quality` remains compatible; `--no-quality`
explicitly restores the legacy workflow. Translation or disabling required
quality stages now needs that explicit opt-out.

- Simplified Debian/RTX 3080 installation and first-run documentation; the
  installer prepares the default quality models without extra flags.
- Added model-access links to token setup and review instructions after install.
- GPU installation stops on CUDA failure. GPU model preparation and doctor require
  CUDA; doctor respects explicit device settings and shows the selected mode.
- Added GitHub CI for Debian 12/13, a shared check script, minimal test requirements,
  and maintainer release/hardware validation instructions.
- Removed the obsolete `requirements-emotion.txt` compatibility file; the default
  requirements already include those dependencies. Historical work notes and
  local media/runtime state are excluded from the release tree.

### Quality and reliability improvements included since 1.9.4

- Offline speaker review with playback, candidate similarities, transcript/time
  context, clean reference selection, turn/range corrections, and audited imports.
- Protected human-reviewed voice anchors, recording-condition matching,
  conservative automatic assignment, mixed-group review, and explicit duplicate
  profile merges. Child roles are human annotations with stricter matching.
- Full ASR decoding fallback and retained diagnostics, ASR/alignment caching,
  and cached speaker rematching without retranscription.
- Full source hashing, private atomic state writes, project locking and crash
  recovery across registry/manifest/transcript updates.
- Better overlap/uncertainty reporting, guarded speaker refinements, subtitle
  edge cases, and more appropriate acoustic measurements.
- GX10/GB10 preflight checks and a local CUDA-enabled CTranslate2 build path.

### Upgrading and validation limits

Back up the complete `Review/` folder and media before upgrading. Legacy IDs
are preserved as candidates; automatic quality matches require human-reviewed
reference audio. Review existing profiles instead of creating replacements.
Version changes may reprocess files; human choices are retained when the source
and local segmentation remain compatible. Changed segmentation requires fresh
review and will not silently overwrite previous choices.

CPU tests and a local sample validate the implemented workflow. RTX 3080/GX10
end-to-end acceptance and independently labeled transcription/diarization/identity
benchmarks remain outstanding. See `CONTRIBUTING.md` before publishing hardware
or accuracy claims.
