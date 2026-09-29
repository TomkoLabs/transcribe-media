# Development validation: 1.16.0.dev1

Performed in the development workspace, using synthetic sources only. Python
3.11.16, existing torch 2.8.0+cpu environment; no packages/models downloaded.

- `PYTHONPATH=tests "$TEST_PYTHON" -m unittest test_review_recovery -v`:
  **15 tests passed**. Includes changed-source isolation, metadata equivalence,
  explicit relocation/ambiguity, obsolete and fresh decisions, UI/CLI snapshots,
  DRAFT/READY and reversed IDs, journal recovery, idempotence, unrelated state and
  registry preservation, single-source mocked transcription, partial batch apply,
  backup failure, and source mutation during recovery.
- `TRANSCRIBE_TEST_PYTHON="$TEST_PYTHON" bash scripts/check.sh`:
  **210 Python tests and 32 Node tests passed**. The script also passed Bash syntax,
  ShellCheck, and Python compileall. Expected synthetic failure-path messages in
  the output are test coverage, not suite failures.
- `git diff --check`: passed. Changes limited to upstream review recovery,
  source selection, readiness, offline UI, version, regression tests and docs.
- Git-free source copy: actual launcher `--help`, `--version`, and
  `bash install.sh --help` passed, reusing an existing Python environment.
  Installer control flow is covered with isolated mocked runtime tools in the
  full suite. There is no wheel/container packaging, Python formatter or static
  type-check configuration in this repository. A real GPU installer was not run.
- Actual existing hub `text_ingest.parse(render_txt(payload))`: **four cases
  passed**, two READY examples with reversed VOICE IDs resolving canonical
  Husband/Wife roles and both corresponding DRAFT examples remaining gated.
  Hub and analysis repositories were not modified.
- Browser smoke check of a synthetic two-recording index: blocked source/action,
  saved pending count, snapshot/time/version and exclusion from exports visible;
  current recording selectable and usable; no browser console errors. Node tests
  verify blocked-source export/import behavior. No private runtime UI was used.

The test Python was the existing Python 3.11 environment in the local
`transcribe-media-local-backup-20260911T164305021017Z` directory. Test artifacts
and logs were kept under `/tmp/transcribe-recovery-*`, outside source control.

No Kraken access, actual source recovery, GPU transcription, GX10, model inference,
model changes, private media inspection, or running-demo changes occurred. Real
Kraken and hardware acceptance remains required; see [the runbook's release
criteria](REVIEW_RECOVERY.md#release-acceptance-still-required). No stable tag.
