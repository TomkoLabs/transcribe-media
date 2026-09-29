# Reviewed TXT contract for Couples Insight

The upstream quality-review workflow exports UTF-8 text:

```text
ANALYSIS-READY TRANSCRIPT
=========================
Source: synthetic-session.wav
Language: en
Speakers: VOICE_0001 (Wife; adult), VOICE_0002 (Husband; adult)
Words are preserved. Timestamps and speaker review details are in Review.

TRANSCRIPT
----------

Husband [VOICE_0002]:
Synthetic first statement.

Wife [VOICE_0001]:
Synthetic reply.
```

- Only current, finalized saved review state with no unresolved items is READY.
  Otherwise the first line is `DRAFT: SPEAKER REVIEW REQUIRED`. Named profiles
  alone do not finalize review. Browser edits must be exported **and applied**.
- Review status uses both `speaker_review.pending` (voice groups) and
  `pending_turns` (soundbites). The authoritative pending item count is the larger
  queue size; these overlapping queues are not added together. CLI diagnosis,
  refresh, apply, generated page **Saved state**, and TXT readiness share this rule.
- Blocked source provenance also makes TXT DRAFT, even with zero saved pending
  items. Refresh backs up and quarantines changed/unavailable source exports.
  Rerendering checks existing review packets against accessible source media;
  absent/inaccessible media for a packet keeps that presentation DRAFT. Saved
  JSON, words and reference learning are not changed by rerendering.
- The renderer remains usable for legacy non-quality transcription; its heading
  alone is not a new provenance attestation. This contract concerns reviewed
  quality output. Keep the matching Review tree for source validation.
- Reviewed visible labels **Husband** and **Wife** are the canonical identities
  expected by this Couples Insight configuration. Reviewers must assign them
  correctly. `VOICE_*` is opaque provenance to downstream consumers. IDs can
  reverse or be reused across independent projects/recordings; never infer a
  partner from the numeric suffix. Within one upstream registry they are durable
  profile IDs, not a guarantee of identity across registries.
- Context/overlap/vocal-tone annotations are metadata, not dialogue or identity
  evidence. Speech text stays under its reviewed speaker heading.
- Renaming a TXT file or deleting its DRAFT heading cannot finalize it. Transfer
  only READY files after diagnosis, review application and rerendering.

`tests/test_review_recovery.py` renders synthetic READY/DRAFT examples with both
VOICE-ID orders. The existing hub identity gate accepts this structure and rejects
DRAFT even when its headings already contain Husband/Wife. No downstream code
change is needed.
