"""Private, offline speaker review with source-bound, auditable decisions."""
from __future__ import annotations

import copy
import html
import json
import math
import os
import re
import wave
from pathlib import Path

from .analysis import annotate_speaker_attribution, build_turns, normalize_segments
from .renderers import write_outputs
from .speakers import (VoiceRegistry, SpeakerRegistryError, apply_speaker_identities,
                       overlapping_speaker_pairs, SPEAKER_EMBEDDING_MODEL, SPEAKER_EMBEDDING_REVISION)
from .storage import (ManifestStore, StateTransaction, atomic_write_json, atomic_write_text,
                      source_fingerprint, stable_hash, utc_now)
from .schema import SAMPLE_RATE


def packet_path(review_dir, source_key):
    return Path(review_dir) / "speaker-reviews" / (stable_hash(source_key)[:24] + ".json")


def packet_digest(packet):
    try:
        return stable_hash({key: packet[key] for key in
                           ("source", "fingerprint", "local_result", "evidence", "settings_hash", "embedding_model")})
    except (KeyError, TypeError) as exc:
        raise SpeakerRegistryError("review packet is incomplete or obsolete; regenerate it before applying decisions") from exc


def _segmentation_key(result):
    return stable_hash([{key: turn.get(key) for key in ("id", "start", "end", "local_speaker", "speaker", "text")}
                        for turn in build_turns(normalize_segments(result))])


def apply_turn_choices(result, turns, review_id, receipt=None):
    def choice_at(local, start, end):
        middle = (start + end) / 2
        return next((turn.get("review_choice") for turn in turns
                     if turn["start"] <= middle <= turn["end"]
                     and str(turn.get("local_speaker") or turn["speaker"]) == local), None)

    for segment in result.get("segments", []):
        for unit in segment.get("words") or [segment]:
            local = str(unit.get("local_speaker") or segment.get("local_speaker") or unit.get("speaker") or segment["speaker"])
            if local == "SPEAKER_UNKNOWN" and segment.get("local_speaker"):
                local = str(segment["local_speaker"])
            if unit.get("start") is None or unit.get("end") is None:
                continue
            choice = choice_at(local, float(unit["start"]), float(unit["end"]))
            if choice is None:
                continue
            unit["speaker"] = "SPEAKER_UNKNOWN" if choice == "unknown" else choice
            unit["local_speaker"] = local
            unit["speaker_identity"] = {"status": "unresolved_human" if choice == "unknown" else "human_verified",
                                         "review_id": review_id, "receipt": receipt}
    for name in ("speaker_timeline", "speaker_assignment_timeline"):
        pieces = []
        for interval in result.get(name) or []:
            local = str(interval.get("local_speaker") or interval["speaker"])
            start, end = float(interval["start"]), float(interval["end"])
            boundaries = sorted({start, end} | {float(turn[key]) for turn in turns for key in ("start", "end")
                if str(turn.get("local_speaker") or turn["speaker"]) == local and start < float(turn[key]) < end})
            for first, last in zip(boundaries, boundaries[1:]):
                choice = choice_at(local, first, last)
                piece = {**interval, "start": first, "end": last, "local_speaker": local}
                if choice is not None:
                    piece["speaker"] = "SPEAKER_UNKNOWN" if choice == "unknown" else choice
                pieces.append(piece)
        if name in result:
            result[name] = pieces
    report = result.get("speaker_identity") or {}
    for match in report.get("matches", []):
        choices = {turn.get("review_choice") for turn in turns
                   if str(turn.get("local_speaker") or turn["speaker"]) == match["local_speaker"]}
        if choices and None not in choices and "unknown" not in choices:
            match["status"] = "human_verified" if len(choices) == 1 else "human_verified_mixed_cluster"
            match["reviewed_speakers"] = sorted(choices)
            if len(choices) == 1:
                match["speaker"] = next(iter(choices))
        elif "unknown" in choices:
            match["status"] = "unresolved_human"
    active = sorted({str(word.get("speaker") or segment.get("speaker"))
                     for segment in result.get("segments", []) for word in segment.get("words") or [segment]})
    report.update(active_speaker_ids=active, active_speaker_count=len(active), human_review_applied=True)
    result["speaker_identity"] = report


def restore_reviewed_choices(review_dir, source_key, fingerprint, local_result, result):
    path = packet_path(review_dir, source_key)
    if not path.exists():
        return
    packet = json.loads(path.read_text(encoding="utf-8"))
    choices = packet.get("applied_decisions")
    if not choices:
        return
    if fingerprint != packet["fingerprint"] or _segmentation_key(local_result) != _segmentation_key(packet["local_result"]):
        raise SpeakerRegistryError("recording or speaker boundaries changed after human review; previous outputs are preserved. Archive this recording's review packet before explicitly re-reviewing the new segmentation")
    turns = _resolve_choices(packet, choices)
    apply_turn_choices(result, turns, packet["review_id"])


def update_overlap_events(payload):
    events = {}
    for turn in payload.get("turns", []):
        for event in turn.get("acoustic_overlap", []):
            item = {**event, "source": "diarization_timeline"}
            events[stable_hash(item)] = item
    payload["overlap_events"] = sorted(events.values(), key=lambda item: (item["start"], item["end"]))


def save_review(review_dir, source, source_key, fingerprint, local_result, evidence,
                payload, outputs, registry, audio):
    path = packet_path(review_dir, source_key)
    matches = (payload.get("speaker_identity") or {}).get("matches", [])
    packet = {"version": 1, "source": str(source), "source_key": source_key,
              "embedding_model": {"name": registry.model, "revision": registry.model_revision},
              "fingerprint": fingerprint, "local_result": local_result,
              "evidence": evidence, "payload": payload,
              "settings_hash": payload["processing"]["settings_hash"],
              "outputs": {key: str(value) for key, value in outputs.items()},
              "profiles": registry.profiles_for_review(), "matches": matches,
              "pending": [item["local_speaker"] for item in matches
                          if item["status"].startswith("unresolved")],
              "created_utc": utc_now()}
    packet["review_id"] = packet_digest(packet)
    # Keep accepted decisions when rebuilding an identical review packet.
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") == fingerprint and _segmentation_key(previous["local_result"]) == _segmentation_key(local_result):
            packet["applied_decisions"] = previous.get("applied_decisions", {})
            packet["receipts"] = previous.get("receipts", [])
            packet["new_profile_ids"] = previous.get("new_profile_ids", {})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    audio_path = path.with_suffix(".wav")
    temporary = path.with_suffix(".wav.tmp")
    import numpy as np
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        with wave.open(handle, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE)
            for first in range(0, len(audio), SAMPLE_RATE * 60):
                chunk = np.asarray(audio[first:first + SAMPLE_RATE * 60])
                output.writeframesraw((np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes())
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(audio_path)
    atomic_write_json(path, packet)
    render_review(path, packet)
    return path


def render_review(path, packet):
    turns = build_turns(normalize_segments(packet["local_result"]))
    annotate_speaker_attribution(turns, packet["local_result"].get("speaker_refinement"))
    from .analysis import annotate_raw_overlap
    annotate_raw_overlap(turns, packet["local_result"].get("speaker_timeline", []))
    public = {"review_id": packet["review_id"], "packet": path.name,
              "source": packet["source_key"], "profiles": packet["profiles"],
              "pending": packet["pending"], "matches": packet["matches"],
              "turns": turns,
              "applied_decisions": {**packet.get("applied_decisions", {}), "profile_updates": {}},
              "new_profile_ids": packet.get("new_profile_ids", {}),
              "draft_revision": stable_hash({key: packet.get(key) for key in
                                             ("applied_decisions", "profiles", "new_profile_ids")})[:16],
              "mixed": [key for key, item in packet["evidence"].items() if item.get("suspected_mixed_speakers")],
              "windows": {key: [{k: v for k, v in window.items() if k != "embedding"}
                                 for window in item.get("windows", [])]
                          for key, item in packet["evidence"].items()}}
    data = json.dumps(public, ensure_ascii=False).replace("<", "\\u003c")
    parts = {"__REVIEW_DATA__": data, "__AUDIO__": html.escape(path.with_suffix(".wav").name, quote=True),
             "__REVIEW_STATE__": Path(__file__).with_name("review_state.js").read_text(encoding="utf-8")}
    document = re.sub(r"__REVIEW_DATA__|__REVIEW_STATE__|__AUDIO__", lambda match: parts[match.group()],
                      Path(__file__).with_name("review_page.html").read_text(encoding="utf-8"))
    atomic_write_text(path.with_suffix(".html"), document)


def review_index(review_dir):
    root = Path(review_dir) / "speaker-reviews"
    rows = []
    catalog = VoiceRegistry(Path(review_dir) / "speaker_registry.json").profiles_for_review()
    for path in sorted(root.glob("*.json")):
        packet = json.loads(path.read_text(encoding="utf-8"))
        if packet.get("version") != 1:
            continue
        # Regenerate existing pages after an app update without loading any models.
        packet["profiles"] = catalog
        if not packet.get("new_profile_ids"):
            packet["new_profile_ids"] = VoiceRegistry(Path(review_dir) / "speaker_registry.json").reviewed_profile_ids(packet.get("receipts", []))
        packet["matches"] = packet["payload"].get("speaker_identity", {}).get("matches", packet["matches"])
        render_review(path, packet)
        count = len(packet["pending"])
        rows.append((not count, packet["source_key"], f'<li><a href="{html.escape(path.with_suffix(".html").name)}">'
                    f'{html.escape(packet["source_key"])}</a><span class="{"pending" if count else "done"}">'
                    f'{str(count) + " voice(s) need review" if count else "Speakers assigned"}</span></li>'))
    index = root / "index.html"
    atomic_write_text(index, '<!doctype html><html lang="en"><meta charset="utf-8">'
                      '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Speaker reviews</title>'
                      '<style>body{font:16px/1.5 system-ui;background:#eef2f7;color:#182235;max-width:1000px;margin:auto;padding:24px}'
                      'h1{color:#13243c}ul{list-style:none;padding:0}li{display:flex;justify-content:space-between;gap:18px;flex-wrap:wrap;background:white;'
                      'border:1px solid #a7b8ce;border-radius:8px;padding:22px;margin:16px 0}a{color:#174bb5;font-weight:650;overflow-wrap:anywhere}'
                      'span{padding:3px 10px;border-radius:5px}.pending{background:#ffdf91}.done{background:#d1f1e3}code{background:white;padding:4px}</style>'
                      '<h1>Speaker reviews</h1><p>Start with recordings that need attention. Confident matches are already selected; '
                      'you only need to identify uncertain voices and correct exceptions.</p>'
                      '<p>Each page supports clip playback, profile edits, saved-review import and export. '
                      'After exporting, use its <strong>Copy apply command</strong> button and run that command in your project terminal.</p>'
                      '<ul>' + "".join(row for _, _, row in sorted(rows)) + '</ul>'
                      + ('' if rows else '<p>No recording reviews yet. Add media and run <code>./transcribe-media</code>.</p>') + '</html>')
    return index


def _resolve_choices(packet, decisions):
    turns = build_turns(normalize_segments(packet["local_result"]))
    assignments = decisions.get("assignments", {})
    overrides = decisions.get("turn_overrides", {})
    if not isinstance(assignments, dict) or not isinstance(overrides, dict):
        raise SpeakerRegistryError("review assignments must be objects")
    if any(not isinstance(value, str) or not value for value in [*assignments.values(), *overrides.values()]):
        raise SpeakerRegistryError("review profile choices must be nonempty strings")
    local_ids = {str(turn.get("local_speaker") or turn["speaker"]) for turn in turns}
    if set(assignments) - local_ids or set(overrides) - {turn["id"] for turn in turns}:
        raise SpeakerRegistryError("review references a speaker or turn absent from this recording")
    if not assignments and not overrides and not decisions.get("range_overrides") and not decisions.get("profile_updates") and decisions.get("format_version") != 2:
        raise SpeakerRegistryError("review contains no decisions")
    for turn in turns:
        local = str(turn.get("local_speaker") or turn["speaker"])
        turn["review_choice"] = overrides.get(turn["id"], assignments.get(local))
    expanded = []
    ranges = decisions.get("range_overrides", [])
    if not isinstance(ranges, list):
        raise SpeakerRegistryError("range overrides must be a list")
    known_turns = {turn["id"]: turn for turn in turns}
    for item in ranges:
        if not isinstance(item, dict) or item.get("turn_id") not in known_turns:
            raise SpeakerRegistryError("range override references an absent turn")
        turn = known_turns[item["turn_id"]]
        first, last = item.get("start"), item.get("end")
        if not isinstance(first, (int, float)) or not isinstance(last, (int, float)) or not all(map(math.isfinite, (first, last))):
            raise SpeakerRegistryError("range timestamps must be finite seconds")
        if not turn["start"] <= first < last <= turn["end"] or not isinstance(item.get("profile"), str):
            raise SpeakerRegistryError("range override must lie inside its displayed turn")
    for turn in turns:
        selected = sorted((item for item in ranges if item["turn_id"] == turn["id"]), key=lambda item: item["start"])
        if any(a["end"] > b["start"] for a, b in zip(selected, selected[1:])):
            raise SpeakerRegistryError("range overrides overlap; use disjoint ranges")
        boundaries = sorted({turn["start"], turn["end"]} | {item[key] for item in selected for key in ("start", "end")})
        for first, last in zip(boundaries, boundaries[1:]):
            middle = (first + last) / 2
            override = next((item for item in selected if item["start"] <= middle < item["end"]), None)
            expanded.append({**turn, "start": first, "end": last,
                             "review_choice": override["profile"] if override else turn["review_choice"]})
    return expanded


def _review_references(packet, turns, excluded):
    references = []
    for local, evidence in packet["evidence"].items():
        for index, window in enumerate(evidence.get("windows", [])):
            if f"{local}:{index}" in excluded or not window.get("reference_eligible"):
                continue
            covered = {}
            for turn in turns:
                if str(turn.get("local_speaker") or turn["speaker"]) != local:
                    continue
                overlap = max(0., min(window["end"], turn["end"]) - max(window["start"], turn["start"]))
                choice = turn.get("review_choice")
                if overlap > 0:
                    covered[choice] = covered.get(choice, 0.) + overlap
            if len(covered) != 1:
                continue  # A clip crossing different identities cannot train either.
            profile, coverage = next(iter(covered.items()))
            # Diarization intervals include pauses that ASR turns don't cover.
            # Those silent gaps do not invalidate an otherwise reviewed voice.
            if profile in (None, "unknown") or coverage < min(0.2, window["duration"] * 0.25):
                continue
            observation = VoiceRegistry._observation(packet["source_key"], packet["fingerprint"]["sha256"],
                f'{local}:{window["start"]:.4f}:{window["end"]:.4f}',
                {"embedding": window["embedding"], "clean_seconds": window["duration"],
                 "window_count": 1, "cohesion": window["cohesion_similarity"]})
            observation["start"], observation["end"] = window["start"], window["end"]
            references.append({"profile": profile, "observation": observation})
    return references


def apply_review(review_dir, decision_path):
    decisions = json.loads(Path(decision_path).read_text(encoding="utf-8"))
    if not isinstance(decisions, dict):
        raise SpeakerRegistryError("review decisions must be a JSON object")
    packet_name = decisions.get("packet", "")
    if Path(packet_name).name != packet_name or not packet_name.endswith(".json"):
        raise SpeakerRegistryError("invalid speaker review packet name")
    path = Path(review_dir) / "speaker-reviews" / packet_name
    packet = json.loads(path.read_text(encoding="utf-8"))
    if packet_digest(packet) != packet.get("review_id") or decisions.get("review_id") != packet["review_id"]:
        raise SpeakerRegistryError("this review is stale or its source evidence has changed; reopen the latest review")
    if packet["embedding_model"] != {"name": SPEAKER_EMBEDDING_MODEL, "revision": SPEAKER_EMBEDDING_REVISION}:
        raise SpeakerRegistryError("review references use a different embedding model; regenerate them with the current model")
    if source_fingerprint(Path(packet["source"])) != packet["fingerprint"]:
        raise SpeakerRegistryError("source recording changed after this review was created")
    receipt = stable_hash(decisions)
    if packet.get("receipts", [])[-1:] == [receipt]:
        return {"already_applied": True, "pending": len(packet["pending"])}
    # A second import updates the same source's decisions rather than erasing prior review.
    if decisions.get("format_version") not in (None, 2):
        raise SpeakerRegistryError("unsupported review decision format")
    combined = {} if decisions.get("format_version") == 2 else copy.deepcopy(packet.get("applied_decisions", {}))
    for key in ("assignments", "turn_overrides", "new_profiles"):
        if not isinstance(decisions.get(key, {}), dict):
            raise SpeakerRegistryError(f"{key} must be an object")
        combined.setdefault(key, {}).update(decisions.get(key, {}))
    combined["format_version"] = decisions.get("format_version", combined.get("format_version"))
    combined["profile_updates"] = decisions.get("profile_updates", {})
    if not isinstance(combined["profile_updates"], dict) or any(not isinstance(item, dict) for item in combined["profile_updates"].values()):
        raise SpeakerRegistryError("profile updates must contain profile objects")
    combined["range_overrides"] = decisions.get("range_overrides", combined.get("range_overrides", []))
    combined["exclude_windows"] = decisions.get("exclude_windows", combined.get("exclude_windows", []))
    if not isinstance(combined["exclude_windows"], list) or any(not isinstance(item, str) for item in combined["exclude_windows"]):
        raise SpeakerRegistryError("excluded reference clips must be a list of clip IDs")
    # Reopening a saved draft must not create a second ID for the same draft person.
    registry = VoiceRegistry(Path(review_dir) / "speaker_registry.json")
    prior_ids = registry.reviewed_profile_ids(packet.get("receipts", []))
    prior_ids.update(packet.get("new_profile_ids", {}))
    packet["new_profile_ids"] = prior_ids
    for key in ("assignments", "turn_overrides"):
        combined[key] = {local: prior_ids.get(choice, choice) if isinstance(choice, str) else choice
                         for local, choice in combined[key].items()}
    for item in combined["range_overrides"] if isinstance(combined["range_overrides"], list) else []:
        if isinstance(item, dict) and isinstance(item.get("profile"), str):
            item["profile"] = prior_ids.get(item["profile"], item["profile"])
    turns = _resolve_choices(packet, combined)
    profiles = {item["voice_id"] for item in registry.profiles_for_review()}
    used = {turn["review_choice"] for turn in turns} - {None, "unknown"}
    if any(not isinstance(key, str) for key in used):
        raise SpeakerRegistryError("profile choices must be strings")
    new_profiles = {}
    for key in used:
        if key in profiles:
            continue
        metadata = combined["new_profiles"].get(key)
        if not key.startswith("new:") or not isinstance(metadata, dict):
            raise SpeakerRegistryError(f"unknown profile choice: {key}")
        label = str(metadata.get("label", "")).strip()
        role = metadata.get("role", "unspecified")
        if not label or len(label) > 80 or any(ord(c) < 32 for c in label) or role not in ("adult", "child", "unspecified"):
            raise SpeakerRegistryError("new profile needs a short label and adult/child/unspecified role")
        new_profiles[key] = {"label": label, "role": role}
    references = _review_references(packet, turns, set(combined["exclude_windows"]))
    manifest = ManifestStore(Path(review_dir) / "transcription_manifest.json")
    entry = manifest.get(packet["source_key"])
    if not entry or entry.get("source_fingerprint") != packet["fingerprint"]:
        raise SpeakerRegistryError("review has no matching processing manifest entry")
    outputs = {key: Path(value) for key, value in entry["outputs"].items()}
    packets = list((Path(review_dir) / "speaker-reviews").glob("*.json"))
    other_outputs = [Path(value) for record in manifest.data["sources"].values() for value in record.get("outputs", {}).values()]
    transaction = StateTransaction(Path(review_dir), [registry.path, manifest.path, *packets,
                                                     *(item.with_suffix(".html") for item in packets), *other_outputs])
    transaction.begin()
    try:
        application_id = stable_hash([packet["review_id"], receipt, len(packet.get("receipts", []))])
        created = registry.confirm_references(references, new_profiles, application_id,
                                               replace_source_fingerprint=packet["fingerprint"]["sha256"],
                                               profile_updates=combined["profile_updates"])
        packet.setdefault("new_profile_ids", {}).update(created)
        for turn in turns:
            turn["review_choice"] = created.get(turn["review_choice"], turn["review_choice"])
        for key in ("assignments", "turn_overrides"):
            combined[key] = {local: created.get(choice, choice) for local, choice in combined[key].items()}
        for item in combined["range_overrides"]:
            item["profile"] = created.get(item["profile"], item["profile"])
        result = copy.deepcopy(packet["payload"])
        base = copy.deepcopy(packet["local_result"])
        policy = result["processing"].get("settings", {})
        matcher = VoiceRegistry(registry.path, reviewed=True, learn=False,
                                known_voices=policy.get("known_voices", ()),
                                match_threshold=policy.get("speaker_match_threshold", .45),
                                match_margin=policy.get("speaker_match_margin", .12))
        labels = {str(turn.get("local_speaker") or turn["speaker"]) for turn in turns}
        report = matcher.identify(source_key=packet["source_key"], source_fingerprint=packet["fingerprint"]["sha256"],
                                  local_speakers=labels, evidence=packet["evidence"],
                                  incompatible_pairs=overlapping_speaker_pairs(base.get("speaker_timeline", []))) if profiles or created else result["speaker_identity"]
        apply_speaker_identities(base, report)
        for key in ("segments", "speaker_timeline", "speaker_assignment_timeline", "speaker_identity"):
            if key in base:
                result[key] = base[key]
        segments = result["segments"]
        # Apply against original local labels, preserving the recognized words.
        apply_turn_choices(result, turns, packet["review_id"], receipt)
        normalized = normalize_segments({"segments": segments})
        result["segments"] = normalized
        result["turns"] = build_turns(normalized)
        annotate_speaker_attribution(result["turns"], result.get("speaker_refinement"))
        from .analysis import annotate_raw_overlap
        annotate_raw_overlap(result["turns"], result.get("speaker_timeline", []))
        update_overlap_events(result)
        # Turn boundaries changed: old tone/acoustic estimates no longer describe them.
        result["human_review"] = {"review_id": packet["review_id"], "applied_utc": utc_now(),
                                  "receipt": receipt, "decisions": combined,
                                  "acoustic_and_tone_recompute_required": True}
        result["processing"]["provenance"]["human_review"] = result["human_review"]
        packet["applied_decisions"] = combined
        packet.setdefault("receipts", []).append(receipt)
        pending = [item["local_speaker"] for item in result["speaker_identity"].get("matches", [])
                   if item["status"].startswith("unresolved")]
        packet["pending"] = pending
        result["speaker_review"] = {"pending": pending, "review_file": str(path.with_suffix(".html"))}
        result.setdefault("speaker_identity", {})["human_review_applied"] = True
        summary = registry.validate()
        result["speaker_identity"].update(profile_count=summary["profile_count"], registry_revision=summary["revision"],
                                         registry_state_hash=summary["state_hash"])
        active_ids = {turn["speaker"] for turn in result["turns"]}
        result["speaker_profiles"] = [profile for profile in registry.profiles_for_review() if profile["voice_id"] in active_ids]
        packet["payload"] = result
        packet["profiles"] = registry.profiles_for_review()
        packet["matches"] = result["speaker_identity"]["matches"]
        write_outputs(outputs, result)
        manifest.record_speaker_registry({"path": str(registry.path), **registry.validate()})
        entry["human_review"] = {"review_id": packet["review_id"], "pending": pending, "receipt": receipt}
        entry["speaker_review_pending"] = len(pending)
        if not entry.get("retry_recommended"):
            entry["status"] = "awaiting_review" if pending else "complete"
            entry["completion_state"] = True
        manifest.update(packet["source_key"], entry)
        atomic_write_json(path, packet)
        render_review(path, packet)
        # Label changes describe the same durable person in every transcript.
        if combined["profile_updates"]:
            _sync_profile_labels(manifest, packet["profiles"], registry.validate(), path)
        transaction.commit()
    except BaseException:
        transaction.rollback()
        raise
    review_index(review_dir)
    return {"created_profiles": created, "pending": len(pending), "references_saved": len(references),
            "profiles_needing_audio": [profile["voice_id"] for profile in packet["profiles"] if profile["training_status"] != "ready"],
            "outputs": {key: str(value) for key, value in outputs.items()}}


def _sync_profile_labels(manifest, catalog, summary, current_path):
    for entry in manifest.data["sources"].values():
        files = {key: Path(value) for key, value in entry.get("outputs", {}).items()}
        if "json" not in files or not files["json"].exists():
            continue
        payload = json.loads(files["json"].read_text(encoding="utf-8"))
        active = {turn["speaker"] for turn in payload.get("turns", [])}
        payload["speaker_profiles"] = [item for item in catalog if item["voice_id"] in active]
        payload.setdefault("speaker_identity", {}).update(profile_count=summary["profile_count"],
            registry_revision=summary["revision"], registry_state_hash=summary["state_hash"])
        write_outputs(files, payload)
    for path in current_path.parent.glob("*.json"):
        packet = json.loads(path.read_text(encoding="utf-8"))
        packet["profiles"] = catalog
        active = {turn["speaker"] for turn in packet["payload"].get("turns", [])}
        packet["payload"]["speaker_profiles"] = [item for item in catalog if item["voice_id"] in active]
        atomic_write_json(path, packet)
        render_review(path, packet)


def merge_project_profiles(review_dir, duplicate, canonical):
    """An explicit human merge keeps an alias and updates all registered outputs."""
    review_dir = Path(review_dir)
    registry = VoiceRegistry(review_dir / "speaker_registry.json")
    manifest = ManifestStore(review_dir / "transcription_manifest.json")
    packets = list((review_dir / "speaker-reviews").glob("*.json"))
    outputs = [{key: Path(value) for key, value in entry.get("outputs", {}).items()}
               for entry in manifest.data["sources"].values()]
    for files in outputs:
        if "json" not in files or not files["json"].exists():
            raise SpeakerRegistryError("merge needs each recording's canonical JSON to update transcripts safely")
    paths = [registry.path, manifest.path, *packets,
             *(path.with_suffix(".html") for path in packets),
             *(path for files in outputs for path in files.values())]
    transaction = StateTransaction(review_dir, paths)
    transaction.begin()
    try:
        aliases = registry.merge_profiles(duplicate, canonical)
        def relabel(value):
            if isinstance(value, str):
                return aliases.get(value, value)
            if isinstance(value, list):
                return [relabel(item) for item in value]
            if isinstance(value, dict):
                return {key: relabel(item) for key, item in value.items()}
            return value
        for files in outputs:
            payload = relabel(json.loads(files["json"].read_text(encoding="utf-8")))
            active = sorted({turn["speaker"] for turn in payload.get("turns", [])})
            payload.setdefault("speaker_identity", {}).update(active_speaker_ids=active, active_speaker_count=len(active))
            payload["speaker_profiles"] = [profile for profile in registry.profiles_for_review() if profile["voice_id"] in active]
            payload.setdefault("identity_merges", []).append({"duplicate": duplicate, "canonical": canonical, "at": utc_now()})
            write_outputs(files, payload)
        for path in packets:
            packet = json.loads(path.read_text(encoding="utf-8"))
            # Source-bound acoustic evidence stays immutable.
            for key in ("payload", "matches", "applied_decisions", "new_profile_ids"):
                if key in packet:
                    packet[key] = relabel(packet[key])
            packet["profiles"] = registry.profiles_for_review()
            atomic_write_json(path, packet)
            render_review(path, packet)
        manifest.record_speaker_registry({"path": str(registry.path), **registry.validate()})
        transaction.commit()
    except BaseException:
        transaction.rollback()
        raise
    return {"merged": duplicate, "canonical": canonical, "aliases": aliases, "recordings_updated": len(outputs)}


def refresh_reviews(review_dir):
    review_dir = Path(review_dir)
    registry_path = review_dir / "speaker_registry.json"
    catalog = VoiceRegistry(registry_path).profiles_for_review()
    if not catalog:
        raise SpeakerRegistryError("confirm the first voice profiles before refreshing matches")
    manifest = ManifestStore(review_dir / "transcription_manifest.json")
    refreshed = []
    for path in sorted((review_dir / "speaker-reviews").glob("*.json")):
        packet = json.loads(path.read_text(encoding="utf-8"))
        if packet_digest(packet) != packet.get("review_id"):
            raise SpeakerRegistryError(f"review evidence was modified: {path}")
        if source_fingerprint(Path(packet["source"])) != packet["fingerprint"]:
            raise SpeakerRegistryError(f"recording changed after review: {packet['source_key']}")
        entry = manifest.get(packet["source_key"])
        if not entry or entry.get("source_fingerprint") != packet["fingerprint"]:
            raise SpeakerRegistryError("review and processing manifest no longer agree")
        settings = packet["payload"]["processing"].get("settings", {})
        registry = VoiceRegistry(registry_path, reviewed=True, learn=False,
                                 known_voices=settings.get("known_voices", ()),
                                 match_threshold=settings.get("speaker_match_threshold", .45),
                                 match_margin=settings.get("speaker_match_margin", .12))
        result = copy.deepcopy(packet["local_result"])
        timeline = result.get("speaker_timeline", [])
        labels = sorted({item["speaker"] for item in timeline if item.get("speaker") not in (None, "SPEAKER_UNKNOWN")})
        report = registry.identify(source_key=packet["source_key"], source_fingerprint=packet["fingerprint"]["sha256"],
                                   local_speakers=labels, evidence=packet["evidence"],
                                   incompatible_pairs=overlapping_speaker_pairs(timeline))
        apply_speaker_identities(result, report)
        if packet.get("applied_decisions"):
            apply_turn_choices(result, _resolve_choices(packet, packet["applied_decisions"]), packet["review_id"])
        payload = copy.deepcopy(packet["payload"])
        payload["segments"] = normalize_segments(result)
        payload["turns"] = build_turns(payload["segments"])
        payload["speaker_timeline"] = result.get("speaker_timeline", [])
        payload["speaker_assignment_timeline"] = result.get("speaker_assignment_timeline")
        payload["speaker_identity"] = result["speaker_identity"]
        active_ids = {turn["speaker"] for turn in payload["turns"]}
        payload["speaker_profiles"] = [profile for profile in catalog if profile["voice_id"] in active_ids]
        annotate_speaker_attribution(payload["turns"], result.get("speaker_refinement"))
        from .analysis import annotate_raw_overlap
        annotate_raw_overlap(payload["turns"], payload["speaker_timeline"])
        update_overlap_events(payload)
        pending = [item["local_speaker"] for item in result["speaker_identity"]["matches"]
                   if item["status"].startswith("unresolved")]
        payload["speaker_review"] = {"pending": pending, "review_file": str(path.with_suffix(".html"))}
        payload["processing"]["provenance"]["voice_refresh"] = {"at": utc_now(), "registry_revision": report["registry_revision"],
                                                                    "acoustic_and_tone_recompute_required": True}
        outputs = {key: Path(value) for key, value in entry["outputs"].items()}
        transaction = StateTransaction(review_dir, [manifest.path, path, path.with_suffix(".html"), *outputs.values()])
        transaction.begin()
        try:
            write_outputs(outputs, payload)
            packet.update(payload=payload, pending=pending, profiles=catalog, matches=report["matches"])
            atomic_write_json(path, packet)
            render_review(path, packet)
            entry["speaker_review_pending"] = len(pending)
            if not entry.get("retry_recommended"):
                entry.update(status="awaiting_review" if pending else "complete", completion_state=True)
            manifest.update(packet["source_key"], entry)
            transaction.commit()
        except BaseException:
            transaction.rollback()
            raise
        refreshed.append({"source": packet["source_key"], "pending": len(pending)})
    return {"recordings": refreshed, "review_index": str(review_index(review_dir))}
