"""Private, offline speaker review with source-bound, auditable decisions."""
from __future__ import annotations

import copy
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
    turns_by_local = {}
    for turn in turns:
        turns_by_local.setdefault(str(turn.get("local_speaker") or turn["speaker"]), []).append(turn)

    def choice_at(local, start, end):
        middle = (start + end) / 2
        candidates = turns_by_local.get(local, [])
        chosen = next((turn for turn in candidates if turn["start"] <= middle < turn["end"]), None)
        # Alignment can give a final word zero duration. Include that endpoint,
        # while shared correction boundaries still belong to the later interval.
        if chosen is None:
            chosen = next((turn for turn in reversed(candidates) if middle == turn["end"]), None)
        return chosen.get("review_choice") if chosen is not None else None

    spoken_choices = {}
    for segment in result.get("segments", []):
        for unit in segment.get("words") or [segment]:
            local = str(unit.get("local_speaker") or segment.get("local_speaker") or unit.get("speaker") or segment["speaker"])
            if local == "SPEAKER_UNKNOWN" and segment.get("local_speaker"):
                local = str(segment["local_speaker"])
            if unit.get("start") is None or unit.get("end") is None:
                spoken_choices.setdefault(local, set()).add(None)
                continue
            choice = choice_at(local, float(unit["start"]), float(unit["end"]))
            spoken_choices.setdefault(local, set()).add(choice)
            if choice is None:
                continue
            unit["speaker"] = "SPEAKER_UNKNOWN" if choice in ("unknown", "ignore") else choice
            unit["local_speaker"] = local
            unit["speaker_identity"] = {"status": "unresolved_human" if choice == "unknown" else "human_excluded" if choice == "ignore" else "human_verified",
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
                    piece["speaker"] = "SPEAKER_UNKNOWN" if choice in ("unknown", "ignore") else choice
                pieces.append(piece)
        if name in result:
            result[name] = pieces
    report = result.get("speaker_identity") or {}
    for match in report.get("matches", []):
        # Match the words we actually assigned, not pauses or trimmed boundaries
        # between them. Reviewing samples must not verify other unreviewed words.
        choices = spoken_choices.get(match["local_speaker"], set())
        if choices and None not in choices and "unknown" not in choices:
            match["status"] = "human_excluded" if choices == {"ignore"} else "human_verified" if len(choices) == 1 else "human_verified_mixed_cluster"
            match["reviewed_speakers"] = sorted(choices - {"ignore"})
            if len(choices) == 1:
                match["speaker"] = "SPEAKER_UNKNOWN" if choices == {"ignore"} else next(iter(choices))
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


def review_data(path, packet):
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
    return public


def batch_digest(reviews):
    return stable_hash(sorted((item["packet"], item["review_id"]) for item in reviews))


def render_review(path, packet=None, *, public=None):
    public = review_data(path, packet) if public is None else public
    data = json.dumps(public, ensure_ascii=False).replace("<", "\\u003c")
    parts = {"__REVIEW_DATA__": data,
             "__REVIEW_STATE__": Path(__file__).with_name("review_state.js").read_text(encoding="utf-8"),
             "__REVIEW_IO__": Path(__file__).with_name("review_io.js").read_text(encoding="utf-8"),
             "__REVIEW_PLAYBACK__": Path(__file__).with_name("review_playback.js").read_text(encoding="utf-8"),
             "__REVIEW_CONTROLLER__": Path(__file__).with_name("review_page.js").read_text(encoding="utf-8"),
             "__REVIEW_CSS__": Path(__file__).with_name("review_page.css").read_text(encoding="utf-8")}
    document = re.sub("|".join(parts), lambda match: parts[match.group()],
                      Path(__file__).with_name("review_page.html").read_text(encoding="utf-8"))
    atomic_write_text(path.with_suffix(".html"), document)


def review_index(review_dir):
    root = Path(review_dir) / "speaker-reviews"
    registry = VoiceRegistry(Path(review_dir) / "speaker_registry.json")
    catalog = registry.profiles_for_review()
    recordings = []
    for path in sorted(root.glob("*.json")):
        packet = json.loads(path.read_text(encoding="utf-8"))
        if packet.get("version") != 1:
            continue
        packet["profiles"] = catalog
        if not packet.get("new_profile_ids"):
            packet["new_profile_ids"] = registry.reviewed_profile_ids(packet.get("receipts", []))
        packet["matches"] = packet["payload"].get("speaker_identity", {}).get("matches", packet["matches"])
        render_review(path, packet)
        recordings.append(review_data(path, packet))
    index = root / "index.html"
    if recordings:
        recordings.sort(key=lambda item: (not item["pending"], item["source"]))
        aliases = {}
        state_path = Path(review_dir) / "speaker-review-batches.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            for batch in state.get("batches", {}).values():
                for key, value in registry.resolve_profile_ids(batch.get("new_profile_ids", {})).items():
                    aliases[key] = value if key not in aliases or aliases[key] == value else None
        render_review(index, public={"kind": "speaker_review_batch", "batch_id": batch_digest(recordings),
                                     "recordings": recordings, "profiles": catalog,
                                     "new_profile_ids": {key: value for key, value in aliases.items() if value}})
    else:
        atomic_write_text(index, '<!doctype html><meta charset="utf-8"><title>Speaker reviews</title>'
                          '<h1>Speaker reviews</h1><p>No recording reviews yet. Add media and run ./transcribe-media.</p>')
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
            if profile in (None, "unknown", "ignore") or coverage < min(0.2, window["duration"] * 0.25):
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
    if isinstance(decisions, dict) and decisions.get("kind") == "speaker_review_batch":
        return apply_batch_review(review_dir, decisions)
    return _apply_recording_review(review_dir, decisions)


def apply_batch_review(review_dir, decisions):
    """One shared enrollment, all explicit corrections, then a final cached rematch.

    The outer journal covers every recording so any failed import or rematch
    restores the whole project. Caller holds the normal project lock.
    """
    review_dir = Path(review_dir)
    reviews = decisions.get("reviews")
    if decisions.get("format_version") != 3 or not isinstance(reviews, list) or not reviews:
        raise SpeakerRegistryError("batch review needs a nonempty list of recording decisions")
    if any(not isinstance(item, dict) or not isinstance(item.get("packet"), str) or
           not isinstance(item.get("review_id"), str) or item.get("format_version") != 2 for item in reviews):
        raise SpeakerRegistryError("invalid recording decisions in batch review")
    if len({item["packet"] for item in reviews}) != len(reviews) or batch_digest(reviews) != decisions.get("batch_id"):
        raise SpeakerRegistryError("batch review identity is invalid or contains duplicate recordings")
    definitions = decisions.get("new_profiles", {})
    updates = decisions.get("profile_updates", {})
    if not isinstance(definitions, dict) or not isinstance(updates, dict) or any(not isinstance(item, dict) for item in updates.values()):
        raise SpeakerRegistryError("batch profiles must be objects")
    # Validate source evidence before enrollment. Each recording is checked again
    # by the regular importer; a later change also rolls back the whole batch.
    used = set()
    for item in reviews:
        packet = _validated_review_packet(review_dir, item)
        used.update(turn["review_choice"] for turn in _resolve_choices(packet, item))
    registry = VoiceRegistry(review_dir / "speaker_registry.json")
    state_path = review_dir / "speaker-review-batches.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"batches": {}}
    prior = state["batches"].setdefault(decisions["batch_id"], {"new_profile_ids": {}, "receipts": []})
    receipt = stable_hash(decisions)
    if prior["receipts"][-1:] == [receipt]:
        return {"already_applied": True, "batch": True}
    mapping = registry.resolve_profile_ids(prior["new_profile_ids"])
    new_profiles = {}
    for key in sorted(used - {None, "unknown", "ignore"}):
        if not key.startswith("new:") or key in mapping:
            continue
        metadata = definitions.get(key)
        if not isinstance(metadata, dict):
            raise SpeakerRegistryError(f"batch lacks a shared profile definition: {key}")
        label, role = metadata.get("label"), metadata.get("role", "unspecified")
        if not isinstance(label, str) or not label.strip() or len(label) > 80 or any(ord(c) < 32 for c in label) or role not in ("adult", "child", "unspecified"):
            raise SpeakerRegistryError("new batch profile needs a short label and valid role")
        new_profiles[key] = {"label": label.strip(), "role": role}
    manifest = ManifestStore(review_dir / "transcription_manifest.json")
    packets = list((review_dir / "speaker-reviews").glob("*.json"))
    outputs = [Path(value) for entry in manifest.data["sources"].values() for value in entry.get("outputs", {}).values()]
    transaction = StateTransaction(review_dir, [registry.path, manifest.path, state_path, *packets,
        *(path.with_suffix(".html") for path in packets), review_dir / "speaker-reviews/index.html", *outputs])
    transaction.begin()
    try:
        created = registry.confirm_references([], new_profiles,
            stable_hash([decisions["batch_id"], receipt, len(prior["receipts"])]), profile_updates=updates)
        mapping.update(created)
        applied = []
        for original in reviews:
            item = copy.deepcopy(original)
            for name in ("assignments", "turn_overrides"):
                item[name] = {key: mapping.get(value, value) for key, value in item.get(name, {}).items()}
            for override in item.get("range_overrides", []):
                override["profile"] = mapping.get(override["profile"], override["profile"])
            item["new_profiles"], item["profile_updates"] = {}, {}
            applied.append(_apply_recording_review(review_dir, item, managed=True))
        # All reviewed references now exist, so processing order cannot determine
        # the final automatic assignments in any of the cached recordings.
        refreshed = refresh_reviews(review_dir, managed=True, allow_empty=True)
        if updates and packets:
            _sync_profile_labels(ManifestStore(manifest.path), registry.profiles_for_review(), registry.validate(), packets[0])
        prior["new_profile_ids"] = mapping
        prior["receipts"].append(receipt)
        atomic_write_json(state_path, state)
        review_index(review_dir)
        transaction.commit()
    except BaseException:
        transaction.rollback()
        raise
    return {"batch": True, "created_profiles": created, "reviews_applied": len(reviews),
            "references_saved": sum(item.get("references_saved", 0) for item in applied),
            "pending": sum(item["pending"] for item in refreshed["recordings"]),
            "recordings_refreshed": refreshed["recordings"], "review_index": refreshed["review_index"]}


def _validated_review_packet(review_dir, decisions):
    if not isinstance(decisions, dict):
        raise SpeakerRegistryError("review decisions must be a JSON object")
    packet_name = decisions.get("packet", "")
    if not isinstance(packet_name, str) or Path(packet_name).name != packet_name or not packet_name.endswith(".json"):
        raise SpeakerRegistryError("invalid speaker review packet name")
    path = Path(review_dir) / "speaker-reviews" / packet_name
    packet = json.loads(path.read_text(encoding="utf-8"))
    if packet_digest(packet) != packet.get("review_id") or decisions.get("review_id") != packet["review_id"]:
        raise SpeakerRegistryError("this review is stale or its source evidence has changed; reopen the latest review")
    if packet["embedding_model"] != {"name": SPEAKER_EMBEDDING_MODEL, "revision": SPEAKER_EMBEDDING_REVISION}:
        raise SpeakerRegistryError("review references use a different embedding model; regenerate them with the current model")
    if source_fingerprint(Path(packet["source"])) != packet["fingerprint"]:
        raise SpeakerRegistryError("source recording changed after this review was created")
    return packet


def _apply_recording_review(review_dir, decisions, *, managed=False):
    packet = _validated_review_packet(review_dir, decisions)
    path = Path(review_dir) / "speaker-reviews" / decisions["packet"]
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
    used = {turn["review_choice"] for turn in turns} - {None, "unknown", "ignore"}
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
    if not managed:
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
        if not managed:
            transaction.commit()
    except BaseException:
        if not managed:
            transaction.rollback()
        raise
    if not managed:
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


def refresh_reviews(review_dir, *, managed=False, allow_empty=False):
    review_dir = Path(review_dir)
    registry_path = review_dir / "speaker_registry.json"
    catalog = VoiceRegistry(registry_path).profiles_for_review()
    if not catalog and not allow_empty:
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
        registry = VoiceRegistry(registry_path, reviewed=True, learn=not bool(catalog),
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
        if not managed:
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
            if not managed:
                transaction.commit()
        except BaseException:
            if not managed:
                transaction.rollback()
            raise
        refreshed.append({"source": packet["source_key"], "pending": len(pending)})
    return {"recordings": refreshed, "review_index": str(Path(review_dir) / "speaker-reviews/index.html") if managed else str(review_index(review_dir))}
