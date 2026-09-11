"""Private, offline speaker review with source-bound, auditable decisions."""
from __future__ import annotations

import copy
import html
import json
import math
import os
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
    public = {"review_id": packet["review_id"], "packet": path.name,
              "source": packet["source_key"], "profiles": packet["profiles"],
              "pending": packet["pending"], "matches": packet["matches"],
              "turns": turns,
              "mixed": [key for key, item in packet["evidence"].items() if item.get("suspected_mixed_speakers")],
              "windows": {key: [{k: v for k, v in window.items() if k != "embedding"}
                                 for window in item.get("windows", [])]
                          for key, item in packet["evidence"].items()}}
    data = json.dumps(public, ensure_ascii=False).replace("<", "\\u003c")
    document = _REVIEW_HTML.replace("__REVIEW_DATA__", data).replace(
        "__AUDIO__", html.escape(path.with_suffix(".wav").name, quote=True))
    atomic_write_text(path.with_suffix(".html"), document)


def review_index(review_dir):
    root = Path(review_dir) / "speaker-reviews"
    rows = []
    for path in sorted(root.glob("*.json")):
        packet = json.loads(path.read_text(encoding="utf-8"))
        if packet.get("version") != 1:
            continue
        rows.append(f'<li><a href="{html.escape(path.with_suffix(".html").name)}">'
                    f'{html.escape(packet["source_key"])}</a> — {len(packet["pending"])} voice(s) awaiting review</li>')
    index = root / "index.html"
    atomic_write_text(index, '<!doctype html><meta charset="utf-8"><title>Speaker reviews</title>'
                      '<h1>Speaker reviews</h1><p>Open a recording, listen, choose identities, then export decisions. '
                      'Apply the downloaded JSON with --apply-speaker-review.</p><ul>' + "".join(rows) + '</ul>')
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
    if not assignments and not overrides and not decisions.get("range_overrides"):
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
            if profile in (None, "unknown") or coverage < 0.90 * window["duration"]:
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
    if receipt in packet.get("receipts", []):
        return {"already_applied": True, "pending": len(packet["pending"])}
    # A second import updates the same source's decisions rather than erasing prior review.
    combined = copy.deepcopy(packet.get("applied_decisions", {}))
    for key in ("assignments", "turn_overrides", "new_profiles"):
        combined.setdefault(key, {}).update(decisions.get(key, {}))
    combined["range_overrides"] = decisions.get("range_overrides", combined.get("range_overrides", []))
    combined["exclude_windows"] = decisions.get("exclude_windows", combined.get("exclude_windows", []))
    turns = _resolve_choices(packet, combined)
    registry = VoiceRegistry(Path(review_dir) / "speaker_registry.json")
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
    if set(new_profiles) - {item["profile"] for item in references}:
        raise SpeakerRegistryError("a new voice needs a clean selected reference clip; keep short/noisy speech unknown or choose an existing profile")
    manifest = ManifestStore(Path(review_dir) / "transcription_manifest.json")
    entry = manifest.get(packet["source_key"])
    if not entry or entry.get("source_fingerprint") != packet["fingerprint"]:
        raise SpeakerRegistryError("review has no matching processing manifest entry")
    outputs = {key: Path(value) for key, value in entry["outputs"].items()}
    transaction = StateTransaction(Path(review_dir), [registry.path, manifest.path, path,
                                                     path.with_suffix(".html"), *outputs.values()])
    transaction.begin()
    try:
        created = registry.confirm_references(references, new_profiles, receipt,
                                               replace_source_fingerprint=packet["fingerprint"]["sha256"])
        for turn in turns:
            turn["review_choice"] = created.get(turn["review_choice"], turn["review_choice"])
        for key in ("assignments", "turn_overrides"):
            combined[key] = {local: created.get(choice, choice) for local, choice in combined[key].items()}
        for item in combined["range_overrides"]:
            item["profile"] = created.get(item["profile"], item["profile"])
        result = copy.deepcopy(packet["payload"])
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
        transaction.commit()
    except BaseException:
        transaction.rollback()
        raise
    review_index(review_dir)
    return {"created_profiles": created, "pending": len(pending), "outputs": {key: str(value) for key, value in outputs.items()}}


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
            for key in ("payload", "matches", "applied_decisions"):
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


_REVIEW_HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Review speakers</title>
<style>body{font:16px system-ui;max-width:1100px;margin:auto;padding:24px;background:#f4f6f9;color:#172235}header{position:sticky;top:0;background:#f4f6f9;padding:12px 0;z-index:2}audio{width:100%}section{background:white;border:1px solid #ccd4df;padding:20px;margin:16px 0;border-radius:10px}select,button,input{padding:8px;margin:4px;font:inherit}button{cursor:pointer}h1{font-size:26px}small{color:#46556c}.turn{border-top:1px solid #dde3ec;padding:12px 0}.pending{color:#963b10}.samples{display:flex;flex-wrap:wrap;gap:8px}.sample{padding:10px;background:#eef2f7}#message{font-weight:bold}</style>
<header><h1 id="title"></h1><audio id="audio" controls preload="metadata" src="__AUDIO__"></audio>
<button id="save">Export decisions</button><span id="message" role="status"></span></header>
<p>Listen to several clips from each voice across the recording. Confirm the whole group only if it is one person. Use individual turn overrides for mixed adult/child groups. Uncheck unsuitable reference clips. Unknown is a valid answer.</p>
<p>Candidate scores are cosine similarities, <strong>not probabilities</strong>. New identities require a selected clean reference clip. Child/adult roles are supplied by you; the software does not infer age from pitch.</p>
<label>New profile label <input id="name" maxlength="80"></label><select id="role"><option value="unspecified">Unspecified</option><option value="adult">Adult</option><option value="child">Child</option></select><button id="add">Add choice</button>
<main id="cards"></main><script>
const data=__REVIEW_DATA__;
const newProfiles={'new:Adult A':{label:'Adult A',role:'adult'},'new:Adult B':{label:'Adult B',role:'adult'},'new:Child':{label:'Child',role:'child'}};
const byId=id=>document.getElementById(id); let stopAt=null;
byId('title').textContent='Speaker review: '+data.source;
function options(select, inherit){let value=select.value;select.replaceChildren();
 let o=new Option(inherit?'Use group decision':'Leave unchanged / review later','');select.add(o);
 select.add(new Option('Unknown / unresolved','unknown'));
 for(const p of data.profiles)select.add(new Option(p.voice_id+' — '+p.label+' ('+p.verified_sessions+' verified sessions)',p.voice_id));
 for(const [key,p] of Object.entries(newProfiles))select.add(new Option('Create '+p.label+' ('+p.role+')',key));select.value=value;}
function time(t){return new Date(t*1000).toISOString().slice(11,23)}
function play(start,end){byId('audio').currentTime=Math.max(0,start);stopAt=end;byId('audio').play().catch(()=>{byId('message').textContent='Audio unavailable. Keep the WAV beside this HTML file.'})}
byId('audio').addEventListener('timeupdate',()=>{if(stopAt!==null&&byId('audio').currentTime>=stopAt){byId('audio').pause();stopAt=null}});
function button(text,start,end){let b=document.createElement('button');b.textContent=text;b.onclick=()=>play(start,end);return b}
const localIds=[...new Set(data.turns.map(t=>t.local_speaker||t.speaker))];
for(const local of localIds){const section=document.createElement('section');let h=document.createElement('h2');h.textContent=local+(data.pending.includes(local)?' — needs review':' — automatically matched');section.append(h);
 const match=data.matches.find(m=>m.local_speaker===local);let scores=document.createElement('p');scores.textContent=(data.mixed.includes(local)?'Inconsistent voice evidence: inspect and split this group. ':'')+'Candidates: '+((match?.candidates||[]).map(c=>c.speaker+' similarity '+c.similarity.toFixed(3)).join(' · ')||'No existing reference profiles');section.append(scores);
 let select=document.createElement('select');select.dataset.local=local;options(select,false);section.append(select);
 let samples=document.createElement('div');samples.className='samples';
 (data.windows[local]||[]).forEach((w,i)=>{let item=document.createElement('label');item.className='sample';let check=document.createElement('input');check.type='checkbox';check.checked=!!w.reference_eligible&&w.retained!==false;check.disabled=!w.reference_eligible;check.dataset.window=local+':'+i;item.append(check,document.createTextNode('Reference '+(i+1)+(w.retained===false?' (different voice?) ':' ')),button(time(w.start)+'–'+time(w.end),w.start,w.end));samples.append(item)});section.append(samples);
 let details=document.createElement('details');let summary=document.createElement('summary');summary.textContent='Inspect all turns / override individual speakers';details.append(summary);
 for(const t of data.turns.filter(t=>(t.local_speaker||t.speaker)===local)){let row=document.createElement('div');row.className='turn';row.append(button(time(t.start)+'–'+time(t.end),t.start,t.end));let text=document.createElement('p');text.textContent=t.text;row.append(text);let choice=document.createElement('select');choice.dataset.turn=t.id;options(choice,true);row.append(choice);for(const key of ['start','end']){let label=document.createElement('label');label.textContent=key+' seconds ';let input=document.createElement('input');input.type='number';input.step='0.001';input.min=t.start;input.max=t.end;input.value=t[key];input.dataset[key]=t.id;label.append(input);row.append(label)}let note=document.createElement('small');note.textContent=' Narrow the time range to correct part of a turn. Words are assigned by their aligned midpoint.';row.append(note);details.append(row)}section.append(details);byId('cards').append(section)}
byId('add').onclick=()=>{const label=byId('name').value.trim();if(!label)return;newProfiles['new:'+label]={label,role:byId('role').value};document.querySelectorAll('select[data-local],select[data-turn]').forEach(s=>options(s,!!s.dataset.turn));byId('name').value=''};
byId('save').onclick=()=>{const assignments={},turn_overrides={},range_overrides=[],exclude_windows=[];
 document.querySelectorAll('select[data-local]').forEach(s=>{if(s.value)assignments[s.dataset.local]=s.value});document.querySelectorAll('select[data-turn]').forEach(s=>{if(s.value){const t=data.turns.find(t=>t.id===s.dataset.turn),start=Number(document.querySelector('input[data-start="'+t.id+'"]').value),end=Number(document.querySelector('input[data-end="'+t.id+'"]').value);if(start===t.start&&end===t.end)turn_overrides[t.id]=s.value;else range_overrides.push({turn_id:t.id,start,end,profile:s.value})}});document.querySelectorAll('input[data-window]').forEach(s=>{if(!s.checked)exclude_windows.push(s.dataset.window)});
 if(!Object.keys(assignments).length&&!Object.keys(turn_overrides).length&&!range_overrides.length){byId('message').textContent='Choose at least one group or turn.';return}
 const result={review_id:data.review_id,packet:data.packet,assignments,turn_overrides,range_overrides,new_profiles:newProfiles,exclude_windows};const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=data.packet.replace('.json','.decisions.json');a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);byId('message').textContent='Apply the downloaded file with: ./transcribe-media --apply-speaker-review /path/to/decisions.json';};
</script></html>'''
