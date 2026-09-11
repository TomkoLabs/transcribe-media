"""Leave-one-recording-out evidence checks; no invented probability estimates."""
from __future__ import annotations

import copy
from pathlib import Path

from .speakers import VoiceRegistry
from .storage import atomic_write_json, utc_now


def evaluate_registry(review_dir):
    registry = VoiceRegistry(Path(review_dir) / "speaker_registry.json")
    with registry._locked():
        data = registry._load_unlocked()
    verified = {key: [item for item in profile["observations"] if item.get("verified")]
                for key, profile in data["profiles"].items()}
    trials = []
    for truth, observations in verified.items():
        for observation in observations:
            # Remove this recording from every profile, not just the true speaker.
            source = observation["source_fingerprint"]
            profiles = copy.deepcopy(data["profiles"])
            for profile in profiles.values():
                profile["observations"] = [item for item in profile["observations"]
                                            if item.get("verified") and item["source_fingerprint"] != source]
            if not profiles[truth]["observations"]:
                continue
            scores = sorted(((registry.profile_similarity(profile, observation["embedding"], True), key)
                             for key, profile in profiles.items()), reverse=True)
            score, winner = scores[0]
            margin = score - (scores[1][0] if len(scores) > 1 else -1.)
            threshold = 0.78 if profiles[winner].get("role") == "child" else 0.70
            accepted = score >= threshold and margin >= 0.15
            trials.append({"truth": truth, "predicted": winner if accepted else None,
                           "score": round(score, 4), "margin": round(margin, 4),
                           "held_out_source": observation["source"],
                           "start": observation.get("start"), "end": observation.get("end")})
    matched = sum(item["predicted"] is not None for item in trials)
    wrong = sum(item["predicted"] is not None and item["predicted"] != item["truth"] for item in trials)
    report = {"created_utc": utc_now(), "evaluation": "leave_one_recording_out_reference_windows",
              "profile_count": len(verified), "trials": len(trials), "accepted": matched,
              "incorrect_accepted": wrong, "abstentions": len(trials) - matched,
              "coverage": matched / len(trials) if trials else None,
              "accuracy_among_accepted": (matched - wrong) / matched if matched else None,
              "calibrated_probability_available": False,
              "limitations": ["Reviewed reference clips are selected evidence, not an independent test corpus.",
                  "Window-level checks do not establish conversation-level WER, DER or identity error rates.",
                  "One recording cannot measure cross-recording generalization.",
                  "Zero observed false matches is not proof of zero risk. Child and unfamiliar-speaker testing is still needed."],
              "details": trials}
    atomic_write_json(Path(review_dir) / "voice-evaluation.json", report)
    return report
