from __future__ import annotations

import copy
import fcntl
import json
import math
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import __version__
from .storage import stable_hash, utc_now

SPEAKER_REGISTRY_SCHEMA_VERSION = "1.0"
SPEAKER_EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
SPEAKER_EMBEDDING_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
DEFAULT_MATCH_THRESHOLD = 0.45
DEFAULT_MATCH_MARGIN = 0.12
DEFAULT_LOCAL_MERGE_THRESHOLD = 0.72
DEFAULT_ENROLLMENT_SECONDS = 12.0
MIN_UPDATE_SECONDS = 4.0
MIN_ENROLLMENT_COHESION = 0.40
MIN_LOCAL_MERGE_SECONDS = 2.0
MIN_LOCAL_MERGE_WINDOWS = 2
MAX_PROFILE_OBSERVATIONS = 20


class SpeakerRegistryError(RuntimeError):
    pass


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _normalize(vector: Iterable[Any]) -> list[float]:
    values = [_number(item) for item in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if not values or norm <= 1e-8:
        raise SpeakerRegistryError("speaker embedding is empty or has zero length")
    return [round(value / norm, 7) for value in values]


def cosine_similarity(left: Iterable[Any], right: Iterable[Any]) -> float:
    left_values = list(left)
    right_values = list(right)
    if not left_values or len(left_values) != len(right_values):
        raise SpeakerRegistryError("speaker embedding dimensions do not match")
    normalized_left = _normalize(left_values)
    normalized_right = _normalize(right_values)
    return sum(
        first * second
        for first, second in zip(normalized_left, normalized_right, strict=True)
    )


def _centroid(vectors: Iterable[Iterable[Any]]) -> list[float]:
    normalized = [_normalize(vector) for vector in vectors]
    if not normalized:
        raise SpeakerRegistryError("cannot build a profile without observations")
    dimension = len(normalized[0])
    if any(len(vector) != dimension for vector in normalized):
        raise SpeakerRegistryError(
            "speaker profile contains mixed embedding dimensions"
        )
    average = [
        sum(vector[index] for vector in normalized) / len(normalized)
        for index in range(dimension)
    ]
    return _normalize(average)


def _weighted_centroid(
    vectors: Iterable[tuple[Iterable[Any], float]],
) -> list[float]:
    normalized = [
        (_normalize(vector), max(_number(weight), 0.0)) for vector, weight in vectors
    ]
    if not normalized:
        raise SpeakerRegistryError("cannot build a profile without observations")
    dimension = len(normalized[0][0])
    if any(len(vector) != dimension for vector, _ in normalized):
        raise SpeakerRegistryError(
            "speaker profile contains mixed embedding dimensions"
        )
    total_weight = sum(weight for _, weight in normalized)
    if total_weight <= 0:
        return _centroid(vector for vector, _ in normalized)
    average = [
        sum(vector[index] * weight for vector, weight in normalized) / total_weight
        for index in range(dimension)
    ]
    return _normalize(average)


def overlapping_speaker_pairs(
    timeline: Iterable[dict[str, Any]],
    *,
    minimum_overlap_seconds: float = 0.25,
) -> set[frozenset[str]]:
    """Return speaker pairs with enough simultaneous speech to forbid merging."""
    intervals = sorted(
        (
            max(0.0, _number(item.get("start"))),
            max(0.0, _number(item.get("end"))),
            str(item.get("speaker") or ""),
        )
        for item in timeline
        if item.get("speaker") not in (None, "", "SPEAKER_UNKNOWN")
    )
    active: list[tuple[float, float, str]] = []
    totals: dict[frozenset[str], float] = {}
    for start, end, speaker in intervals:
        if end <= start:
            continue
        active = [item for item in active if item[1] > start]
        for other_start, other_end, other_speaker in active:
            if other_speaker == speaker:
                continue
            overlap = min(end, other_end) - max(start, other_start)
            if overlap <= 0:
                continue
            pair = frozenset((speaker, other_speaker))
            totals[pair] = totals.get(pair, 0.0) + overlap
        active.append((start, end, speaker))
    return {
        pair for pair, seconds in totals.items() if seconds >= minimum_overlap_seconds
    }


class VoiceRegistry:
    """Atomic, project-scoped store for anonymous cross-recording voice IDs."""

    def __init__(
        self,
        path: Path,
        *,
        model: str = SPEAKER_EMBEDDING_MODEL,
        model_revision: str = SPEAKER_EMBEDDING_REVISION,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
        match_margin: float = DEFAULT_MATCH_MARGIN,
        local_merge_threshold: float = DEFAULT_LOCAL_MERGE_THRESHOLD,
        enrollment_seconds: float = DEFAULT_ENROLLMENT_SECONDS,
        known_voices: Iterable[str] = (),
        learn: bool = True,
        reviewed: bool = False,
    ) -> None:
        self.path = path
        self.lock_path = path.with_name(f".{path.name}.lock")
        self.model = model
        self.model_revision = model_revision
        self.match_threshold = match_threshold
        self.match_margin = match_margin
        self.local_merge_threshold = local_merge_threshold
        self.enrollment_seconds = enrollment_seconds
        self.known_voices = tuple(sorted(set(known_voices)))
        self.learn = learn
        self.reviewed = reviewed

    def _empty(self) -> dict[str, Any]:
        return {
            "schema_version": SPEAKER_REGISTRY_SCHEMA_VERSION,
            "program_version": __version__,
            "created_utc": utc_now(),
            "updated_utc": utc_now(),
            "revision": 0,
            "embedding_model": {
                "name": self.model,
                "revision": self.model_revision,
                "dimension": None,
            },
            "next_voice_number": 1,
            "profiles": {},
        }

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.chmod(self.lock_path, 0o600)
            with os.fdopen(descriptor, "a+") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise

    def _load_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise SpeakerRegistryError(
                f"speaker registry is unreadable: {self.path}: {exc}"
            ) from exc
        if (
            not isinstance(data, dict)
            or data.get("schema_version") != SPEAKER_REGISTRY_SCHEMA_VERSION
        ):
            raise SpeakerRegistryError(
                "speaker registry has an unsupported or invalid schema"
            )
        model = data.get("embedding_model") or {}
        if not isinstance(model, dict):
            raise SpeakerRegistryError("speaker registry embedding model is invalid")
        if (
            model.get("name") != self.model
            or model.get("revision") != self.model_revision
        ):
            raise SpeakerRegistryError(
                "speaker registry embedding model does not match this program version"
            )
        profiles = data.get("profiles")
        if not isinstance(profiles, dict):
            raise SpeakerRegistryError("speaker registry profiles are invalid")
        dimension = model.get("dimension")
        try:
            revision = int(data.get("revision"))
            next_voice_number = int(data.get("next_voice_number"))
        except (TypeError, ValueError) as exc:
            raise SpeakerRegistryError("speaker registry counters are invalid") from exc
        if revision < 0 or next_voice_number < 1:
            raise SpeakerRegistryError("speaker registry counters are invalid")
        observation_ids: set[str] = set()
        for voice_id, profile in profiles.items():
            if (
                not isinstance(voice_id, str)
                or not voice_id.startswith("VOICE_")
                or not isinstance(profile, dict)
                or profile.get("voice_id") != voice_id
            ):
                raise SpeakerRegistryError(f"speaker profile {voice_id!r} is invalid")
            centroid = profile.get("centroid") or []
            try:
                valid_dimension = int(dimension)
            except (TypeError, ValueError) as exc:
                raise SpeakerRegistryError(
                    "speaker registry embedding dimension is invalid"
                ) from exc
            if (
                valid_dimension < 1
                or not isinstance(centroid, list)
                or len(centroid) != valid_dimension
            ):
                raise SpeakerRegistryError(
                    f"speaker profile {voice_id!r} has an invalid embedding"
                )
            _normalize(centroid)
            observations = profile.get("observations")
            if not isinstance(observations, list) or (not observations and profile.get("reference_status") != "needs_review"):
                raise SpeakerRegistryError(
                    f"speaker profile {voice_id!r} has no observations"
                )
            for observation in observations:
                if not isinstance(observation, dict):
                    raise SpeakerRegistryError(
                        f"speaker profile {voice_id!r} has an invalid observation"
                    )
                observation_id = observation.get("observation_id")
                embedding = observation.get("embedding") or []
                if (
                    not isinstance(observation_id, str)
                    or not observation_id
                    or observation_id in observation_ids
                    or not isinstance(embedding, list)
                    or len(embedding) != valid_dimension
                ):
                    raise SpeakerRegistryError(
                        f"speaker profile {voice_id!r} has an invalid observation"
                    )
                observation_ids.add(observation_id)
                _normalize(embedding)
            if not observations:
                continue
            expected_centroid = _centroid(
                observation["embedding"] for observation in observations
            )
            if any(
                abs(left - right) > 2e-6
                for left, right in zip(
                    _normalize(centroid), expected_centroid, strict=True
                )
            ):
                raise SpeakerRegistryError(
                    f"speaker profile {voice_id!r} centroid is inconsistent"
                )
        return data

    def validate(self) -> dict[str, Any]:
        with self._locked():
            data = self._load_unlocked()
            self._validate_matching_policy(data)
        return self._summary(data)

    def profiles_for_review(self) -> list[dict[str, Any]]:
        with self._locked():
            data = self._load_unlocked()
        return [{"voice_id": key, "label": item.get("label", key),
                 "role": item.get("role", "unspecified"),
                 "verified_windows": sum(bool(obs.get("verified")) for obs in item["observations"]),
                 "verified_sessions": len({obs["source_fingerprint"] for obs in item["observations"] if obs.get("verified")})}
                for key, item in sorted(data["profiles"].items())]

    @staticmethod
    def profile_similarity(profile, embedding, reviewed=False):
        if not profile.get("observations"):
            return -1.0
        if not reviewed:
            return cosine_similarity(embedding, profile["centroid"])
        sessions: dict[str, list] = {}
        for observation in profile["observations"]:
            if observation.get("verified") and observation.get("clean_seconds", 0.) >= 2.5:
                sessions.setdefault(observation["source_fingerprint"], []).append(observation["embedding"])
        scores = []
        for vectors in sessions.values():
            if len(vectors) < 2:
                continue
            # Two supporting reviewed clips are required in a recording condition.
            support = sorted((cosine_similarity(embedding, vector) for vector in vectors), reverse=True)
            scores.append(min(cosine_similarity(embedding, _centroid(vectors)), support[1]))
        return max(scores, default=-1.0)

    def confirm_references(self, references, new_profiles=None, review_id=None, replace_source_fingerprint=None):
        """Apply validated human decisions atomically; never reinterpret a score as probability."""
        with self._locked():
            data = self._load_unlocked()
            if review_id and review_id in data.get("applied_reviews", {}):
                return data["applied_reviews"][review_id]
            if replace_source_fingerprint:
                for profile in data["profiles"].values():
                    profile["observations"] = [item for item in profile["observations"]
                                               if not (item.get("verified") and item["source_fingerprint"] == replace_source_fingerprint)]
            mapping = {}
            for key, metadata in sorted((new_profiles or {}).items()):
                voice_id = self._new_voice_id(data)
                mapping[key] = voice_id
                data["profiles"][voice_id] = {"voice_id": voice_id, "label": metadata["label"],
                    "role": metadata.get("role", "unspecified"), "created_utc": utc_now(),
                    "observations": [], "centroid": []}
            for reference in references:
                voice_id = mapping.get(reference["profile"], reference["profile"])
                if voice_id not in data["profiles"]:
                    raise SpeakerRegistryError(f"unknown reviewed profile {voice_id}")
                observation = copy.deepcopy(reference["observation"])
                observation["embedding"] = _normalize(observation["embedding"])
                dimension = data["embedding_model"]["dimension"]
                if dimension is not None and len(observation["embedding"]) != dimension:
                    raise SpeakerRegistryError("reviewed reference uses a different embedding model")
                data["embedding_model"]["dimension"] = len(observation["embedding"])
                observation.update(verified=True, review_id=review_id, verified_utc=utc_now())
                self._add_observation(data, voice_id, observation)
            for voice_id in mapping.values():
                if not data["profiles"][voice_id]["observations"]:
                    raise SpeakerRegistryError("a new profile needs at least one clean reviewed reference window")
            self._refresh_profiles(data)
            if review_id:
                data.setdefault("applied_reviews", {})[review_id] = mapping
            data.setdefault("audit", []).append({"action": "human_review", "review_id": review_id,
                "created_utc": utc_now(), "references": len(references), "new_profiles": mapping})
            data["revision"] += 1
            self._save_unlocked(data)
            return mapping

    def merge_profiles(self, source_id, target_id):
        with self._locked():
            data = self._load_unlocked()
            if source_id == target_id or any(key not in data["profiles"] for key in (source_id, target_id)):
                raise SpeakerRegistryError("merge requires two different existing VOICE IDs")
            source = data["profiles"].pop(source_id)
            data["profiles"][target_id]["observations"].extend(source["observations"])
            aliases = data.setdefault("aliases", {})
            for alias, destination in list(aliases.items()):
                if destination == source_id:
                    aliases[alias] = target_id
            aliases[source_id] = target_id
            data.setdefault("audit", []).append({"action": "merge", "source": source_id,
                "target": target_id, "created_utc": utc_now()})
            self._refresh_profiles(data)
            data["revision"] += 1
            self._save_unlocked(data)
            return aliases

    def _validate_matching_policy(self, data: dict[str, Any]) -> None:
        missing = set(self.known_voices) - set(data["profiles"])
        if missing:
            raise SpeakerRegistryError(
                "known VOICE IDs are absent from this registry: "
                + ", ".join(sorted(missing))
            )
        if not self.learn and not data["profiles"]:
            raise SpeakerRegistryError(
                "--no-speaker-learning requires an existing, nonempty voice registry"
            )

    def _save_unlocked(self, data: dict[str, Any]) -> None:
        data["program_version"] = __version__
        data["updated_utc"] = utc_now()
        content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                descriptor = -1
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self.path)
            self.path.chmod(0o600)
            directory_descriptor = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _summary(data: dict[str, Any]) -> dict[str, Any]:
        profiles = data.get("profiles") or {}
        public_state = {
            "schema_version": data.get("schema_version"),
            "revision": data.get("revision"),
            "embedding_model": data.get("embedding_model"),
            "profile_ids": sorted(profiles),
            "profile_observations": {
                voice_id: [
                    {
                        "observation_id": item.get("observation_id"),
                        "embedding_hash": stable_hash(item.get("embedding") or []),
                    }
                    for item in profile.get("observations") or []
                ]
                for voice_id, profile in sorted(profiles.items())
            },
            "profile_centroid_hashes": {
                voice_id: stable_hash(profile.get("centroid") or [])
                for voice_id, profile in sorted(profiles.items())
            },
        }
        return {
            "schema_version": data.get("schema_version"),
            "revision": int(data.get("revision") or 0),
            "profile_count": len(profiles),
            "embedding_model": copy.deepcopy(data.get("embedding_model")),
            "state_hash": stable_hash(public_state),
        }

    @staticmethod
    def _new_voice_id(data: dict[str, Any]) -> str:
        profiles = data["profiles"]
        number = max(1, int(data.get("next_voice_number") or 1))
        while f"VOICE_{number:04d}" in profiles:
            number += 1
        data["next_voice_number"] = number + 1
        return f"VOICE_{number:04d}"

    @staticmethod
    def _eligible_for_enrollment(
        evidence: dict[str, Any], enrollment_seconds: float
    ) -> bool:
        recognized_speech_is_sufficient = not evidence.get("word_timing_available") or (
            int(evidence.get("recognized_word_count") or 0) >= 3
            and _number(evidence.get("recognized_speech_seconds")) >= 1.0
        )
        return (
            _number(evidence.get("clean_seconds")) >= enrollment_seconds
            and int(evidence.get("window_count") or 0) >= 3
            and _number(evidence.get("cohesion")) >= MIN_ENROLLMENT_COHESION
            and bool(evidence.get("embedding"))
            and recognized_speech_is_sufficient
        )

    @staticmethod
    def _eligible_for_update(evidence: dict[str, Any]) -> bool:
        recognized_speech_is_sufficient = not evidence.get("word_timing_available") or (
            int(evidence.get("recognized_word_count") or 0) >= 2
            and _number(evidence.get("recognized_speech_seconds")) >= 0.5
        )
        return (
            _number(evidence.get("clean_seconds")) >= MIN_UPDATE_SECONDS
            and int(evidence.get("window_count") or 0) >= 2
            and _number(evidence.get("cohesion")) >= MIN_ENROLLMENT_COHESION
            and bool(evidence.get("embedding"))
            and recognized_speech_is_sufficient
        )

    @staticmethod
    def _eligible_for_match(evidence: dict[str, Any]) -> bool:
        recognized_speech_is_sufficient = not evidence.get("word_timing_available") or (
            int(evidence.get("recognized_word_count") or 0) >= 1
            and _number(evidence.get("recognized_speech_seconds")) >= 0.2
        )
        return (
            _number(evidence.get("clean_seconds")) >= 2.0
            and int(evidence.get("window_count") or 0) >= 1
            and _number(evidence.get("cohesion")) >= MIN_ENROLLMENT_COHESION
            and bool(evidence.get("embedding"))
            and recognized_speech_is_sufficient
        )

    @staticmethod
    def _observation(
        source_key: str,
        source_fingerprint: str,
        local_speaker: str,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "observation_id": stable_hash(
                {
                    "source_fingerprint": source_fingerprint,
                    "local_speaker": local_speaker,
                }
            ),
            "source": source_key,
            "source_fingerprint": source_fingerprint,
            "local_speaker": local_speaker,
            "created_utc": utc_now(),
            "clean_seconds": round(_number(evidence.get("clean_seconds")), 3),
            "window_count": int(evidence.get("window_count") or 0),
            "cohesion": round(_number(evidence.get("cohesion")), 4),
            "embedding": _normalize(evidence.get("embedding") or []),
        }

    @staticmethod
    def _same_observation(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return (
            left.get("observation_id") == right.get("observation_id")
            and left.get("embedding") == right.get("embedding")
            and left.get("clean_seconds") == right.get("clean_seconds")
            and left.get("window_count") == right.get("window_count")
            and left.get("cohesion") == right.get("cohesion")
        )

    def _add_observation(
        self,
        data: dict[str, Any],
        voice_id: str,
        observation: dict[str, Any],
    ) -> bool:
        existing_profile: str | None = None
        existing_observation: dict[str, Any] | None = None
        for candidate_id, profile in data["profiles"].items():
            for item in profile.get("observations") or []:
                if item.get("observation_id") == observation["observation_id"]:
                    existing_profile = candidate_id
                    existing_observation = item
                    break
            if existing_profile:
                break
        if (
            existing_profile == voice_id
            and existing_observation is not None
            and bool(existing_observation.get("verified")) == bool(observation.get("verified"))
            and self._same_observation(existing_observation, observation)
        ):
            return False

        if existing_observation and existing_observation.get("verified") and not observation.get("verified"):
            return False

        for profile in data["profiles"].values():
            profile["observations"] = [
                item
                for item in profile.get("observations") or []
                if item.get("observation_id") != observation["observation_id"]
            ]
        profile = data["profiles"][voice_id]
        observations = list(profile.get("observations") or [])
        observations.append(observation)
        anchors = [item for item in observations if item.get("verified")]
        automatic = [item for item in observations if not item.get("verified")]
        profile["observations"] = anchors + automatic[-MAX_PROFILE_OBSERVATIONS:]
        profile["last_seen_utc"] = utc_now()
        return True

    @staticmethod
    def _refresh_profiles(data: dict[str, Any]) -> None:
        for voice_id, profile in data["profiles"].items():
            observations = profile.get("observations") or []
            if not observations:
                profile["reference_status"] = "needs_review"
                profile["sessions_seen"] = 0
                profile["clean_seconds"] = 0.
                continue
            profile["reference_status"] = "available"
            profile["centroid"] = _centroid(item["embedding"] for item in observations)
            profile["sessions_seen"] = len(
                {item.get("source_fingerprint") for item in observations}
            )
            profile["clean_seconds"] = round(
                sum(
                    min(30.0, _number(item.get("clean_seconds")))
                    for item in observations
                ),
                3,
            )

    @staticmethod
    def _eligible_for_local_merge(evidence: dict[str, Any]) -> bool:
        return (
            _number(evidence.get("clean_seconds")) >= MIN_LOCAL_MERGE_SECONDS
            and int(evidence.get("window_count") or 0) >= MIN_LOCAL_MERGE_WINDOWS
            and _number(evidence.get("cohesion")) >= MIN_ENROLLMENT_COHESION
            and bool(evidence.get("embedding"))
        )

    def _local_groups(
        self,
        local_speakers: Iterable[str],
        usable: dict[str, dict[str, Any]],
        incompatible_pairs: set[frozenset[str]],
        minimum_groups: int | None,
    ) -> tuple[list[tuple[str, ...]], dict[frozenset[str], float]]:
        """Complete-link merge likely duplicate clusters from one recording."""
        speakers = sorted(set(local_speakers))
        similarities: dict[frozenset[str], float] = {}
        candidates: list[tuple[float, str, str]] = []
        for index, first in enumerate(speakers):
            if self.reviewed:
                # In quality mode cluster merges are human decisions. A toddler's
                # short cluster must not disappear into a nearby adult prototype.
                break
            first_evidence = usable.get(first)
            if first_evidence is None or not self._eligible_for_local_merge(
                first_evidence
            ):
                continue
            for second in speakers[index + 1 :]:
                second_evidence = usable.get(second)
                pair = frozenset((first, second))
                if (
                    pair in incompatible_pairs
                    or second_evidence is None
                    or not self._eligible_for_local_merge(second_evidence)
                ):
                    continue
                similarity = cosine_similarity(
                    first_evidence["embedding"], second_evidence["embedding"]
                )
                similarities[pair] = similarity
                if similarity >= self.local_merge_threshold:
                    candidates.append((similarity, first, second))

        groups: list[set[str]] = [{speaker} for speaker in speakers]
        for _similarity, first, second in sorted(candidates, reverse=True):
            if minimum_groups is not None and len(groups) <= minimum_groups:
                break
            first_group = next(group for group in groups if first in group)
            second_group = next(group for group in groups if second in group)
            if first_group is second_group:
                continue
            can_merge = all(
                frozenset((left, right)) not in incompatible_pairs
                and similarities.get(frozenset((left, right)), -1.0)
                >= self.local_merge_threshold
                for left in first_group
                for right in second_group
            )
            if not can_merge:
                continue
            first_group.update(second_group)
            groups.remove(second_group)
        return sorted(tuple(sorted(group)) for group in groups), similarities

    @staticmethod
    def _group_evidence(
        group: tuple[str, ...],
        usable: dict[str, dict[str, Any]],
        similarities: dict[frozenset[str], float],
    ) -> dict[str, Any] | None:
        items = [usable[speaker] for speaker in group if speaker in usable]
        if not items:
            return None
        embedding = _weighted_centroid(
            (
                item["embedding"],
                min(30.0, max(1.0, _number(item.get("clean_seconds")))),
            )
            for item in items
        )
        cohesion_values = [_number(item.get("cohesion")) for item in items]
        for index, first in enumerate(group):
            for second in group[index + 1 :]:
                pair_similarity = similarities.get(frozenset((first, second)))
                if pair_similarity is not None:
                    cohesion_values.append(pair_similarity)
        return {
            "embedding": embedding,
            "windows": [window for item in items for window in item.get("windows", [])],
            "suspected_mixed_speakers": any(item.get("suspected_mixed_speakers") for item in items),
            "clean_seconds": sum(_number(item.get("clean_seconds")) for item in items),
            "available_clean_seconds": sum(
                _number(item.get("available_clean_seconds")) for item in items
            ),
            "window_count": sum(int(item.get("window_count") or 0) for item in items),
            "cohesion": min(cohesion_values, default=0.0),
            "recognized_word_count": sum(
                int(item.get("recognized_word_count") or 0) for item in items
            ),
            "recognized_speech_seconds": sum(
                _number(item.get("recognized_speech_seconds")) for item in items
            ),
            "word_timing_available": any(
                bool(item.get("word_timing_available")) for item in items
            ),
            "lexical_filter_applied": all(
                bool(item.get("lexical_filter_applied")) for item in items
            ),
        }

    def identify(
        self,
        *,
        source_key: str,
        source_fingerprint: str,
        local_speakers: Iterable[str],
        evidence: dict[str, dict[str, Any]],
        incompatible_pairs: Iterable[frozenset[str]] = (),
        minimum_groups: int | None = None,
    ) -> dict[str, Any]:
        """Reconcile local clusters, match profiles, and safely learn observations."""
        local_speakers = tuple(sorted(set(local_speakers)))
        if minimum_groups is not None and minimum_groups < 1:
            raise SpeakerRegistryError("minimum speaker groups must be at least one")
        incompatible = {
            frozenset(pair) for pair in incompatible_pairs if len(pair) == 2
        }
        with self._locked():
            existed = self.path.exists()
            data = self._load_unlocked()
            self._validate_matching_policy(data)
            profiles = data["profiles"]
            profile_ids = list(self.known_voices) or sorted(profiles)
            initial_profile_count = len(profile_ids)
            dimension = data["embedding_model"].get("dimension")

            usable: dict[str, dict[str, Any]] = {}
            for local_speaker, item in evidence.items():
                embedding = _normalize(item.get("embedding") or [])
                if dimension is None:
                    dimension = len(embedding)
                    data["embedding_model"]["dimension"] = dimension
                if len(embedding) != int(dimension):
                    raise SpeakerRegistryError(
                        "speaker evidence does not match registry embedding dimension"
                    )
                usable[local_speaker] = {**item, "embedding": embedding}

            groups, local_similarities = self._local_groups(
                local_speakers, usable, incompatible, minimum_groups
            )
            grouped_evidence = {
                group: self._group_evidence(group, usable, local_similarities)
                for group in groups
            }
            scored: dict[tuple[str, ...], list[tuple[str, float]]] = {}
            for group, item in grouped_evidence.items():
                if item is None or not self._eligible_for_match(item):
                    continue
                scored[group] = sorted(
                    (
                        (
                            voice_id,
                            self.profile_similarity(profiles[voice_id], item["embedding"], self.reviewed),
                        )
                        for voice_id in profile_ids
                    ),
                    key=lambda pair: pair[1],
                    reverse=True,
                )

            proposals: list[tuple[float, tuple[str, ...], str, float]] = []
            for group, candidates in scored.items():
                if not candidates:
                    continue
                voice_id, score = candidates[0]
                runner_up = candidates[1][1] if len(candidates) > 1 else -1.0
                margin = score - runner_up
                threshold = max(self.match_threshold, 0.70 if self.reviewed else -1.)
                required_margin = max(self.match_margin, 0.15 if self.reviewed else 0.)
                reliable = True
                if self.reviewed:
                    item = grouped_evidence[group]
                    windows = [window for window in item.get("windows", []) if window.get("retained", True)]
                    votes = 0
                    for window in windows:
                        window_scores = sorted((self.profile_similarity(profiles[key], window["embedding"], True), key)
                                               for key in profile_ids)
                        best_window, winner = window_scores[-1]
                        second_window = window_scores[-2][0] if len(window_scores) > 1 else -1.
                        votes += winner == voice_id and best_window >= threshold and best_window - second_window >= required_margin
                    reliable = bool(len(windows) >= 2 and votes / len(windows) >= 0.8
                                    and not item.get("suspected_mixed_speakers"))
                    if profiles[voice_id].get("role") == "child":
                        threshold = max(threshold, 0.78)
                if reliable and score >= threshold and margin >= required_margin:
                    proposals.append((score, group, voice_id, margin))

            group_assignments: dict[tuple[str, ...], dict[str, Any]] = {}
            used_profiles: set[str] = set()
            for score, group, voice_id, margin in sorted(proposals, reverse=True):
                if group in group_assignments or voice_id in used_profiles:
                    continue
                group_assignments[group] = {
                    "speaker": voice_id,
                    "status": "matched",
                    "similarity": round(score, 4),
                    "margin": round(margin, 4),
                }
                used_profiles.add(voice_id)

            dirty = not existed and self.learn
            new_voice_ceiling = self.match_threshold - self.match_margin / 2.0
            for group in groups:
                item = grouped_evidence[group]
                if group in group_assignments:
                    continue
                candidates = scored.get(group) or []
                best_score = candidates[0][1] if candidates else None
                can_enroll = item is not None and self._eligible_for_enrollment(
                    item, self.enrollment_seconds
                )
                registry_was_empty = initial_profile_count == 0
                sufficiently_novel = (
                    best_score is None or best_score <= new_voice_ceiling
                )
                enrollment_allowed = self.learn and not self.known_voices and not self.reviewed
                if (
                    enrollment_allowed
                    and can_enroll
                    and (registry_was_empty or sufficiently_novel)
                ):
                    voice_id = self._new_voice_id(data)
                    data["profiles"][voice_id] = {
                        "voice_id": voice_id,
                        "created_utc": utc_now(),
                        "last_seen_utc": utc_now(),
                        "sessions_seen": 0,
                        "clean_seconds": 0.0,
                        "centroid": item["embedding"],
                        "observations": [],
                    }
                    group_assignments[group] = {
                        "speaker": voice_id,
                        "status": "enrolled",
                        "similarity": (
                            round(best_score, 4) if best_score is not None else None
                        ),
                        "margin": None,
                    }
                    dirty = True
                else:
                    group_assignments[group] = {
                        "speaker": group[0],
                        "status": (
                            "unresolved_insufficient_audio"
                            if item is None or not can_enroll
                            else (
                                "unresolved_no_confident_known_match"
                                if not enrollment_allowed
                                else "unresolved_ambiguous"
                            )
                        ),
                        "similarity": (
                            round(best_score, 4) if best_score is not None else None
                        ),
                        "margin": None,
                    }

            for group, decision in group_assignments.items():
                if not self.learn:
                    continue
                if decision["status"] not in {"matched", "enrolled"}:
                    continue
                item = grouped_evidence[group]
                if item is None:
                    continue
                if decision["status"] == "matched" and not self._eligible_for_update(
                    item
                ):
                    continue
                if self.reviewed and (decision.get("similarity", 0.) < 0.85
                                      or decision.get("margin", 0.) < 0.20
                                      or item.get("clean_seconds", 0.) < 12):
                    continue
                observation = self._observation(
                    source_key,
                    source_fingerprint,
                    "|".join(group),
                    item,
                )
                if self._add_observation(data, str(decision["speaker"]), observation):
                    dirty = True

            if dirty:
                self._refresh_profiles(data)
                data["revision"] = int(data.get("revision") or 0) + 1
                self._save_unlocked(data)
            summary = self._summary(data)

        assignments: dict[str, dict[str, Any]] = {}
        group_reports = []
        for group in groups:
            pair_scores = [
                local_similarities[frozenset((first, second))]
                for index, first in enumerate(group)
                for second in group[index + 1 :]
                if frozenset((first, second)) in local_similarities
            ]
            decision = group_assignments[group]
            candidates = scored.get(group) or []
            # Retain alternatives for review without exposing biometric vectors.
            if self.reviewed and grouped_evidence[group]:
                candidates = sorted(((key, cosine_similarity(grouped_evidence[group]["embedding"], profiles[key]["centroid"]))
                                     for key in profile_ids), key=lambda pair: pair[1], reverse=True)
            candidate_report = [
                {"speaker": voice_id, "similarity": round(score, 4),
                 "score_kind": "centroid_cosine", "probability": None}
                for voice_id, score in candidates[:3]
            ]
            group_report = {
                "local_speakers": list(group),
                "speaker": decision["speaker"],
                "merged": len(group) > 1,
                "minimum_similarity": (
                    round(min(pair_scores), 4) if pair_scores else None
                ),
            }
            group_reports.append(group_report)
            for local_speaker in group:
                assignments[local_speaker] = {
                    **decision,
                    "local_cluster_merged": len(group) > 1,
                    "reconciled_local_speakers": list(group),
                    "local_cluster_similarity": group_report["minimum_similarity"],
                    "candidates": candidate_report,
                }

        matches = []
        for local_speaker in sorted(assignments):
            item = evidence.get(local_speaker) or {}
            matches.append(
                {
                    "local_speaker": local_speaker,
                    **assignments[local_speaker],
                    "clean_seconds": round(_number(item.get("clean_seconds")), 3),
                    "window_count": int(item.get("window_count") or 0),
                    "cohesion": (
                        round(_number(item.get("cohesion")), 4) if item else None
                    ),
                    "recognized_word_count": int(
                        item.get("recognized_word_count") or 0
                    ),
                    "recognized_speech_seconds": round(
                        _number(item.get("recognized_speech_seconds")), 3
                    ),
                }
            )
        active_speaker_ids = sorted(
            {str(decision["speaker"]) for decision in assignments.values()}
        )
        if len(active_speaker_ids) != len(groups):
            raise SpeakerRegistryError(
                "whole-recording identity reconciliation produced an inconsistent "
                "active speaker set"
            )
        return {
            "enabled": True,
            "scope": "whole_recording",
            "registry": str(self.path),
            "registry_schema_version": summary["schema_version"],
            "registry_revision": summary["revision"],
            "registry_state_hash": summary["state_hash"],
            "profile_count": summary["profile_count"],
            "embedding_model": summary["embedding_model"],
            "match_threshold": self.match_threshold,
            "match_margin": self.match_margin,
            "local_merge_threshold": self.local_merge_threshold,
            "enrollment_seconds": self.enrollment_seconds,
            "known_voices": list(self.known_voices),
            "learning_enabled": self.learn,
            "enrollment_enabled": self.learn and not self.known_voices and not self.reviewed,
            "reviewed_matching": self.reviewed,
            "score_kind": "verified_condition_consensus" if self.reviewed else "cosine_similarity",
            "scores_are_probabilities": False,
            "minimum_speaker_groups": minimum_groups,
            "active_speaker_count": len(active_speaker_ids),
            "active_speaker_ids": active_speaker_ids,
            "local_clusters_detected": len(local_speakers),
            "speaker_groups_after_reconciliation": len(groups),
            "merged_local_clusters": len(local_speakers) - len(groups),
            "overlap_conflict_pairs": len(incompatible),
            "local_cluster_groups": group_reports,
            "matches": matches,
        }


def apply_speaker_identities(
    result: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    decisions = {
        str(item["local_speaker"]): item for item in report.get("matches") or []
    }

    def apply(item: dict[str, Any]) -> None:
        local = str(
            item.get("local_speaker") or item.get("speaker") or "SPEAKER_UNKNOWN"
        )
        decision = decisions.get(local)
        item["local_speaker"] = local
        if decision is None:
            return
        item["speaker"] = decision["speaker"]
        item["speaker_identity"] = {
            key: value
            for key, value in decision.items()
            if key not in {"local_speaker", "clean_seconds", "window_count", "cohesion"}
        }

    for timeline_name in ("speaker_timeline", "speaker_assignment_timeline"):
        for interval in result.get(timeline_name) or []:
            apply(interval)
    for segment in result.get("segments") or []:
        apply(segment)
        for word in segment.get("words") or []:
            apply(word)
    result["speaker_identity"] = report
    return result
