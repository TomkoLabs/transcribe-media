from __future__ import annotations

import gc
import io
import logging
import math
import os
import re
import subprocess
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Optional, Protocol

from .analysis import ToneEstimator
from .schema import SAMPLE_RATE, RuntimeSettings
from .speakers import (
    SPEAKER_EMBEDDING_MODEL,
    SPEAKER_EMBEDDING_REVISION,
    cosine_similarity,
)

LOG = logging.getLogger(__name__)
SUPPORTED_CTRANSLATE2_VERSION = "4.7.2"
SPEAKER_REFINEMENT_VERSION = "1.2"
SPEAKER_REFINEMENT_MAX_RUN_SECONDS = 2.0
SPEAKER_REFINEMENT_CONTEXT_GAP_SECONDS = 1.25
SPEAKER_REFINEMENT_SENTENCE_SEAM_GAP_SECONDS = 0.12
SPEAKER_REFINEMENT_SENTENCE_SEAM_MAX_RUN_SECONDS = 12.0
SPEAKER_REFINEMENT_MIN_PROTOTYPE_SECONDS = 4.0
SPEAKER_REFINEMENT_MIN_SIMILARITY = 0.52
SPEAKER_REFINEMENT_SINGLE_CONTEXT_MARGIN = 0.14
SPEAKER_REFINEMENT_DOUBLE_CONTEXT_MARGIN = 0.10
SPEAKER_REFINEMENT_MICRO_RUN_SECONDS = 0.50
SPEAKER_REFINEMENT_MICRO_MIN_SIMILARITY = 0.42
SPEAKER_REFINEMENT_MICRO_MIN_MARGIN = 0.20
SPEAKER_REFINEMENT_UTTERANCE_GAP_SECONDS = 0.35
SPEAKER_REFINEMENT_UTTERANCE_MAX_SECONDS = 12.0
SPEAKER_REFINEMENT_UTTERANCE_MIN_SIMILARITY = 0.52
SPEAKER_REFINEMENT_UTTERANCE_MIN_MARGIN = 0.14
SORTFORMER_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"


def _model_cache(*parts: str) -> Path:
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    path = root / "transcribe-media" / "models"
    for part in parts:
        path /= part
    path.mkdir(parents=True, exist_ok=True)
    return path


def package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _load_speaker_encoder(device: str) -> Any:
    from huggingface_hub import snapshot_download
    from speechbrain.inference.speaker import EncoderClassifier

    model_path = snapshot_download(
        repo_id=SPEAKER_EMBEDDING_MODEL,
        revision=SPEAKER_EMBEDDING_REVISION,
    )
    return EncoderClassifier.from_hparams(
        source=model_path,
        savedir=str(_model_cache("speechbrain-ecapa")),
        run_opts={"device": device},
    )


class SpeechRecognitionBackend(Protocol):
    name: str

    def transcribe(
        self,
        source: Path,
        align: bool,
        verbose: bool,
    ) -> tuple[dict[str, Any], Any, dict[str, Any], list[str]]: ...


class SpeakerDiarizer(Protocol):
    name: str

    def assign(
        self,
        source: Path,
        audio: Any,
        result: dict[str, Any],
        min_speakers: Optional[int],
        max_speakers: Optional[int],
    ) -> dict[str, Any]: ...


class WhisperXBackend:
    name = "whisperx"

    def __init__(
        self,
        model_name: str,
        language: Optional[str],
        task: str,
        runtime: RuntimeSettings,
        hf_token: Optional[str],
    ) -> None:
        import whisperx

        self.whisperx = whisperx
        self.model_name = model_name
        self.language = language
        self.task = task
        self.runtime = runtime
        self.hf_token = hf_token
        self.model = self._load_model(runtime)
        self._align_language: Optional[str] = None
        self._align_model: Any = None
        self._align_metadata: Optional[dict[str, Any]] = None
        self._active_audio: Optional[tuple[Any, Path]] = None

    def _load_model(self, runtime: RuntimeSettings) -> Any:
        return self.whisperx.load_model(
            self.model_name,
            runtime.device,
            device_index=0,
            compute_type=runtime.compute_type,
            language=self.language,
            task=self.task,
            threads=runtime.threads,
            use_auth_token=self.hf_token,
        )

    @classmethod
    def create(
        cls,
        model_name: str,
        language: Optional[str],
        task: str,
        runtime: RuntimeSettings,
        hf_token: Optional[str],
    ) -> tuple["WhisperXBackend", RuntimeSettings]:
        try:
            return cls(model_name, language, task, runtime, hf_token), runtime
        except Exception:
            if runtime.device != "cuda" or not runtime.device_was_auto:
                raise
            fallback = RuntimeSettings(
                device="cpu",
                compute_type="int8",
                batch_size=1,
                threads=runtime.threads,
                device_was_auto=True,
                description="CUDA initialization failed; using CPU / int8",
            )
            return cls(model_name, language, task, fallback, hf_token), fallback

    def _alignment_model(self, language: str) -> tuple[Any, dict[str, Any]]:
        if self._align_language != language:
            self._align_model = None
            self._align_metadata = None
            gc.collect()
            if self.runtime.device == "cuda":
                import torch

                torch.cuda.empty_cache()
            model, metadata = self.whisperx.load_align_model(
                language_code=language,
                device=self.runtime.device,
            )
            self._align_language = language
            self._align_model = model
            self._align_metadata = metadata
        assert self._align_metadata is not None
        return self._align_model, self._align_metadata

    def transcribe(
        self,
        source: Path,
        align: bool,
        verbose: bool,
    ) -> tuple[dict[str, Any], Any, dict[str, Any], list[str]]:
        audio = self._decode_audio(source)
        batch_sizes = [self.runtime.batch_size]
        if self.runtime.device == "cuda":
            candidate = self.runtime.batch_size
            while candidate > 1:
                candidate = max(1, candidate // 2)
                if candidate not in batch_sizes:
                    batch_sizes.append(candidate)

        result: Optional[dict[str, Any]] = None
        last_error: Optional[Exception] = None
        used_batch_size = batch_sizes[0]
        for batch_size in batch_sizes:
            try:
                result = self.model.transcribe(
                    audio,
                    batch_size=batch_size,
                    chunk_size=30,
                    print_progress=verbose,
                )
                used_batch_size = batch_size
                break
            except Exception as exc:
                last_error = exc
                error_text = str(exc).lower()
                if (
                    self.runtime.device == "cuda"
                    and "parallel_for failed" in error_text
                    and "invalid device ordinal" in error_text
                ):
                    installed = package_version("ctranslate2") or "unknown"
                    raise RuntimeError(
                        "CTranslate2 "
                        f"{installed} failed inside a CUDA kernel on device 0. "
                        "CTranslate2 4.8.x has a known upstream regression with "
                        "this error; this project requires the stable "
                        f"{SUPPORTED_CTRANSLATE2_VERSION} release. Run "
                        "./install.sh to repair the managed environment, then "
                        "retry."
                    ) from exc
                if self.runtime.device != "cuda" or "memory" not in error_text:
                    raise
                gc.collect()
                import torch

                torch.cuda.empty_cache()
        if result is None:
            assert last_error is not None
            raise last_error

        language = result.get("language") or self.language
        degraded: list[str] = []
        alignment_model_name = None
        if align and self.task != "translate":
            try:
                if not language:
                    raise RuntimeError("speech recognizer did not return a language")
                align_model, align_metadata = self._alignment_model(language)
                aligned = self.whisperx.align(
                    result.get("segments") or [],
                    align_model,
                    align_metadata,
                    audio,
                    self.runtime.device,
                    return_char_alignments=False,
                    print_progress=verbose,
                )
                aligned["language"] = language
                result = aligned
                alignment_model_name = f"whisperx-default:{language}"
            except Exception as exc:
                degraded.append(f"alignment: {exc}")
        elif self.task == "translate" and align:
            degraded.append("alignment: unavailable for translation task")

        provenance = {
            "transcription_engine": self.name,
            "transcription_engine_version": package_version("whisperx"),
            "faster_whisper_version": package_version("faster-whisper"),
            "ctranslate2_version": package_version("ctranslate2"),
            "transcription_model": self.model_name,
            "alignment_model": alignment_model_name,
            "language_detected": language,
            "batch_size_used": used_batch_size,
        }
        return result, audio, provenance, degraded

    def _decode_audio(self, source: Path) -> Any:
        """Decode to a disk-backed float array so long files do not fill RAM."""
        import numpy as np

        self.release_audio()
        descriptor, raw_name = tempfile.mkstemp(
            prefix="transcribe-media-audio-", suffix=".f32"
        )
        os.close(descriptor)
        raw_path = Path(raw_name)
        try:
            completed = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-f",
                    "f32le",
                    "-y",
                    str(raw_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                detail = " ".join(completed.stderr.split())[-1500:]
                raise RuntimeError(f"FFmpeg could not decode audio: {detail}")
            if raw_path.stat().st_size < 4:
                raise RuntimeError("FFmpeg decoded no audio samples")
            audio = np.memmap(raw_path, dtype=np.float32, mode="r+")
            self._active_audio = (audio, raw_path)
            return audio
        except Exception:
            raw_path.unlink(missing_ok=True)
            raise

    def release_audio(self) -> None:
        if self._active_audio is None:
            return
        audio, raw_path = self._active_audio
        self._active_audio = None
        mapping = getattr(audio, "_mmap", None)
        if mapping is not None:
            mapping.close()
        raw_path.unlink(missing_ok=True)

    def __del__(self) -> None:
        try:
            self.release_audio()
        except Exception:
            pass


class PyannoteDiarizer:
    name = "pyannote/whisperx"

    def __init__(self, token: str, device: str, batch_size: int = 4) -> None:
        from whisperx.diarize import DiarizationPipeline

        self.model_name = "pyannote/speaker-diarization-community-1"
        self.device = device
        self.batch_size = max(1, batch_size)
        self.last_run: dict[str, Any] = {}
        self.pipeline = DiarizationPipeline(
            model_name=self.model_name,
            token=token,
            device=device,
        )
        self._set_batch_size(self.batch_size)

    def _set_batch_size(self, batch_size: int) -> None:
        """Bound both pyannote inference batches, including model-config defaults."""
        model = self.pipeline.model
        if hasattr(model, "segmentation_batch_size"):
            model.segmentation_batch_size = batch_size
        if hasattr(model, "embedding_batch_size"):
            model.embedding_batch_size = batch_size

    @staticmethod
    def _memory_failure(exc: BaseException) -> bool:
        message = str(exc).lower()
        return isinstance(exc, MemoryError) or any(
            marker in message
            for marker in (
                "out of memory",
                "batch_size",
                "cuda_error_out_of_memory",
                "cudnn_status_alloc_failed",
            )
        )

    def _move_to_cpu(self) -> None:
        import torch

        self.pipeline.model.to(torch.device("cpu"))
        self.device = "cpu"
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _annotation_frame(annotation: Any) -> Any:
        import pandas as pd

        records = [
            {
                "segment": segment,
                "label": track,
                "speaker": speaker,
                "start": float(segment.start),
                "end": float(segment.end),
            }
            for segment, track, speaker in annotation.itertracks(yield_label=True)
        ]
        return pd.DataFrame(
            records,
            columns=("segment", "label", "speaker", "start", "end"),
        )

    def _diarize(
        self,
        audio: Any,
        min_speakers: Optional[int],
        max_speakers: Optional[int],
    ) -> tuple[Any, Any, bool]:
        exact_speakers = (
            min_speakers
            if min_speakers is not None and min_speakers == max_speakers
            else None
        )
        options = (
            {"num_speakers": exact_speakers}
            if exact_speakers is not None
            else {
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            }
        )
        model = getattr(self.pipeline, "model", None)
        if not callable(model):
            diarization = self.pipeline(audio, **options)
            return diarization, diarization, False

        import numpy as np
        import torch

        samples = np.asarray(audio, dtype=np.float32)
        output = model(
            {
                "waveform": torch.from_numpy(samples[None, :]),
                "sample_rate": SAMPLE_RATE,
            },
            **options,
        )
        raw = self._annotation_frame(output.speaker_diarization)
        exclusive_annotation = getattr(output, "exclusive_speaker_diarization", None)
        if exclusive_annotation is None:
            return raw, raw, False
        exclusive = self._annotation_frame(exclusive_annotation)
        return raw, exclusive if not exclusive.empty else raw, not exclusive.empty

    def assign(
        self,
        source: Path,
        audio: Any,
        result: dict[str, Any],
        min_speakers: Optional[int],
        max_speakers: Optional[int],
    ) -> dict[str, Any]:
        del source
        import whisperx

        candidates = []
        candidate = self.batch_size
        while candidate >= 1:
            if candidate not in candidates:
                candidates.append(candidate)
            if candidate == 1:
                break
            candidate = max(1, candidate // 2)

        attempted: list[dict[str, Any]] = []
        diarization = None
        assignment_diarization = None
        used_exclusive_diarization = False
        for batch_size in candidates:
            self._set_batch_size(batch_size)
            try:
                (
                    diarization,
                    assignment_diarization,
                    used_exclusive_diarization,
                ) = self._diarize(audio, min_speakers, max_speakers)
                used_batch_size = batch_size
                break
            except Exception as exc:
                attempted.append(
                    {
                        "device": self.device,
                        "batch_size": batch_size,
                        "error": f"{type(exc).__name__}: {str(exc)}"[:500],
                    }
                )
                if not self._memory_failure(exc):
                    raise
                LOG.warning(
                    "pyannote memory pressure on %s at batch %d; retrying smaller",
                    self.device,
                    batch_size,
                )
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
        else:
            used_batch_size = 1

        if diarization is None and self.device == "cuda":
            LOG.warning("pyannote did not fit on CUDA at batch 1; retrying on CPU")
            self._move_to_cpu()
            self._set_batch_size(1)
            (
                diarization,
                assignment_diarization,
                used_exclusive_diarization,
            ) = self._diarize(audio, min_speakers, max_speakers)
            used_batch_size = 1
        elif diarization is None:
            last = attempted[-1]["error"] if attempted else "unknown memory failure"
            raise MemoryError(f"pyannote failed at batch size 1: {last}")

        timeline = []
        for _, row in diarization.iterrows():
            timeline.append(
                {
                    "start": float(row["start"]),
                    "end": float(row["end"]),
                    "speaker": str(row["speaker"]),
                }
            )
        assigned = whisperx.assign_word_speakers(assignment_diarization, result)
        assigned["speaker_timeline"] = timeline
        assigned["speaker_assignment_timeline"] = [
            {
                "start": float(row["start"]),
                "end": float(row["end"]),
                "speaker": str(row["speaker"]),
            }
            for _, row in assignment_diarization.iterrows()
        ]
        self.batch_size = used_batch_size
        self.last_run = {
            "device": self.device,
            "batch_size_used": used_batch_size,
            "attempts": attempted,
            "detected_speakers": len(
                {item["speaker"] for item in timeline if item.get("speaker")}
            ),
            "timeline_intervals": len(timeline),
            "word_assignment_timeline": (
                "exclusive" if used_exclusive_diarization else "standard"
            ),
        }
        return assigned


def _sortformer_speaker(label: Any) -> str:
    text = str(label)
    match = re.search(r"(\d+)$", text)
    if match:
        return f"SPEAKER_{int(match.group(1)):02d}"
    safe = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").upper()
    return f"SORTFORMER_{safe or 'UNKNOWN'}"


def _parse_sortformer_segment(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        values = item.split()
        if len(values) < 3:
            raise ValueError(f"invalid Sortformer segment: {item!r}")
        start, end, speaker = values[0], values[1], values[2]
    elif isinstance(item, dict):
        start = item.get("start", item.get("begin"))
        end = item.get("end")
        speaker = item.get("speaker", item.get("label"))
    elif isinstance(item, (tuple, list)) and len(item) >= 3:
        start, end, speaker = item[0], item[1], item[2]
    else:
        start = getattr(item, "start", getattr(item, "begin", None))
        end = getattr(item, "end", None)
        speaker = getattr(item, "speaker", getattr(item, "label", None))
    if start is None or end is None or speaker is None:
        raise ValueError(f"invalid Sortformer segment: {item!r}")
    start_value = max(0.0, float(start))
    end_value = max(start_value, float(end))
    return {
        "start": start_value,
        "end": end_value,
        "speaker": _sortformer_speaker(speaker),
        "model_speaker": str(speaker),
    }


class SortformerDiarizer:
    """NVIDIA Sortformer v2.1 long-form diarization through NeMo."""

    name = "nvidia/sortformer-v2.1"
    model_name = SORTFORMER_MODEL

    def __init__(self, device: str, model: Any = None) -> None:
        import torch

        if model is None:
            from nemo.collections.asr.models import SortformerEncLabelModel

            model = SortformerEncLabelModel.from_pretrained(self.model_name)
        self.device = device
        self.model = model.to(torch.device(device)) if hasattr(model, "to") else model
        if hasattr(self.model, "eval"):
            self.model.eval()
        modules = getattr(self.model, "sortformer_modules", None)
        if modules is not None:
            modules.chunk_len = 340
            modules.chunk_right_context = 40
            modules.fifo_len = 40
            modules.spkcache_update_period = 300
            if hasattr(modules, "spkcache_len"):
                modules.spkcache_len = 188
            check = getattr(modules, "_check_streaming_parameters", None)
            if callable(check):
                check()
        self.last_run: dict[str, Any] = {}

    def diarize(self, audio: Any) -> list[dict[str, Any]]:
        import numpy as np
        import torch

        samples = np.asarray(audio, dtype=np.float32)
        with torch.inference_mode():
            predicted = self.model.diarize(
                audio=[samples],
                batch_size=1,
                sample_rate=SAMPLE_RATE,
            )
        raw_segments = predicted[0] if predicted else []
        timeline = [
            _parse_sortformer_segment(item)
            for item in raw_segments
        ]
        timeline = [item for item in timeline if item["end"] > item["start"]]
        if not timeline:
            raise RuntimeError("Sortformer returned no speaker segments")
        timeline.sort(key=lambda item: (item["start"], item["end"]))
        self.last_run = {
            "device": self.device,
            "batch_size_used": 1,
            "timeline_intervals": len(timeline),
            "detected_speakers": len({item["speaker"] for item in timeline}),
            "streaming_configuration": "30.4_second_high_accuracy",
        }
        return timeline

    def assign(
        self,
        source: Path,
        audio: Any,
        result: dict[str, Any],
        min_speakers: Optional[int],
        max_speakers: Optional[int],
    ) -> dict[str, Any]:
        del source, min_speakers, max_speakers
        import pandas as pd
        import whisperx

        timeline = self.diarize(audio)
        frame = pd.DataFrame(
            [
                {
                    "start": item["start"],
                    "end": item["end"],
                    "speaker": item["speaker"],
                }
                for item in timeline
            ]
        )
        assigned = whisperx.assign_word_speakers(frame, result)
        assigned["speaker_timeline"] = [dict(item) for item in timeline]
        assigned["speaker_assignment_timeline"] = [dict(item) for item in timeline]
        return assigned


def _timeline_overlap(
    left: dict[str, Any], right: dict[str, Any]
) -> float:
    return max(
        0.0,
        min(float(left["end"]), float(right["end"]))
        - max(float(left["start"]), float(right["start"])),
    )


def _map_secondary_timeline(
    primary: list[dict[str, Any]], secondary: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    primary_speakers = sorted({str(item["speaker"]) for item in primary})
    secondary_speakers = sorted({str(item["speaker"]) for item in secondary})
    if not primary_speakers or not secondary_speakers:
        return [], []
    matrix = np.zeros(
        (len(secondary_speakers), len(primary_speakers)), dtype=np.float64
    )
    secondary_indexes = {
        speaker: index for index, speaker in enumerate(secondary_speakers)
    }
    primary_indexes = {
        speaker: index for index, speaker in enumerate(primary_speakers)
    }
    for secondary_item in secondary:
        for primary_item in primary:
            matrix[
                secondary_indexes[str(secondary_item["speaker"])],
                primary_indexes[str(primary_item["speaker"])],
            ] += _timeline_overlap(secondary_item, primary_item)
    rows, columns = linear_sum_assignment(-matrix)
    mapping: dict[str, str] = {}
    reports: list[dict[str, Any]] = []
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        overlap = float(matrix[row, column])
        if overlap <= 0.0:
            continue
        secondary_speaker = secondary_speakers[row]
        primary_speaker = primary_speakers[column]
        total = sum(
            max(0.0, float(item["end"]) - float(item["start"]))
            for item in secondary
            if str(item["speaker"]) == secondary_speaker
        )
        mapping[secondary_speaker] = primary_speaker
        reports.append(
            {
                "sortformer_speaker": secondary_speaker,
                "primary_speaker": primary_speaker,
                "overlap_seconds": round(overlap, 3),
                "overlap_fraction": round(overlap / max(total, 1e-8), 4),
            }
        )
    mapped = [
        {
            "start": float(item["start"]),
            "end": float(item["end"]),
            "speaker": mapping.get(str(item["speaker"])),
            "sortformer_speaker": str(item["speaker"]),
            "model_speaker": item.get("model_speaker"),
        }
        for item in secondary
    ]
    return mapped, reports


def _annotate_sortformer_words(
    result: dict[str, Any], timeline: list[dict[str, Any]]
) -> tuple[int, int]:
    annotated = 0
    agreements = 0
    for segment in result.get("segments") or []:
        for word in segment.get("words") or []:
            if word.get("start") is None or word.get("end") is None:
                continue
            candidates: dict[str, float] = {}
            raw_labels: dict[str, float] = {}
            for item in timeline:
                overlap = _timeline_overlap(
                    {"start": word["start"], "end": word["end"]}, item
                )
                mapped = item.get("speaker")
                if overlap <= 0.0 or not mapped:
                    continue
                candidates[str(mapped)] = candidates.get(str(mapped), 0.0) + overlap
                raw = str(item.get("sortformer_speaker") or "")
                raw_labels[raw] = raw_labels.get(raw, 0.0) + overlap
            if not candidates:
                continue
            speaker = max(candidates, key=candidates.get)
            word["sortformer_speaker"] = speaker
            word["sortformer_model_speaker"] = max(raw_labels, key=raw_labels.get)
            annotated += 1
            if speaker == str(word.get("speaker")):
                agreements += 1
    return annotated, agreements


class PyannoteSortformerEnsemble:
    """Keep pyannote primary and add Sortformer as an independent second opinion."""

    name = "pyannote-community-1+nvidia-sortformer-v2.1"
    model_name = (
        "pyannote/speaker-diarization-community-1 + "
        f"{SORTFORMER_MODEL}"
    )

    def __init__(
        self,
        token: str,
        device: str,
        batch_size: int = 4,
        primary: Any = None,
        secondary: Any = None,
    ) -> None:
        self.primary = primary or PyannoteDiarizer(token, device, batch_size)
        self.secondary = secondary or SortformerDiarizer(device)
        self.device = device
        self.last_run: dict[str, Any] = {}

    def assign(
        self,
        source: Path,
        audio: Any,
        result: dict[str, Any],
        min_speakers: Optional[int],
        max_speakers: Optional[int],
    ) -> dict[str, Any]:
        assigned = self.primary.assign(
            source, audio, result, min_speakers, max_speakers
        )
        primary_timeline = list(
            assigned.get("speaker_assignment_timeline")
            or assigned.get("speaker_timeline")
            or []
        )
        try:
            secondary_timeline = self.secondary.diarize(audio)
            mapped, mappings = _map_secondary_timeline(
                primary_timeline, secondary_timeline
            )
            annotated, agreements = _annotate_sortformer_words(assigned, mapped)
            assigned["sortformer_timeline"] = mapped
            assigned["diarization_ensemble"] = {
                "enabled": True,
                "primary_model": getattr(self.primary, "model_name", None),
                "secondary_model": getattr(self.secondary, "model_name", None),
                "secondary_available": True,
                "speaker_mappings": mappings,
                "annotated_words": annotated,
                "agreement_words": agreements,
                "agreement_fraction": round(
                    agreements / max(annotated, 1), 4
                ),
            }
            secondary_error = None
        except Exception as exc:
            secondary_error = f"{type(exc).__name__}: {str(exc)}"[:1000]
            assigned["diarization_ensemble"] = {
                "enabled": True,
                "primary_model": getattr(self.primary, "model_name", None),
                "secondary_model": getattr(self.secondary, "model_name", None),
                "secondary_available": False,
                "secondary_error": secondary_error,
            }
        self.last_run = {
            "device": self.device,
            "primary": getattr(self.primary, "last_run", {}),
            "secondary": getattr(self.secondary, "last_run", {}),
            "secondary_error": secondary_error,
        }
        return assigned


def _cosine(left: Any, right: Any) -> float:
    import numpy as np

    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def _cluster_embeddings(
    embeddings: list[Any],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> list[int]:
    """Globally cluster normalized voice embeddings.

    Complete-linkage agglomerative clustering is deliberately used instead of
    an arrival-order threshold: the latter makes early mistakes permanent and
    tends to invent a therapist speaker for short participant replies.  When a
    speaker count is not supplied, silhouette quality selects a conservative
    count and permits a single-speaker result when separation is weak.
    """
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    if not embeddings:
        return []
    matrix = np.asarray(embeddings, dtype=np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    exact = min_speakers if min_speakers == max_speakers else None

    if exact is not None:
        target = max(1, min(exact, len(matrix)))
        if target == 1:
            return [0] * len(matrix)
        return (
            AgglomerativeClustering(
                n_clusters=target,
                metric="cosine",
                linkage="complete",
            )
            .fit_predict(matrix)
            .astype(int)
            .tolist()
        )

    minimum = max(1, min(min_speakers or 1, len(matrix)))
    maximum = max(minimum, min(max_speakers or 8, len(matrix)))
    if maximum == 1:
        return [0] * len(matrix)
    if minimum >= len(matrix):
        return list(range(len(matrix)))
    if len(matrix) < 3:
        return [0] * len(matrix)

    best_score = float("-inf")
    best_assignments: Optional[list[int]] = None
    for count in range(max(2, minimum), maximum + 1):
        if count >= len(matrix):
            break
        labels = AgglomerativeClustering(
            n_clusters=count,
            metric="cosine",
            linkage="complete",
        ).fit_predict(matrix)
        populations = np.bincount(labels)
        # Tiny clusters are commonly a noisy short utterance rather than a
        # distinct voice. Penalize them without making a known count impossible.
        singleton_penalty = 0.07 * int(np.count_nonzero(populations < 2))
        complexity_penalty = 0.004 * max(0, count - 2)
        score = float(silhouette_score(matrix, labels, metric="cosine"))
        score -= singleton_penalty + complexity_penalty
        if score > best_score:
            best_score = score
            best_assignments = labels.astype(int).tolist()

    if best_assignments is None:
        return [0] * len(matrix)
    if minimum == 1 and best_score < 0.25:
        return [0] * len(matrix)
    return best_assignments


def _word_chunks(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Create short acoustic windows while retaining references to ASR words."""
    chunks: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(result.get("segments") or []):
        timed_words = [
            word
            for word in segment.get("words") or []
            if word.get("start") is not None and word.get("end") is not None
        ]
        if not timed_words:
            start = float(segment.get("start") or 0.0)
            end = max(start, float(segment.get("end") or start))
            chunks.append(
                {
                    "segment_index": segment_index,
                    "words": [],
                    "start": start,
                    "end": end,
                }
            )
            continue

        group: list[dict[str, Any]] = []
        for word in timed_words:
            if group:
                group_duration = float(group[-1]["end"]) - float(group[0]["start"])
                gap = float(word["start"]) - float(group[-1]["end"])
                if (gap >= 0.32 and group_duration >= 0.55) or group_duration >= 1.25:
                    chunks.append(
                        {
                            "segment_index": segment_index,
                            "words": group,
                            "start": float(group[0]["start"]),
                            "end": float(group[-1]["end"]),
                        }
                    )
                    group = []
            group.append(word)
        if group:
            chunks.append(
                {
                    "segment_index": segment_index,
                    "words": group,
                    "start": float(group[0]["start"]),
                    "end": float(group[-1]["end"]),
                }
            )
    return chunks


class SpeechBrainEmbeddingDiarizer:
    """Windowed global voice clustering when gated pyannote is unavailable."""

    name = "speechbrain/ecapa-windowed-global-clustering"

    def __init__(self, device: str) -> None:
        self.model_name = SPEAKER_EMBEDDING_MODEL
        self.device = device
        self.last_run: dict[str, Any] = {}
        self.classifier = _load_speaker_encoder(device)

    def assign(
        self,
        source: Path,
        audio: Any,
        result: dict[str, Any],
        min_speakers: Optional[int],
        max_speakers: Optional[int],
    ) -> dict[str, Any]:
        del source
        import numpy as np
        import torch

        segments = result.get("segments") or []
        chunks = _word_chunks(result)
        embeddings: list[Any] = []
        embedded_indexes: list[int] = []
        durations: list[float] = []
        for index, chunk in enumerate(chunks):
            start = max(0, int(float(chunk["start"]) * SAMPLE_RATE))
            end = min(
                len(audio),
                max(start + 1, int(float(chunk["end"]) * SAMPLE_RATE)),
            )
            samples = np.array(audio[start:end], dtype=np.float32, copy=True)
            duration = len(samples) / SAMPLE_RATE
            if duration < 0.12:
                continue
            minimum_samples = int(0.8 * SAMPLE_RATE)
            if len(samples) < minimum_samples:
                samples = np.pad(samples, (0, minimum_samples - len(samples)))
            waveform = torch.from_numpy(samples).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                embedding = (
                    self.classifier.encode_batch(waveform).detach().cpu().numpy()
                )
            embeddings.append(embedding.reshape(-1))
            embedded_indexes.append(index)
            durations.append(duration)

        if not embeddings:
            raise RuntimeError(
                "no speech windows were long enough for acoustic clustering"
            )

        matrix = np.asarray(embeddings, dtype=np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
        anchor_indexes = [
            index for index, duration in enumerate(durations) if duration >= 0.8
        ]
        minimum_anchors = max(2, min_speakers or 1)
        if len(anchor_indexes) < minimum_anchors:
            anchor_indexes = list(range(len(matrix)))
        anchor_embeddings = [matrix[index] for index in anchor_indexes]
        anchor_assignments = _cluster_embeddings(
            anchor_embeddings, min_speakers, max_speakers
        )
        cluster_ids = sorted(set(anchor_assignments))
        centroids = []
        for cluster in cluster_ids:
            members = [
                matrix[anchor_indexes[index]]
                for index, assignment in enumerate(anchor_assignments)
                if assignment == cluster
            ]
            centroid = np.mean(members, axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
            centroids.append(centroid)
        assignments = [
            int(np.argmax([_cosine(embedding, centroid) for centroid in centroids]))
            for embedding in matrix
        ]

        # A tiny trailing fragment is usually a chunk boundary, not a new
        # speaker. Smooth only sub-350 ms fragments without a real pause.
        for index in range(1, len(assignments)):
            current_chunk = chunks[embedded_indexes[index]]
            previous_chunk = chunks[embedded_indexes[index - 1]]
            if (
                current_chunk["segment_index"] == previous_chunk["segment_index"]
                and durations[index] < 0.35
                and float(current_chunk["start"]) - float(previous_chunk["end"]) < 0.25
            ):
                assignments[index] = assignments[index - 1]

        first_seen: dict[int, int] = {}
        for assignment in assignments:
            first_seen.setdefault(assignment, len(first_seen))
        assigned_by_chunk: dict[int, str] = {}
        for chunk_index, assignment in zip(embedded_indexes, assignments, strict=True):
            assigned_by_chunk[chunk_index] = f"SPEAKER_{first_seen[assignment]:02d}"

        if len(assigned_by_chunk) != len(chunks):
            for index, chunk in enumerate(chunks):
                if index in assigned_by_chunk:
                    continue
                nearest = min(
                    assigned_by_chunk,
                    key=lambda item: abs(
                        float(chunks[item]["start"]) - float(chunk["start"])
                    ),
                )
                assigned_by_chunk[index] = assigned_by_chunk[nearest]

        # Apply conservative boundary smoothing after even the too-short-to-
        # embed chunks have labels. A sub-850 ms minority within one ASR
        # segment is not enough acoustic evidence to claim a speaker switch.
        for index in range(1, len(chunks)):
            current = chunks[index]
            previous = chunks[index - 1]
            if (
                current["segment_index"] == previous["segment_index"]
                and float(current["end"]) - float(current["start"]) < 0.35
                and float(current["start"]) - float(previous["end"]) < 0.25
            ):
                assigned_by_chunk[index] = assigned_by_chunk[index - 1]
        by_segment: dict[int, list[int]] = {}
        for index, chunk in enumerate(chunks):
            by_segment.setdefault(int(chunk["segment_index"]), []).append(index)
        for indexes in by_segment.values():
            totals: dict[str, float] = {}
            for index in indexes:
                label = assigned_by_chunk[index]
                duration = float(chunks[index]["end"]) - float(chunks[index]["start"])
                totals[label] = totals.get(label, 0.0) + max(0.0, duration)
            if len(totals) < 2:
                continue
            majority = max(totals, key=totals.get)
            weak = {label for label, duration in totals.items() if duration < 0.85}
            for index in indexes:
                if assigned_by_chunk[index] in weak:
                    assigned_by_chunk[index] = majority

        speaker_durations: dict[int, dict[str, float]] = {}
        timeline: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            speaker = assigned_by_chunk[index]
            segment_index = int(chunk["segment_index"])
            for word in chunk["words"]:
                word["speaker"] = speaker
            duration = max(0.0, float(chunk["end"]) - float(chunk["start"]))
            per_segment = speaker_durations.setdefault(segment_index, {})
            per_segment[speaker] = per_segment.get(speaker, 0.0) + duration
            interval = {
                "start": float(chunk["start"]),
                "end": float(chunk["end"]),
                "speaker": speaker,
            }
            if (
                timeline
                and timeline[-1]["speaker"] == speaker
                and interval["start"] - timeline[-1]["end"] <= 0.12
            ):
                timeline[-1]["end"] = interval["end"]
            else:
                timeline.append(interval)

        for index, segment in enumerate(segments):
            choices = speaker_durations.get(index) or {}
            if choices:
                segment["speaker"] = max(choices, key=choices.get)
            elif timeline:
                start = float(segment.get("start") or 0.0)
                nearest = min(timeline, key=lambda item: abs(item["start"] - start))
                segment["speaker"] = nearest["speaker"]
        result["speaker_timeline"] = timeline
        self.last_run = {
            "device": self.device,
            "detected_speakers": len(set(assigned_by_chunk.values())),
            "timeline_intervals": len(timeline),
        }
        return result


def _subtract_intervals(
    interval: tuple[float, float], blockers: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    parts = [interval]
    for blocker_start, blocker_end in blockers:
        remaining = []
        for start, end in parts:
            if blocker_end <= start or blocker_start >= end:
                remaining.append((start, end))
                continue
            if start < blocker_start:
                remaining.append((start, blocker_start))
            if blocker_end < end:
                remaining.append((blocker_end, end))
        parts = remaining
        if not parts:
            break
    return parts


def _exclusive_intervals(
    timeline: list[dict[str, Any]], speaker: str
) -> list[tuple[float, float]]:
    blockers = sorted(
        (
            (float(item.get("start") or 0.0), float(item.get("end") or 0.0))
            for item in timeline
            if str(item.get("speaker")) != speaker
        ),
        key=lambda item: item[0],
    )
    intervals: list[tuple[float, float]] = []
    for item in timeline:
        if str(item.get("speaker")) != speaker:
            continue
        start = max(0.0, float(item.get("start") or 0.0))
        end = max(start, float(item.get("end") or start))
        for clean_start, clean_end in _subtract_intervals((start, end), blockers):
            # Avoid speaker-change boundary leakage in the identity embedding.
            clean_start += 0.08
            clean_end -= 0.08
            if clean_end - clean_start >= 0.5:
                intervals.append((clean_start, clean_end))

    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start - merged[-1][1] <= 0.08:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _merge_intervals(
    intervals: list[tuple[float, float]], *, maximum_gap: float = 0.0
) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start - merged[-1][1] <= maximum_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _intersect_intervals(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    intersections: list[tuple[float, float]] = []
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if end > start:
            intersections.append((start, end))
        if left[left_index][1] <= right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return _merge_intervals(intersections, maximum_gap=0.08)


def _recognized_speech_intervals(
    result: dict[str, Any], speaker: str
) -> tuple[list[tuple[float, float]], int, bool]:
    intervals: list[tuple[float, float]] = []
    word_count = 0
    timing_available = False
    for segment in result.get("segments") or []:
        segment_speaker = str(segment.get("speaker") or "SPEAKER_UNKNOWN")
        for word in segment.get("words") or []:
            if word.get("start") is None or word.get("end") is None:
                continue
            timing_available = True
            word_speaker = str(word.get("speaker") or segment_speaker)
            text = str(word.get("word") or word.get("text") or "").strip()
            bracketed_annotation = len(text) >= 2 and (text[0], text[-1]) in {
                ("[", "]"),
                ("(", ")"),
                ("<", ">"),
            }
            if (
                word_speaker != speaker
                or bracketed_annotation
                or not any(character.isalnum() for character in text)
            ):
                continue
            start = max(0.0, float(word["start"]) - 0.06)
            end = max(start, float(word["end"]) + 0.06)
            intervals.append((start, end))
            word_count += 1
    return _merge_intervals(intervals, maximum_gap=0.35), word_count, timing_available


def _speaker_word_runs(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return mutable, time-ordered runs of lexical words with one speaker label."""
    runs: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(result.get("segments") or []):
        fallback = str(segment.get("speaker") or "SPEAKER_UNKNOWN")
        for word in segment.get("words") or []:
            if word.get("start") is None or word.get("end") is None:
                continue
            text = str(word.get("word") or word.get("text") or "").strip()
            if not text or not any(character.isalnum() for character in text):
                continue
            speaker = str(word.get("speaker") or fallback)
            if speaker in {"", "SPEAKER_UNKNOWN", "None"}:
                continue
            start = max(0.0, float(word["start"]))
            end = max(start, float(word["end"]))
            if end <= start:
                continue
            if (
                runs
                and runs[-1]["speaker"] == speaker
                and start - float(runs[-1]["end"]) <= 0.45
            ):
                runs[-1]["end"] = max(float(runs[-1]["end"]), end)
                runs[-1]["words"].append(word)
                runs[-1]["segment_indexes"].add(segment_index)
            else:
                runs.append(
                    {
                        "start": start,
                        "end": end,
                        "speaker": speaker,
                        "words": [word],
                        "segment_indexes": {segment_index},
                    }
                )
    return sorted(runs, key=lambda item: (float(item["start"]), float(item["end"])))


def _sentence_continues(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Use aligned English punctuation/casing only as a conservative weak gate."""
    if not left.get("words") or not right.get("words"):
        return False
    left_word = left["words"][-1]
    right_word = right["words"][0]
    left_text = str(left_word.get("word") or left_word.get("text") or "").strip()
    right_text = str(right_word.get("word") or right_word.get("text") or "").strip()
    if not left_text or not right_text:
        return False
    left_core = left_text.rstrip("\"')]} ")
    if not left_core or left_core[-1] in ".!?":
        return False
    first_letter = next((character for character in right_text if character.isalpha()), "")
    return bool(first_letter and first_letter.islower())


def _sentence_ends(word: dict[str, Any]) -> bool:
    text = str(word.get("word") or word.get("text") or "").strip()
    core = text.rstrip("\"')]} ")
    return bool(core and core[-1] in ".!?")


def _lexical_utterances(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Group aligned English words without using their diarization labels."""
    entries: list[tuple[int, dict[str, Any]]] = []
    for segment_index, segment in enumerate(result.get("segments") or []):
        for word in segment.get("words") or []:
            if word.get("start") is None or word.get("end") is None:
                continue
            text = str(word.get("word") or word.get("text") or "").strip()
            if not text or not any(character.isalnum() for character in text):
                continue
            entries.append((segment_index, word))
    entries.sort(key=lambda item: (float(item[1]["start"]), float(item[1]["end"])))

    utterances: list[dict[str, Any]] = []
    group: list[tuple[int, dict[str, Any]]] = []

    def flush() -> None:
        if not group:
            return
        utterances.append(
            {
                "start": float(group[0][1]["start"]),
                "end": float(group[-1][1]["end"]),
                "words": [word for _, word in group],
                "segment_indexes": {index for index, _ in group},
            }
        )

    for segment_index, word in entries:
        if group:
            previous = group[-1][1]
            gap = max(0.0, float(word["start"]) - float(previous["end"]))
            duration = float(previous["end"]) - float(group[0][1]["start"])
            if (
                gap > SPEAKER_REFINEMENT_UTTERANCE_GAP_SECONDS
                or _sentence_ends(previous)
                or duration >= SPEAKER_REFINEMENT_UTTERANCE_MAX_SECONDS
            ):
                flush()
                group = []
        group.append((segment_index, word))
    flush()
    return utterances


def _overlap_intervals(timeline: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Collect regions where the raw diarizer reports simultaneous speakers."""
    entries = sorted(
        (
            max(0.0, float(item.get("start") or 0.0)),
            max(0.0, float(item.get("end") or 0.0)),
            str(item.get("speaker") or ""),
        )
        for item in timeline
        if item.get("speaker") not in (None, "", "SPEAKER_UNKNOWN")
    )
    active: list[tuple[float, float, str]] = []
    overlaps: list[tuple[float, float]] = []
    for start, end, speaker in entries:
        if end <= start:
            continue
        active = [item for item in active if item[1] > start]
        for other_start, other_end, other_speaker in active:
            if other_speaker == speaker:
                continue
            overlap_start = max(start, other_start)
            overlap_end = min(end, other_end)
            if overlap_end > overlap_start:
                overlaps.append((overlap_start, overlap_end))
        active.append((start, end, speaker))
    return _merge_intervals(overlaps, maximum_gap=0.02)


def _interval_overlap_seconds(
    start: float, end: float, intervals: list[tuple[float, float]]
) -> float:
    total = 0.0
    for interval_start, interval_end in intervals:
        if interval_start >= end:
            break
        if interval_end <= start:
            continue
        total += max(0.0, min(end, interval_end) - max(start, interval_start))
    return total


def _relabel_timeline_interval(
    timeline: list[dict[str, Any]],
    *,
    start: float,
    end: float,
    original_speaker: str,
    refined_speaker: str,
) -> list[dict[str, Any]]:
    """Relabel one bounded region of an exclusive assignment timeline."""
    pieces: list[dict[str, Any]] = []
    for item in timeline:
        item_start = float(item.get("start") or 0.0)
        item_end = max(item_start, float(item.get("end") or item_start))
        speaker = str(item.get("speaker") or "SPEAKER_UNKNOWN")
        overlap_start = max(start, item_start)
        overlap_end = min(end, item_end)
        if speaker != original_speaker or overlap_end <= overlap_start:
            pieces.append({"start": item_start, "end": item_end, "speaker": speaker})
            continue
        if item_start < overlap_start:
            pieces.append(
                {"start": item_start, "end": overlap_start, "speaker": speaker}
            )
        pieces.append(
            {
                "start": overlap_start,
                "end": overlap_end,
                "speaker": refined_speaker,
            }
        )
        if overlap_end < item_end:
            pieces.append({"start": overlap_end, "end": item_end, "speaker": speaker})

    merged: list[dict[str, Any]] = []
    for item in sorted(pieces, key=lambda value: (value["start"], value["end"])):
        if item["end"] <= item["start"]:
            continue
        if (
            merged
            and merged[-1]["speaker"] == item["speaker"]
            and item["start"] - merged[-1]["end"] <= 0.02
        ):
            merged[-1]["end"] = max(merged[-1]["end"], item["end"])
        else:
            merged.append(item)
    return merged


class SpeakerIdentityEncoder:
    """Extract bounded, overlap-free ECAPA evidence for each local speaker."""

    name = SPEAKER_EMBEDDING_MODEL
    version = SPEAKER_EMBEDDING_REVISION

    def __init__(self, device: str, classifier: Any = None) -> None:
        self.device = device
        self.classifier = classifier or _load_speaker_encoder(device)

    def _encode_samples(self, samples: Any) -> Any:
        import numpy as np
        import torch

        values = np.asarray(samples, dtype=np.float32)
        if len(values) < int(0.8 * SAMPLE_RATE):
            values = np.pad(values, (0, int(0.8 * SAMPLE_RATE) - len(values)))
        rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
        if not math.isfinite(rms) or rms < 1e-4:
            return None
        waveform = torch.from_numpy(np.array(values, copy=True)).unsqueeze(0)
        waveform = waveform.to(self.device)
        with torch.inference_mode():
            embedding = (
                self.classifier.encode_batch(waveform)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
        norm = float(np.linalg.norm(embedding))
        return embedding / norm if math.isfinite(norm) and norm > 1e-8 else None

    def _encode_interval(self, audio: Any, start: float, end: float) -> Any:
        import numpy as np

        first = max(0, int(start * SAMPLE_RATE))
        last = min(len(audio), max(first + 1, int(end * SAMPLE_RATE)))
        if last - first < int(0.35 * SAMPLE_RATE):
            return None
        return self._encode_samples(
            np.asarray(audio[first:last], dtype=np.float32)
        )

    def refine(
        self,
        audio: Any,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Conservatively correct diarization flips using acoustic context.

        The raw overlap-aware timeline is never changed. Only recognized-word
        assignments and the exclusive assignment timeline are refined, and only
        when a clean alternate prototype wins by a strong cosine margin while a
        neighboring run supports the same speaker.
        """
        timeline = list(result.get("speaker_timeline") or [])
        runs = _speaker_word_runs(result)
        overlap_regions = _overlap_intervals(timeline)
        language = str(result.get("language") or "").lower()
        english_alignment = language in {"en", "eng", "english"}
        evidence = self.extract(audio, timeline, result)
        prototypes = {
            speaker: item
            for speaker, item in evidence.items()
            if float(item.get("clean_seconds") or 0.0)
            >= SPEAKER_REFINEMENT_MIN_PROTOTYPE_SECONDS
            and int(item.get("window_count") or 0) >= 2
            and float(item.get("cohesion") or 0.0) >= 0.40
            and item.get("embedding")
        }
        corrections: list[dict[str, Any]] = []
        protected_intervals: list[tuple[float, float]] = []
        evaluated_candidates: list[dict[str, Any]] = []
        candidates_examined = 0

        if english_alignment:
            for utterance in _lexical_utterances(result):
                utterance_start = float(utterance["start"])
                utterance_end = float(utterance["end"])
                utterance_duration = utterance_end - utterance_start
                if (
                    utterance_duration < 0.35
                    or utterance_duration
                    > SPEAKER_REFINEMENT_UTTERANCE_MAX_SECONDS
                    or _interval_overlap_seconds(
                        utterance_start, utterance_end, overlap_regions
                    )
                    > min(0.12, utterance_duration * 0.15)
                ):
                    continue

                label_runs: list[dict[str, Any]] = []
                for word in utterance["words"]:
                    speaker = str(word.get("speaker") or "SPEAKER_UNKNOWN")
                    if speaker not in prototypes:
                        label_runs = []
                        break
                    if label_runs and label_runs[-1]["speaker"] == speaker:
                        label_runs[-1]["words"].append(word)
                        label_runs[-1]["end"] = float(word["end"])
                    else:
                        label_runs.append(
                            {
                                "speaker": speaker,
                                "words": [word],
                                "start": float(word["start"]),
                                "end": float(word["end"]),
                            }
                        )
                if (
                    len(label_runs) != 3
                    or label_runs[0]["speaker"] != label_runs[2]["speaker"]
                    or label_runs[0]["speaker"] == label_runs[1]["speaker"]
                ):
                    continue

                target = str(label_runs[0]["speaker"])
                original = str(label_runs[1]["speaker"])
                middle = label_runs[1]
                embedding = self._encode_interval(
                    audio, utterance_start, utterance_end
                )
                if embedding is None:
                    continue
                candidates_examined += 1
                scores = {
                    speaker: cosine_similarity(embedding, item["embedding"])
                    for speaker, item in prototypes.items()
                }
                best = max(scores, key=scores.get)
                target_score = scores[target]
                original_score = scores[original]
                runner_up = max(
                    (score for speaker, score in scores.items() if speaker != target),
                    default=-1.0,
                )
                margin = target_score - runner_up

                secondary_durations: dict[str, float] = {}
                for word in utterance["words"]:
                    secondary = word.get("sortformer_speaker")
                    if not secondary:
                        continue
                    word_duration = max(
                        0.0, float(word["end"]) - float(word["start"])
                    )
                    secondary_durations[str(secondary)] = (
                        secondary_durations.get(str(secondary), 0.0)
                        + word_duration
                    )
                secondary_consensus = None
                secondary_fraction = 0.0
                if secondary_durations:
                    secondary_total = sum(secondary_durations.values())
                    secondary_consensus = max(
                        secondary_durations, key=secondary_durations.get
                    )
                    secondary_fraction = (
                        secondary_durations[secondary_consensus]
                        / max(secondary_total, 1e-8)
                    )
                    if secondary_fraction < 0.65:
                        secondary_consensus = None

                required_similarity = SPEAKER_REFINEMENT_UTTERANCE_MIN_SIMILARITY
                required_margin = SPEAKER_REFINEMENT_UTTERANCE_MIN_MARGIN
                if secondary_consensus == target:
                    required_similarity = 0.45
                    required_margin = 0.10
                audit = {
                    "strategy": "bracketed_utterance",
                    "start": round(utterance_start, 3),
                    "end": round(utterance_end, 3),
                    "duration_seconds": round(utterance_duration, 3),
                    "original_local_speaker": original,
                    "best_local_speaker": best,
                    "bracketing_local_speaker": target,
                    "original_similarity": round(original_score, 4),
                    "best_similarity": round(scores[best], 4),
                    "target_similarity": round(target_score, 4),
                    "similarity_margin": round(margin, 4),
                    "required_similarity": round(required_similarity, 4),
                    "required_margin": round(required_margin, 4),
                    "sortformer_consensus": secondary_consensus,
                    "sortformer_consensus_fraction": round(secondary_fraction, 4),
                }
                if secondary_consensus not in (None, target):
                    audit["decision"] = "abstained_sortformer_disagreement"
                    evaluated_candidates.append(audit)
                    protected_intervals.append(
                        (float(middle["start"]), float(middle["end"]))
                    )
                    continue
                if (
                    best != target
                    or target_score < required_similarity
                    or margin < required_margin
                ):
                    audit["decision"] = "abstained_below_confidence_threshold"
                    evaluated_candidates.append(audit)
                    continue

                audit["decision"] = "corrected"
                evaluated_candidates.append(audit)
                corrections.append(
                    {
                        "strategy": "bracketed_utterance",
                        "start": round(float(middle["start"]), 3),
                        "end": round(float(middle["end"]), 3),
                        "duration_seconds": round(
                            float(middle["end"]) - float(middle["start"]), 3
                        ),
                        "evidence_start": round(utterance_start, 3),
                        "evidence_end": round(utterance_end, 3),
                        "from_local_speaker": original,
                        "to_local_speaker": target,
                        "original_similarity": round(original_score, 4),
                        "refined_similarity": round(target_score, 4),
                        "similarity_margin": round(margin, 4),
                        "context_support": 2,
                        "sandwiched_sentence_continuation": True,
                        "sentence_seam_continuation": True,
                        "sentence_seam_support": 2,
                        "sortformer_consensus": secondary_consensus,
                        "sortformer_consensus_fraction": round(
                            secondary_fraction, 4
                        ),
                        "word_count": len(middle["words"]),
                        "run": {
                            "words": middle["words"],
                            "segment_indexes": utterance["segment_indexes"],
                        },
                    }
                )

        accepted_intervals = [
            (float(item["start"]), float(item["end"])) for item in corrections
        ] + protected_intervals
        for index, run in enumerate(runs):
            start = float(run["start"])
            end = float(run["end"])
            duration = end - start
            original = str(run["speaker"])
            if duration < 0.35 or original not in prototypes or len(prototypes) < 2:
                continue
            if _interval_overlap_seconds(start, end, accepted_intervals) > 0.0:
                continue
            overlap_seconds = _interval_overlap_seconds(start, end, overlap_regions)
            if overlap_seconds > min(0.12, duration * 0.15):
                continue

            neighbors: list[str] = []
            sentence_seam_speakers: list[str] = []
            previous = None
            following = None
            if index > 0:
                previous = runs[index - 1]
                gap = max(0.0, start - float(previous["end"]))
                if gap <= SPEAKER_REFINEMENT_CONTEXT_GAP_SECONDS:
                    neighbors.append(str(previous["speaker"]))
                if (
                    english_alignment
                    and gap <= SPEAKER_REFINEMENT_SENTENCE_SEAM_GAP_SECONDS
                    and _sentence_continues(previous, run)
                ):
                    sentence_seam_speakers.append(str(previous["speaker"]))
            if index + 1 < len(runs):
                following = runs[index + 1]
                gap = max(0.0, float(following["start"]) - end)
                if gap <= SPEAKER_REFINEMENT_CONTEXT_GAP_SECONDS:
                    neighbors.append(str(following["speaker"]))
                if (
                    english_alignment
                    and gap <= SPEAKER_REFINEMENT_SENTENCE_SEAM_GAP_SECONDS
                    and _sentence_continues(run, following)
                ):
                    sentence_seam_speakers.append(str(following["speaker"]))
            sentence_seam_alternatives = {
                speaker
                for speaker in sentence_seam_speakers
                if speaker != original and speaker in prototypes
            }
            short_candidate = duration <= SPEAKER_REFINEMENT_MAX_RUN_SECONDS
            sentence_seam_candidate = bool(
                duration <= SPEAKER_REFINEMENT_SENTENCE_SEAM_MAX_RUN_SECONDS
                and sentence_seam_alternatives
            )
            if not short_candidate and not sentence_seam_candidate:
                continue
            supported_alternatives = {
                speaker
                for speaker in (
                    neighbors if short_candidate else sentence_seam_speakers
                )
                if speaker != original and speaker in prototypes
            }
            if not supported_alternatives:
                continue

            embedding = self._encode_interval(audio, start, end)
            if embedding is None:
                continue
            candidates_examined += 1
            scores = {
                speaker: cosine_similarity(embedding, item["embedding"])
                for speaker, item in prototypes.items()
            }
            refined = max(scores, key=scores.get)
            original_score = scores.get(original, -1.0)
            refined_score = scores.get(refined, -1.0)
            audit = {
                "strategy": "word_run",
                "start": round(start, 3),
                "end": round(end, 3),
                "duration_seconds": round(duration, 3),
                "original_local_speaker": original,
                "best_local_speaker": refined,
                "original_similarity": round(original_score, 4),
                "best_similarity": round(refined_score, 4),
                "context_speakers": neighbors,
                "sentence_seam_speakers": sentence_seam_speakers,
            }
            if refined == original or refined not in supported_alternatives:
                audit["decision"] = (
                    "kept_original"
                    if refined == original
                    else "abstained_best_voice_not_supported_by_context"
                )
                evaluated_candidates.append(audit)
                continue
            margin = refined_score - original_score
            context_support = sum(speaker == refined for speaker in neighbors)
            sentence_seam_support = sum(
                speaker == refined for speaker in sentence_seam_speakers
            )
            sentence_seam_continuation = sentence_seam_support > 0
            sandwiched_sentence_continuation = bool(
                english_alignment
                and previous is not None
                and following is not None
                and str(previous["speaker"]) == refined
                and str(following["speaker"]) == refined
                and _sentence_continues(previous, run)
                and _sentence_continues(run, following)
            )
            required_margin = (
                SPEAKER_REFINEMENT_DOUBLE_CONTEXT_MARGIN
                if context_support >= 2
                else SPEAKER_REFINEMENT_SINGLE_CONTEXT_MARGIN
            )
            if duration > 1.4:
                required_margin += 0.04
            required_similarity = SPEAKER_REFINEMENT_MIN_SIMILARITY
            if duration < 0.60:
                required_similarity += 0.03
            if (
                duration <= SPEAKER_REFINEMENT_MICRO_RUN_SECONDS
                and sentence_seam_continuation
            ):
                required_similarity = SPEAKER_REFINEMENT_MICRO_MIN_SIMILARITY
                required_margin = SPEAKER_REFINEMENT_MICRO_MIN_MARGIN
            if refined_score < required_similarity or margin < required_margin:
                audit.update(
                    {
                        "decision": "abstained_below_confidence_threshold",
                        "similarity_margin": round(margin, 4),
                        "required_similarity": round(required_similarity, 4),
                        "required_margin": round(required_margin, 4),
                        "sandwiched_sentence_continuation": (
                            sandwiched_sentence_continuation
                        ),
                        "sentence_seam_continuation": (
                            sentence_seam_continuation
                        ),
                        "sentence_seam_support": sentence_seam_support,
                    }
                )
                evaluated_candidates.append(audit)
                continue
            audit.update(
                {
                    "decision": "corrected",
                    "similarity_margin": round(margin, 4),
                    "required_similarity": round(required_similarity, 4),
                    "required_margin": round(required_margin, 4),
                    "sandwiched_sentence_continuation": (
                        sandwiched_sentence_continuation
                    ),
                    "sentence_seam_continuation": sentence_seam_continuation,
                    "sentence_seam_support": sentence_seam_support,
                }
            )
            evaluated_candidates.append(audit)
            corrections.append(
                {
                    "strategy": "word_run",
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration_seconds": round(duration, 3),
                    "from_local_speaker": original,
                    "to_local_speaker": refined,
                    "original_similarity": round(original_score, 4),
                    "refined_similarity": round(refined_score, 4),
                    "similarity_margin": round(margin, 4),
                    "context_support": context_support,
                    "sandwiched_sentence_continuation": (
                        sandwiched_sentence_continuation
                    ),
                    "sentence_seam_continuation": sentence_seam_continuation,
                    "sentence_seam_support": sentence_seam_support,
                    "word_count": len(run["words"]),
                    "run": run,
                }
            )

        assignment_timeline = [
            {
                "start": float(item.get("start") or 0.0),
                "end": float(item.get("end") or 0.0),
                "speaker": str(item.get("speaker") or "SPEAKER_UNKNOWN"),
            }
            for item in (
                result.get("speaker_assignment_timeline")
                or result.get("speaker_timeline")
                or []
            )
        ]
        if corrections:
            result["speaker_assignment_timeline_original"] = [
                dict(item) for item in assignment_timeline
            ]
        for correction in corrections:
            run = correction.pop("run")
            for word in run["words"]:
                word["diarization_speaker"] = correction["from_local_speaker"]
                word["speaker"] = correction["to_local_speaker"]
                word["speaker_refinement"] = {
                    "algorithm_version": SPEAKER_REFINEMENT_VERSION,
                    "from_local_speaker": correction["from_local_speaker"],
                    "to_local_speaker": correction["to_local_speaker"],
                    "original_similarity": correction["original_similarity"],
                    "refined_similarity": correction["refined_similarity"],
                    "similarity_margin": correction["similarity_margin"],
                    "context_support": correction["context_support"],
                    "strategy": correction["strategy"],
                    "sandwiched_sentence_continuation": correction[
                        "sandwiched_sentence_continuation"
                    ],
                    "sentence_seam_continuation": correction[
                        "sentence_seam_continuation"
                    ],
                    "sentence_seam_support": correction[
                        "sentence_seam_support"
                    ],
                }
            assignment_timeline = _relabel_timeline_interval(
                assignment_timeline,
                start=float(correction["start"]),
                end=float(correction["end"]),
                original_speaker=str(correction["from_local_speaker"]),
                refined_speaker=str(correction["to_local_speaker"]),
            )

        for segment in result.get("segments") or []:
            durations: dict[str, float] = {}
            refined_words = []
            for word in segment.get("words") or []:
                if word.get("start") is None or word.get("end") is None:
                    continue
                speaker = str(word.get("speaker") or segment.get("speaker") or "")
                duration = max(0.0, float(word["end"]) - float(word["start"]))
                durations[speaker] = durations.get(speaker, 0.0) + duration
                if word.get("speaker_refinement"):
                    refined_words.append(word)
            if not durations:
                continue
            if not refined_words:
                continue
            original = str(segment.get("speaker") or "SPEAKER_UNKNOWN")
            refined = max(durations, key=durations.get)
            if original != refined:
                segment["diarization_speaker"] = original
                segment["speaker"] = refined
            segment["speaker_refinement"] = {
                "algorithm_version": SPEAKER_REFINEMENT_VERSION,
                "corrected_word_count": len(refined_words),
                "segment_speaker_changed": original != refined,
            }

        if corrections:
            result["speaker_assignment_timeline"] = assignment_timeline
        report = {
            "enabled": True,
            "algorithm": "ecapa_context_gated_word_run_resegmentation",
            "algorithm_version": SPEAKER_REFINEMENT_VERSION,
            "embedding_model": self.name,
            "embedding_model_revision": self.version,
            "prototype_count": len(prototypes),
            "word_runs": len(runs),
            "candidates_examined": candidates_examined,
            "corrections_applied": len(corrections),
            "evaluated_candidates": evaluated_candidates,
            "thresholds": {
                "maximum_run_seconds": SPEAKER_REFINEMENT_MAX_RUN_SECONDS,
                "maximum_context_gap_seconds": (
                    SPEAKER_REFINEMENT_CONTEXT_GAP_SECONDS
                ),
                "sentence_seam_gap_seconds": (
                    SPEAKER_REFINEMENT_SENTENCE_SEAM_GAP_SECONDS
                ),
                "sentence_seam_maximum_run_seconds": (
                    SPEAKER_REFINEMENT_SENTENCE_SEAM_MAX_RUN_SECONDS
                ),
                "minimum_prototype_seconds": (
                    SPEAKER_REFINEMENT_MIN_PROTOTYPE_SECONDS
                ),
                "minimum_similarity": SPEAKER_REFINEMENT_MIN_SIMILARITY,
                "single_context_margin": (
                    SPEAKER_REFINEMENT_SINGLE_CONTEXT_MARGIN
                ),
                "double_context_margin": (
                    SPEAKER_REFINEMENT_DOUBLE_CONTEXT_MARGIN
                ),
                "micro_run_seconds": SPEAKER_REFINEMENT_MICRO_RUN_SECONDS,
                "micro_minimum_similarity": (
                    SPEAKER_REFINEMENT_MICRO_MIN_SIMILARITY
                ),
                "micro_minimum_margin": SPEAKER_REFINEMENT_MICRO_MIN_MARGIN,
                "utterance_gap_seconds": (
                    SPEAKER_REFINEMENT_UTTERANCE_GAP_SECONDS
                ),
                "utterance_maximum_seconds": (
                    SPEAKER_REFINEMENT_UTTERANCE_MAX_SECONDS
                ),
                "utterance_minimum_similarity": (
                    SPEAKER_REFINEMENT_UTTERANCE_MIN_SIMILARITY
                ),
                "utterance_minimum_margin": (
                    SPEAKER_REFINEMENT_UTTERANCE_MIN_MARGIN
                ),
            },
            "corrections": corrections,
            "raw_diarization_preserved": True,
        }
        result["speaker_refinement"] = report
        return report

    @staticmethod
    def _virtual_slice(
        audio: Any,
        intervals: list[tuple[float, float]],
        offset_seconds: float,
        duration_seconds: float,
    ) -> Any:
        import numpy as np

        wanted_start = offset_seconds
        wanted_end = offset_seconds + duration_seconds
        cursor = 0.0
        pieces = []
        for start, end in intervals:
            interval_duration = end - start
            virtual_start = cursor
            virtual_end = cursor + interval_duration
            cursor = virtual_end
            overlap_start = max(wanted_start, virtual_start)
            overlap_end = min(wanted_end, virtual_end)
            if overlap_end <= overlap_start:
                continue
            source_start = start + (overlap_start - virtual_start)
            source_end = start + (overlap_end - virtual_start)
            first = max(0, int(source_start * SAMPLE_RATE))
            last = min(len(audio), max(first + 1, int(source_end * SAMPLE_RATE)))
            pieces.append(np.asarray(audio[first:last], dtype=np.float32))
            if virtual_end >= wanted_end:
                break
        return (
            np.concatenate(pieces).astype(np.float32, copy=False)
            if pieces
            else np.asarray([], dtype=np.float32)
        )

    def extract(
        self,
        audio: Any,
        timeline: list[dict[str, Any]],
        result: Optional[dict[str, Any]] = None,
    ) -> dict[str, dict[str, Any]]:
        import numpy as np

        speakers = sorted(
            {
                str(item.get("speaker"))
                for item in timeline
                if item.get("speaker") not in (None, "", "SPEAKER_UNKNOWN")
            }
        )
        evidence: dict[str, dict[str, Any]] = {}
        for speaker in speakers:
            intervals = _exclusive_intervals(timeline, speaker)
            recognized_intervals, recognized_word_count, word_timing_available = (
                _recognized_speech_intervals(result or {}, speaker)
            )
            recognized_speech_seconds = sum(
                end - start for start, end in recognized_intervals
            )
            lexical_intervals = _intersect_intervals(intervals, recognized_intervals)
            lexical_seconds = sum(end - start for start, end in lexical_intervals)
            lexical_filter_applied = word_timing_available and lexical_seconds >= 1.0
            if lexical_filter_applied:
                intervals = lexical_intervals
            total_clean = sum(end - start for start, end in intervals)
            if total_clean < 1.0:
                continue
            window_count = min(12, max(1, int(total_clean // 2.0)))
            partition = total_clean / window_count
            window_duration = min(6.0, partition)
            embeddings = []
            used_seconds = 0.0
            for index in range(window_count):
                samples = self._virtual_slice(
                    audio,
                    intervals,
                    index * partition,
                    window_duration,
                )
                if len(samples) < int(0.8 * SAMPLE_RATE):
                    continue
                embedding = self._encode_samples(samples)
                if embedding is None:
                    continue
                embeddings.append(embedding)
                used_seconds += len(samples) / SAMPLE_RATE
            if not embeddings:
                continue

            matrix = np.asarray(embeddings, dtype=np.float32)
            centroid = np.mean(matrix, axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
            similarities = matrix @ centroid
            if len(matrix) >= 4:
                cutoff = float(np.quantile(similarities, 0.2))
                retained = matrix[similarities >= cutoff]
                if len(retained) >= 2:
                    centroid = np.mean(retained, axis=0)
                    centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
                    similarities = retained @ centroid
            evidence[speaker] = {
                "embedding": [round(float(value), 7) for value in centroid],
                "clean_seconds": round(used_seconds, 3),
                "available_clean_seconds": round(total_clean, 3),
                "window_count": len(embeddings),
                "cohesion": round(float(np.median(similarities)), 4),
                "recognized_word_count": recognized_word_count,
                "recognized_speech_seconds": round(recognized_speech_seconds, 3),
                "word_timing_available": word_timing_available,
                "lexical_filter_applied": lexical_filter_applied,
            }
        return evidence


class Emotion2VecToneEstimator(ToneEstimator):
    """Nine-label, large-corpus utterance-level speech emotion estimator."""

    name = "emotion2vec/emotion2vec_plus_large"
    # Pin the checkpoint exercised on both the CPU development VM and the RTX
    # 3080 production host. FunASR currently ignores model_revision for its
    # Hugging Face downloader, so resolve the snapshot before using AutoModel.
    model_revision = "6c303ba987b86b93193de93e34bb2b077a6bedc4"
    version = model_revision

    def __init__(self, device: str) -> None:
        from funasr import AutoModel
        from huggingface_hub import snapshot_download

        self.device = device
        model_path = snapshot_download(
            repo_id=self.name,
            revision=self.model_revision,
        )
        # FunASR's checkpoint loader is exceptionally chatty at INFO level.
        # Keep normal transcript runs readable while retaining failures.
        captured = io.StringIO()
        prior_disable = logging.root.manager.disable
        logging.disable(logging.WARNING)
        try:
            with redirect_stdout(captured), redirect_stderr(captured):
                self.classifier = AutoModel(
                    model=model_path,
                    hub="hf",
                    device=device,
                    ncpu=max(1, min(os.cpu_count() or 1, 8)),
                    disable_update=True,
                    check_latest=False,
                    log_level="ERROR",
                )
        finally:
            logging.disable(prior_disable)
        loaded_path = Path(str(getattr(self.classifier, "model_path", "")))
        self.version = loaded_path.name or self.model_revision

    @staticmethod
    def _label(raw: Any) -> str:
        value = str(raw).strip()
        if "/" in value:
            value = value.rsplit("/", 1)[-1]
        if value.lower() in {"<unk>", "unk"}:
            return "unknown"
        return value.lower() or "unknown"

    @classmethod
    def _rank_scores(
        cls,
        labels: list[Any],
        values: list[Any],
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        """Normalize scores while retaining the model's unknown class."""
        return sorted(
            (
                {
                    "label": cls._label(label),
                    "probability": round(float(value), 4),
                }
                for label, value in zip(labels, values, strict=True)
            ),
            key=lambda item: item["probability"],
            reverse=True,
        )[:limit]

    def estimate(self, audio: Any, turns: list[dict[str, Any]]) -> None:
        import numpy as np

        for turn in turns:
            duration = float(turn["end"]) - float(turn["start"])
            if duration < 0.25:
                continue
            first = max(0, int(float(turn["start"]) * SAMPLE_RATE))
            last = min(
                len(audio),
                max(first + 1, int(float(turn["end"]) * SAMPLE_RATE)),
            )
            try:
                totals: dict[str, float] = {}
                total_weight = 0.0
                cursor = first
                max_window = 12 * SAMPLE_RATE
                windows: list[dict[str, Any]] = []
                while cursor < last:
                    window_end = min(last, cursor + max_window)
                    samples = np.array(
                        audio[cursor:window_end], dtype=np.float32, copy=True
                    )
                    if len(samples) < int(0.25 * SAMPLE_RATE):
                        break
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        prediction = self.classifier.generate(
                            input=samples,
                            granularity="utterance",
                            extract_embedding=False,
                            disable_pbar=True,
                        )
                    if not prediction or not isinstance(prediction[0], dict):
                        raise RuntimeError("emotion2vec returned no prediction")
                    labels = prediction[0].get("labels") or []
                    values = prediction[0].get("scores") or []
                    if not labels or len(labels) != len(values):
                        raise RuntimeError("emotion2vec returned invalid scores")
                    weight = len(samples) / SAMPLE_RATE
                    windows.append(
                        {
                            "start": round(cursor / SAMPLE_RATE, 3),
                            "end": round(window_end / SAMPLE_RATE, 3),
                            "scores": self._rank_scores(labels, values),
                        }
                    )
                    for label, value in zip(labels, values, strict=True):
                        normalized = self._label(label)
                        totals[normalized] = (
                            totals.get(normalized, 0.0) + float(value) * weight
                        )
                    total_weight += weight
                    cursor = window_end
                if not totals or total_weight <= 0:
                    raise RuntimeError("emotion2vec produced no usable scores")
                scores = self._rank_scores(
                    list(totals),
                    [value / total_weight for value in totals.values()],
                )
                top_window_labels = [
                    str(window["scores"][0]["label"])
                    for window in windows
                    if window.get("scores")
                ]
                turn["tone"] = {
                    "kind": "approximate_model_estimate",
                    "model": self.name,
                    "model_version": self.version,
                    "scores": scores,
                    "classification_status": (
                        "unclassified"
                        if scores and scores[0]["label"] == "unknown"
                        else "classified"
                    ),
                    "windows": windows,
                    "temporal_variation": len(set(top_window_labels)) > 1,
                    "limitations": (
                        "Utterance-level speech emotion estimate; labels describe "
                        "vocal presentation and can be wrong under domain, language, "
                        "speaker, cultural, or recording-condition shift."
                    ),
                }
            except Exception as exc:
                turn["tone"] = {
                    "kind": "unavailable",
                    "model": self.name,
                    "scores": [],
                    "error": str(exc),
                }


class SpeechBrainToneEstimator(ToneEstimator):
    name = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
    version = "model-main"

    LABELS = {"ang": "angry", "hap": "happy", "neu": "neutral", "sad": "sad"}

    def __init__(self, device: str) -> None:
        from speechbrain.inference.interfaces import foreign_class

        self.device = device
        savedir = _model_cache("speechbrain-tone-iemocap")
        original_directory = Path.cwd()
        try:
            # This model's published hyperparameters use a relative transformers
            # cache path. Anchor it in our cache instead of polluting the caller's
            # working directory.
            os.chdir(savedir.parent)
            self.classifier = foreign_class(
                source=self.name,
                savedir=str(savedir),
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier",
                run_opts={"device": device},
            )
        finally:
            os.chdir(original_directory)
        self.classifier.hparams.label_encoder.expect_len(len(self.LABELS))

    def _labels(self, count: int) -> list[str]:
        import torch

        encoder = self.classifier.hparams.label_encoder
        labels: list[str] = []
        for index in range(count):
            decoded = encoder.decode_torch(torch.tensor([index]))
            value = str(decoded[0] if isinstance(decoded, (list, tuple)) else decoded)
            labels.append(self.LABELS.get(value.lower(), value.lower()))
        return labels

    def estimate(self, audio: Any, turns: list[dict[str, Any]]) -> None:
        import numpy as np
        import torch

        for turn in turns:
            duration = turn["end"] - turn["start"]
            if duration < 0.25:
                continue
            first = max(0, int(turn["start"] * SAMPLE_RATE))
            last = min(
                len(audio),
                max(first + 1, int(min(turn["end"], turn["start"] + 30) * SAMPLE_RATE)),
            )
            samples = np.asarray(audio[first:last], dtype=np.float32)
            waveform = torch.from_numpy(samples).unsqueeze(0).to(self.device)
            lengths = torch.ones(1, device=self.device)
            try:
                with torch.inference_mode():
                    probabilities, _score, _class_index, _label = (
                        self.classifier.classify_batch(waveform, lengths)
                    )
                    values = probabilities.detach().cpu().reshape(-1).tolist()
                    if (
                        not values
                        or any(value < 0.0 or value > 1.0 for value in values)
                        or not 0.98 <= sum(values) <= 1.02
                    ):
                        largest = max(values, default=0.0)
                        exponentials = [
                            pow(2.718281828459045, value - largest) for value in values
                        ]
                        total = sum(exponentials) or 1.0
                        values = [value / total for value in exponentials]
                    labels = self._labels(len(values))
                    scores = sorted(
                        (
                            {"label": label, "probability": round(float(value), 4)}
                            for label, value in zip(labels, values, strict=True)
                        ),
                        key=lambda item: item["probability"],
                        reverse=True,
                    )[:3]
                turn["tone"] = {
                    "kind": "approximate_model_estimate",
                    "model": self.name,
                    "model_version": package_version("speechbrain"),
                    "scores": scores,
                    "limitations": (
                        "English IEMOCAP model estimate; recording conditions and "
                        "domain mismatch can substantially affect the result."
                    ),
                }
            except Exception as exc:
                turn["tone"] = {
                    "kind": "unavailable",
                    "model": self.name,
                    "scores": [],
                    "error": str(exc),
                }


def release_accelerator_memory(device: str) -> None:
    gc.collect()
    if device == "cuda":
        import torch

        torch.cuda.empty_cache()
