from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from muq_beat_weaver.model.audio import load_audio

logger = logging.getLogger(__name__)

TIMING_METADATA_FILENAME = "timing_metadata.json"
_NEGATIVE_BEAT_TOLERANCE = 1.0 / 32.0


@dataclass(slots=True)
class SongTiming:
    song_hash: str
    bpm: float
    first_downbeat_sec: float
    timing_source: str
    timing_confidence: str = "unknown"
    needs_review: bool = False
    beat_count: int = 0
    meter: int = 4
    notes_dropped_before_downbeat: int = 0
    review_reasons: list[str] | None = None
    audio_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["review_reasons"] is None:
            payload["review_reasons"] = []
        return payload


def timing_metadata_path(processed_dir: Path, explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)
    return Path(processed_dir) / TIMING_METADATA_FILENAME


def load_metadata_dict(processed_dir: Path) -> dict[str, dict]:
    meta_path = Path(processed_dir) / "metadata.json"
    raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if isinstance(raw_meta, list):
        return {item["hash"]: item for item in raw_meta}
    return raw_meta


def build_default_timing(song_hash: str, metadata_entry: dict[str, Any] | None = None) -> SongTiming:
    metadata_entry = metadata_entry or {}
    bpm = float(metadata_entry.get("bpm", 0.0) or 0.0)
    return SongTiming(
        song_hash=song_hash,
        bpm=bpm,
        first_downbeat_sec=0.0,
        timing_source="metadata_bpm_only" if bpm > 0 else "missing",
        timing_confidence="medium" if bpm > 0 else "low",
        needs_review=bpm <= 0,
        review_reasons=[] if bpm > 0 else ["missing_bpm"],
    )


def _normalize_timing_entry(song_hash: str, payload: dict[str, Any]) -> SongTiming:
    return SongTiming(
        song_hash=song_hash,
        bpm=float(payload.get("bpm", 0.0) or 0.0),
        first_downbeat_sec=float(payload.get("first_downbeat_sec", 0.0) or 0.0),
        timing_source=str(payload.get("timing_source", payload.get("source", "unknown"))),
        timing_confidence=str(payload.get("timing_confidence", "unknown")),
        needs_review=bool(payload.get("needs_review", False)),
        beat_count=int(payload.get("beat_count", 0) or 0),
        meter=int(payload.get("meter", 4) or 4),
        notes_dropped_before_downbeat=int(payload.get("notes_dropped_before_downbeat", 0) or 0),
        review_reasons=list(payload.get("review_reasons", []) or []),
        audio_path=payload.get("audio_path"),
    )


def load_timing_metadata(
    processed_dir: Path,
    *,
    timing_path: Path | None = None,
    metadata: dict[str, dict] | None = None,
) -> dict[str, dict[str, Any]]:
    metadata = metadata or load_metadata_dict(processed_dir)
    resolved_path = timing_metadata_path(processed_dir, timing_path)
    timing_lookup: dict[str, dict[str, Any]] = {}

    if resolved_path.exists():
        raw = json.loads(resolved_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            for item in raw:
                song_hash = str(item["song_hash"])
                timing_lookup[song_hash] = _normalize_timing_entry(song_hash, item).to_dict()
        else:
            for song_hash, item in raw.items():
                timing_lookup[song_hash] = _normalize_timing_entry(song_hash, item).to_dict()

    for song_hash, meta in metadata.items():
        timing_lookup.setdefault(song_hash, build_default_timing(song_hash, meta).to_dict())

    return timing_lookup


def save_timing_metadata(path: Path, timing_lookup: dict[str, dict[str, Any]]) -> None:
    normalized = {
        song_hash: _normalize_timing_entry(song_hash, payload).to_dict()
        for song_hash, payload in sorted(timing_lookup.items())
    }
    Path(path).write_text(json.dumps(normalized, indent=2), encoding="utf-8")


def timing_fingerprint(timing_entry: dict[str, Any]) -> str:
    import hashlib

    stable = {
        "bpm": round(float(timing_entry.get("bpm", 0.0) or 0.0), 6),
        "first_downbeat_sec": round(float(timing_entry.get("first_downbeat_sec", 0.0) or 0.0), 6),
        "timing_source": timing_entry.get("timing_source", "unknown"),
        "timing_confidence": timing_entry.get("timing_confidence", "unknown"),
        "meter": int(timing_entry.get("meter", 4) or 4),
    }
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def rebase_note_dicts_to_timing(
    note_dicts: list[dict[str, Any]],
    *,
    timing_entry: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    bpm = float(timing_entry.get("bpm", 0.0) or 0.0)
    first_downbeat_sec = float(timing_entry.get("first_downbeat_sec", 0.0) or 0.0)
    if bpm <= 0:
        raise ValueError("Timing BPM must be positive to rebase notes.")

    rebased: list[dict[str, Any]] = []
    dropped = 0
    for note in note_dicts:
        source_bpm = float(note.get("bpm", bpm) or bpm)
        time_seconds = note.get("time_seconds")
        if time_seconds is None:
            if source_bpm <= 0:
                continue
            time_seconds = float(note["beat"]) * 60.0 / source_bpm
        absolute_seconds = float(time_seconds)
        aligned_beat = (absolute_seconds - first_downbeat_sec) * bpm / 60.0
        if aligned_beat < 0.0:
            if aligned_beat >= -_NEGATIVE_BEAT_TOLERANCE:
                aligned_beat = 0.0
            else:
                dropped += 1
                continue
        rebased_note = dict(note)
        rebased_note["original_beat"] = float(note.get("beat", 0.0) or 0.0)
        rebased_note["absolute_time_seconds"] = absolute_seconds
        rebased_note["beat"] = float(aligned_beat)
        rebased_note["time_seconds"] = float(aligned_beat * 60.0 / bpm)
        rebased_note["bpm"] = bpm
        rebased.append(rebased_note)

    rebased.sort(key=lambda item: (item["beat"], item["color"], item["x"], item["y"]))
    return rebased, dropped


def resolve_single_song_timing(
    timing_payload: dict[str, Any],
    *,
    timing_hash: str | None = None,
) -> dict[str, Any]:
    if "bpm" in timing_payload:
        song_hash = timing_hash or str(timing_payload.get("song_hash", "inference"))
        return _normalize_timing_entry(song_hash, timing_payload).to_dict()

    if timing_hash is None:
        raise ValueError("timing_hash is required when timing metadata contains multiple songs.")
    try:
        return _normalize_timing_entry(timing_hash, timing_payload[timing_hash]).to_dict()
    except KeyError as exc:
        raise KeyError(f"Timing metadata missing entry for {timing_hash!r}") from exc


def estimate_song_timing(
    audio_path: Path,
    *,
    song_hash: str = "inference",
    bpm_hint: float | None = None,
    sample_rate: int = 22050,
    beats_per_bar: tuple[int, ...] = (4,),
) -> dict[str, Any]:
    audio_path = Path(audio_path)
    try:
        return _estimate_song_timing_madmom(
            audio_path,
            song_hash=song_hash,
            bpm_hint=bpm_hint,
            beats_per_bar=beats_per_bar,
        ).to_dict()
    except Exception as exc:
        logger.debug("madmom timing failed for %s: %s: %s", audio_path, type(exc).__name__, exc)
        return _estimate_song_timing_librosa(
            audio_path,
            song_hash=song_hash,
            bpm_hint=bpm_hint,
            sample_rate=sample_rate,
        ).to_dict()


def _estimate_song_timing_librosa(
    audio_path: Path,
    *,
    song_hash: str,
    bpm_hint: float | None,
    sample_rate: int,
) -> SongTiming:
    audio, sr = load_audio(audio_path, sr=sample_rate)
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
    tempo = float(np.asarray(tempo).reshape(-1)[0]) if np.size(tempo) else 0.0
    beat_frames = np.asarray(beat_frames).reshape(-1)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr) if beat_frames.size else np.empty(0)

    bpm = tempo
    if bpm <= 0 and bpm_hint and bpm_hint > 0:
        bpm = float(bpm_hint)
    first_beat = float(beat_times[0]) if beat_times.size else 0.0
    review_reasons: list[str] = []
    if beat_times.size == 0:
        review_reasons.append("no_beats_detected")
    if bpm_hint and bpm > 0:
        ratio = max(bpm, bpm_hint) / max(min(bpm, bpm_hint), 1e-6)
        if ratio >= 1.9:
            review_reasons.append("tempo_disagrees_with_metadata")
    return SongTiming(
        song_hash=song_hash,
        bpm=float(bpm),
        first_downbeat_sec=first_beat,
        timing_source="librosa_first_beat",
        timing_confidence="low",
        needs_review=True,
        beat_count=int(beat_times.size),
        meter=4,
        review_reasons=review_reasons or ["downbeat_not_verified"],
        audio_path=str(audio_path),
    )


def _estimate_song_timing_madmom(
    audio_path: Path,
    *,
    song_hash: str,
    bpm_hint: float | None,
    beats_per_bar: tuple[int, ...],
) -> SongTiming:
    _prepare_madmom_runtime()
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor, _process_dbn

    import itertools as it

    def patched_process(self, activations, **kwargs):
        first = 0
        if self.threshold:
            idx = np.nonzero(activations >= self.threshold)[0]
            if idx.any():
                first = max(first, np.min(idx))
                last = min(len(activations), np.max(idx) + 1)
            else:
                last = first
            activations = activations[first:last]
        if not activations.any():
            return np.empty((0, 2))
        results = list(self.map(_process_dbn, zip(self.hmms, it.repeat(activations))))
        best = int(np.argmax([score for _, score in results]))
        path, _ = results[best]
        st = self.hmms[best].transition_model.state_space
        om = self.hmms[best].observation_model
        positions = st.state_positions[path]
        beat_numbers = positions.astype(int) + 1
        if self.correct:
            beats = np.empty(0, dtype=int)
            beat_range = om.pointers[path] >= 1
            idx = np.nonzero(np.diff(beat_range.astype(int)))[0] + 1
            if beat_range[0]:
                idx = np.r_[0, idx]
            if beat_range[-1]:
                idx = np.r_[idx, beat_range.size]
            if idx.any():
                for left, right in idx.reshape((-1, 2)):
                    peak = np.argmax(activations[left:right]) // 2 + left
                    beats = np.hstack((beats, peak))
        else:
            beats = np.nonzero(np.diff(beat_numbers))[0] + 1
        return np.vstack(((beats + first) / float(self.fps), beat_numbers[beats])).T

    DBNDownBeatTrackingProcessor.process = patched_process

    activations = RNNDownBeatProcessor()(str(audio_path))
    tracker = DBNDownBeatTrackingProcessor(beats_per_bar=list(beats_per_bar), fps=100)
    result = tracker(activations)
    if result.size == 0:
        raise RuntimeError("madmom returned no beats")

    times = result[:, 0]
    beat_numbers = result[:, 1].astype(int)
    downbeats = times[beat_numbers == 1]
    if downbeats.size == 0:
        raise RuntimeError("madmom returned beats but no downbeats")
    ibis = np.diff(times)
    bpm = float(60.0 / np.median(ibis)) if ibis.size else float(bpm_hint or 0.0)

    review_reasons: list[str] = []
    if bpm_hint and bpm > 0:
        ratio = max(bpm, bpm_hint) / max(min(bpm, bpm_hint), 1e-6)
        if ratio >= 1.9:
            review_reasons.append("tempo_disagrees_with_metadata")

    return SongTiming(
        song_hash=song_hash,
        bpm=bpm,
        first_downbeat_sec=float(downbeats[0]),
        timing_source="madmom",
        timing_confidence="high",
        needs_review=bool(review_reasons),
        beat_count=int(times.size),
        meter=int(max(beats_per_bar)),
        review_reasons=review_reasons,
        audio_path=str(audio_path),
    )


def _prepare_madmom_runtime() -> None:
    import collections
    import collections.abc

    for name in ("MutableSequence", "MutableMapping", "Sequence", "Mapping"):
        if not hasattr(collections, name) and hasattr(collections.abc, name):
            setattr(collections, name, getattr(collections.abc, name))

    for name, value in (
        ("float", float),
        ("int", int),
        ("bool", bool),
        ("object", object),
        ("complex", complex),
    ):
        if not hasattr(np, name):
            setattr(np, name, value)
