"""MuQ embedding extraction and export helpers."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from muq_beat_weaver.model.audio import _find_audio_in_folder, load_audio

_AUDIO_EXTENSIONS = (".wav", ".ogg", ".egg", ".mp3", ".flac")
_DEFAULT_MUQ_LABEL_RATE = 25.0
_DEFAULT_WINDOW_SECONDS = 180.0
_DEFAULT_OVERLAP_SECONDS = 30.0


@dataclass
class MuQEmbeddingStats:
    """Summary statistics for a single embedding export."""

    audio_path: str
    embedding_path: str
    sample_rate: int
    audio_seconds: float
    embedding_shape: list[int]
    embedding_dtype: str
    embedding_bytes: int
    load_audio_seconds: float
    inference_seconds: float
    save_seconds: float
    total_seconds: float
    mean_abs: float
    std: float
    min_value: float
    max_value: float
    contains_nan: bool


class MuQEmbedder:
    """Thin wrapper around a pretrained MuQ model."""

    def __init__(
        self,
        model_name: str = "OpenMuQ/MuQ-large-msd-iter",
        device: str | None = None,
        *,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        from muq import MuQ

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.local_files_only = local_files_only
        self.model = MuQ.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
            local_files_only=local_files_only,
        ).to(self.device).eval()
        self.label_rate = float(
            getattr(getattr(self.model, "config", None), "label_rate", _DEFAULT_MUQ_LABEL_RATE)
        )

    def extract_file(self, audio_path: Path, sample_rate: int = 24000) -> tuple[np.ndarray, dict[str, float]]:
        """Load an audio file and extract MuQ last_hidden_state embeddings."""
        audio_path = Path(audio_path)

        load_t0 = time.perf_counter()
        audio, sr = load_audio(audio_path, sr=sample_rate)
        load_audio_seconds = time.perf_counter() - load_t0

        infer_t0 = time.perf_counter()
        with torch.no_grad():
            wav = torch.from_numpy(audio).unsqueeze(0).float().to(self.device)
            output = self.model(wav)
            embedding = output.last_hidden_state[0].detach().cpu().numpy().astype(np.float32)
        inference_seconds = time.perf_counter() - infer_t0

        return embedding, {
            "sample_rate": float(sr),
            "audio_seconds": float(len(audio) / sr),
            "load_audio_seconds": load_audio_seconds,
            "inference_seconds": inference_seconds,
        }

    def extract_file_windowed(
        self,
        audio_path: Path,
        sample_rate: int = 24000,
        *,
        window_seconds: float = _DEFAULT_WINDOW_SECONDS,
        overlap_seconds: float = _DEFAULT_OVERLAP_SECONDS,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Extract MuQ embeddings in overlapping windows and merge by midpoint ownership."""
        audio_path = Path(audio_path)
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")
        if overlap_seconds < 0:
            raise ValueError("overlap_seconds must be non-negative.")
        if overlap_seconds >= window_seconds:
            raise ValueError("overlap_seconds must be smaller than window_seconds.")

        load_t0 = time.perf_counter()
        audio, sr = load_audio(audio_path, sr=sample_rate)
        load_audio_seconds = time.perf_counter() - load_t0
        audio_seconds = float(len(audio) / sr)
        windows = _build_window_plan(
            audio_seconds,
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
        )

        infer_t0 = time.perf_counter()
        with torch.no_grad():
            if len(windows) == 1:
                wav = torch.from_numpy(audio).unsqueeze(0).float().to(self.device)
                output = self.model(wav)
                embedding = output.last_hidden_state[0].detach().cpu().numpy().astype(np.float32)
            else:
                parts: list[np.ndarray] = []
                for start_sec, end_sec in windows:
                    start_sample = int(round(start_sec * sr))
                    end_sample = min(len(audio), int(round(end_sec * sr)))
                    wav = torch.from_numpy(audio[start_sample:end_sample]).unsqueeze(0).float().to(self.device)
                    output = self.model(wav)
                    parts.append(output.last_hidden_state[0].detach().cpu().numpy().astype(np.float32))
                embedding = _merge_window_embeddings(parts, windows, frame_hz=self.label_rate)
        inference_seconds = time.perf_counter() - infer_t0

        return embedding, {
            "sample_rate": float(sr),
            "audio_seconds": audio_seconds,
            "load_audio_seconds": load_audio_seconds,
            "inference_seconds": inference_seconds,
        }


def summarize_embedding(
    embedding: np.ndarray,
    *,
    audio_path: Path,
    embedding_path: Path,
    sample_rate: int,
    audio_seconds: float,
    load_audio_seconds: float,
    inference_seconds: float,
    save_seconds: float,
) -> MuQEmbeddingStats:
    """Build a serializable summary for one embedding."""
    return MuQEmbeddingStats(
        audio_path=str(audio_path),
        embedding_path=str(embedding_path),
        sample_rate=sample_rate,
        audio_seconds=audio_seconds,
        embedding_shape=list(embedding.shape),
        embedding_dtype=str(embedding.dtype),
        embedding_bytes=int(embedding.nbytes),
        load_audio_seconds=load_audio_seconds,
        inference_seconds=inference_seconds,
        save_seconds=save_seconds,
        total_seconds=load_audio_seconds + inference_seconds + save_seconds,
        mean_abs=float(np.abs(embedding).mean()),
        std=float(embedding.std()),
        min_value=float(embedding.min()),
        max_value=float(embedding.max()),
        contains_nan=bool(np.isnan(embedding).any()),
    )


def find_audio_files_in_subfolders(
    root: Path,
    *,
    limit: int | None = None,
) -> list[Path]:
    """Find one audio file in each first-level subfolder, sorted by folder name."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")

    audio_files: list[Path] = []
    subfolders = sorted(path for path in root.iterdir() if path.is_dir())

    for folder in subfolders:
        info_dat = folder / "Info.dat"
        info_dat_lower = folder / "info.dat"
        audio_path = None
        if info_dat.exists():
            audio_path = _find_audio_in_folder(folder, info_dat)
        elif info_dat_lower.exists():
            audio_path = _find_audio_in_folder(folder, info_dat_lower)

        if audio_path is None:
            for ext in _AUDIO_EXTENSIONS:
                matches = sorted(folder.glob(f"*{ext}"))
                if matches:
                    audio_path = matches[0]
                    break

        if audio_path is None:
            continue

        audio_files.append(audio_path)
        if limit is not None and len(audio_files) >= limit:
            break

    return audio_files


def export_embeddings(
    audio_paths: list[Path],
    output_dir: Path,
    *,
    model_name: str = "OpenMuQ/MuQ-large-msd-iter",
    device: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    sample_rate: int = 24000,
    window_seconds: float = _DEFAULT_WINDOW_SECONDS,
    overlap_seconds: float = _DEFAULT_OVERLAP_SECONDS,
) -> dict[str, object]:
    """Extract and save MuQ embeddings for a list of audio paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedder = MuQEmbedder(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    (output_dir / "VERSION").write_text(
        _muq_cache_version_key(
            model_name=model_name,
            label_rate=embedder.label_rate,
            sample_rate=sample_rate,
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
        ),
        encoding="utf-8",
    )
    items: list[MuQEmbeddingStats] = []
    started_at = time.perf_counter()

    for audio_path in audio_paths:
        audio_path = Path(audio_path)
        output_name = f"{audio_path.parent.name}_{audio_path.stem}.npy"
        embedding_path = output_dir / output_name
        stats_path = output_dir / f"{audio_path.parent.name}_{audio_path.stem}.json"

        embedding, timing = embedder.extract_file_windowed(
            audio_path,
            sample_rate=sample_rate,
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
        )

        save_t0 = time.perf_counter()
        np.save(embedding_path, embedding)
        save_seconds = time.perf_counter() - save_t0

        stats = summarize_embedding(
            embedding,
            audio_path=audio_path,
            embedding_path=embedding_path,
            sample_rate=int(timing["sample_rate"]),
            audio_seconds=timing["audio_seconds"],
            load_audio_seconds=timing["load_audio_seconds"],
            inference_seconds=timing["inference_seconds"],
            save_seconds=save_seconds,
        )
        stats_path.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")
        items.append(stats)

    total_seconds = time.perf_counter() - started_at
    summary = {
        "model_name": model_name,
        "device": str(embedder.device),
        "num_files": len(items),
        "total_wall_seconds": total_seconds,
        "items": [asdict(item) for item in items],
        "aggregate": _aggregate_stats(items),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _build_window_plan(
    audio_seconds: float,
    *,
    window_seconds: float,
    overlap_seconds: float,
) -> list[tuple[float, float]]:
    """Build overlapping window boundaries in seconds."""
    if audio_seconds <= 0:
        return [(0.0, 0.0)]
    if audio_seconds <= window_seconds:
        return [(0.0, audio_seconds)]

    stride = window_seconds - overlap_seconds
    starts: list[float] = []
    start = 0.0
    while start < audio_seconds:
        starts.append(start)
        if start + window_seconds >= audio_seconds:
            break
        start += stride
    return [(s, min(audio_seconds, s + window_seconds)) for s in starts]


def _merge_window_embeddings(
    windows: list[np.ndarray],
    window_ranges: list[tuple[float, float]],
    *,
    frame_hz: float,
) -> np.ndarray:
    """Merge windowed MuQ features by assigning overlap ownership at midpoints."""
    if not windows:
        return np.zeros((0, 0), dtype=np.float32)
    if len(windows) != len(window_ranges):
        raise ValueError("windows and window_ranges must have the same length.")
    if len(windows) == 1:
        return windows[0].astype(np.float32, copy=False)

    merged: list[np.ndarray] = []
    for idx, (embedding, (start_sec, end_sec)) in enumerate(zip(windows, window_ranges, strict=False)):
        if embedding.ndim != 2:
            raise ValueError("Each embedding window must be 2D (time, hidden).")
        if embedding.shape[0] == 0:
            continue

        min_keep = -math.inf
        max_keep = math.inf
        if idx > 0:
            prev_start, prev_end = window_ranges[idx - 1]
            min_keep = (start_sec + prev_end) / 2.0
        if idx < len(window_ranges) - 1:
            next_start, _ = window_ranges[idx + 1]
            max_keep = (end_sec + next_start) / 2.0

        local_times = start_sec + (np.arange(embedding.shape[0], dtype=np.float32) / frame_hz)
        keep = local_times >= min_keep
        if idx < len(window_ranges) - 1:
            keep &= local_times < max_keep
        if not np.any(keep):
            center_idx = min(
                embedding.shape[0] - 1,
                max(0, int(round(((min_keep + max_keep) * 0.5 - start_sec) * frame_hz))),
            )
            keep = np.zeros(embedding.shape[0], dtype=bool)
            keep[center_idx] = True
        merged.append(embedding[keep])

    if not merged:
        return windows[0].astype(np.float32, copy=False)
    return np.concatenate(merged, axis=0).astype(np.float32, copy=False)


def _muq_cache_version_key(
    *,
    model_name: str,
    label_rate: float,
    sample_rate: int,
    window_seconds: float,
    overlap_seconds: float,
) -> str:
    return (
        "muq_cache_v2\n"
        f"model={model_name}\n"
        f"label_rate={label_rate:.6f}\n"
        f"sample_rate={sample_rate}\n"
        f"window_seconds={window_seconds:.3f}\n"
        f"overlap_seconds={overlap_seconds:.3f}\n"
        "merge=midpoint_ownership\n"
    )


def _aggregate_stats(items: list[MuQEmbeddingStats]) -> dict[str, float | int | list[int]]:
    """Aggregate output size and timing statistics across exports."""
    if not items:
        return {
            "total_embedding_bytes": 0,
            "mean_embedding_bytes": 0.0,
            "mean_audio_seconds": 0.0,
            "mean_inference_seconds": 0.0,
            "max_inference_seconds": 0.0,
            "mean_total_seconds": 0.0,
            "shape_set": [],
        }

    return {
        "total_embedding_bytes": int(sum(item.embedding_bytes for item in items)),
        "mean_embedding_bytes": float(np.mean([item.embedding_bytes for item in items])),
        "mean_audio_seconds": float(np.mean([item.audio_seconds for item in items])),
        "mean_inference_seconds": float(np.mean([item.inference_seconds for item in items])),
        "max_inference_seconds": float(max(item.inference_seconds for item in items)),
        "mean_total_seconds": float(np.mean([item.total_seconds for item in items])),
        "shape_set": sorted({tuple(item.embedding_shape) for item in items}),
    }


