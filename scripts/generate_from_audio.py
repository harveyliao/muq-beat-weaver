from __future__ import annotations

import argparse
import json
import logging
import math
import re
import statistics
from pathlib import Path

import torch

from muq_beat_weaver.model.audio import interpolate_muq_to_beat_grid
from muq_beat_weaver.model.config import ModelConfig
from muq_beat_weaver.model.evaluate import evaluate_standalone
from muq_beat_weaver.model.exporter import export_notes
from muq_beat_weaver.model.inference import generate_full_song
from muq_beat_weaver.model.muq_embeddings import MuQEmbedder
from muq_beat_weaver.model.timing import estimate_song_timing, resolve_single_song_timing
from muq_beat_weaver.schemas.normalized import Note
from muq_beat_weaver.model.transformer import BeatWeaverModel

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a Beat Saber map from one audio file and a trained checkpoint.",
    )
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "output" / "muq_precomputed_2000_45ep_base_bs8" / "checkpoints" / "best",
        help="Checkpoint directory containing model.pt and config.json.",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=repo_root / "data" / "inference" / "song.ogg",
        help="Path to the input audio file.",
    )
    parser.add_argument(
        "--bpm",
        type=float,
        default=None,
        help="Song BPM used for beat-grid alignment and map export. Overrides timing metadata if provided.",
    )
    parser.add_argument(
        "--song-name",
        type=str,
        default=None,
        help="Display name for the generated map. Defaults to the audio filename stem.",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="Expert",
        choices=["Easy", "Normal", "Hard", "Expert", "ExpertPlus"],
        help="Target Beat Saber difficulty token.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the generated Beat Saber map folder.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override, for example 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling filter. 0 disables it.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling filter. 1.0 disables it.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for generation.",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=6,
        help="Number of seed candidates to generate and rerank.",
    )
    parser.add_argument(
        "--retry-empty",
        type=int,
        default=4,
        help="Legacy minimum retry count for empty generations. Attempts use max(candidates, retry_empty + 1).",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="Increment applied between retry seeds.",
    )
    parser.add_argument(
        "--export-all-candidates",
        action="store_true",
        help="Export each generated candidate into its own subfolder under the output directory.",
    )
    parser.add_argument(
        "--inference-mode",
        type=str,
        default="independent",
        choices=["independent", "rolling"],
        help="Long-song inference strategy for songs that exceed one audio window.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Audio sample rate for MuQ feature extraction.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=180.0,
        help="MuQ extraction window size for long songs.",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=30.0,
        help="MuQ extraction overlap size for long songs.",
    )
    parser.add_argument(
        "--muq-model",
        type=Path,
        default=None,
        help="Optional local path to a downloaded MuQ model directory. Defaults to the checkpoint config model id.",
    )
    parser.add_argument(
        "--muq-cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory to use for MuQ model files.",
    )
    parser.add_argument(
        "--muq-local-only",
        action="store_true",
        help="Load MuQ from a local directory or existing cache only, without contacting the Hugging Face Hub.",
    )
    parser.add_argument(
        "--timing-metadata",
        type=Path,
        default=None,
        help="Optional path to timing metadata JSON. Can be a single-song entry or a corpus mapping.",
    )
    parser.add_argument(
        "--timing-hash",
        type=str,
        default=None,
        help="Song hash used to select an entry from --timing-metadata when it contains multiple songs.",
    )
    parser.add_argument(
        "--auto-beat-offset",
        action="store_true",
        help="Estimate timing from the audio and align the beat grid to it when no manual timing is supplied.",
    )
    parser.add_argument(
        "--beat-offset-seconds",
        type=float,
        default=None,
        help="Manual first downbeat offset in seconds. Overrides timing metadata and auto detection.",
    )
    parser.add_argument(
        "--beat-offset-beats",
        type=float,
        default=None,
        help="Manual first downbeat offset in beats. Overrides seconds, timing metadata, and auto detection.",
    )
    parser.add_argument(
        "--beat-detect-sr",
        type=int,
        default=22050,
        help="Sample rate used for automatic timing fallback detection.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    return cleaned or "generated_song"


def _resolve_output_dir(repo_root: Path, requested: Path | None, song_name: str) -> Path:
    if requested is not None:
        return requested
    return repo_root / "output" / "generated" / _sanitize_name(song_name)


def _candidate_output_dir(base_output_dir: Path, seed: int) -> Path:
    return base_output_dir / f"candidate_{seed}"


def _load_checkpoint(checkpoint_dir: Path, device: torch.device) -> tuple[BeatWeaverModel, ModelConfig]:
    checkpoint_dir = Path(checkpoint_dir)
    config_path = checkpoint_dir / "config.json"
    model_path = checkpoint_dir / "model.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"Checkpoint config not found: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint weights not found: {model_path}")

    config = ModelConfig.load(config_path)
    model = BeatWeaverModel(config)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, config


def _resolve_generation_timing(args: argparse.Namespace, audio_path: Path) -> dict:
    timing_entry = None
    if args.timing_metadata is not None:
        timing_payload = json.loads(Path(args.timing_metadata).read_text(encoding="utf-8"))
        timing_entry = resolve_single_song_timing(timing_payload, timing_hash=args.timing_hash)

    if timing_entry is None and args.auto_beat_offset:
        timing_entry = estimate_song_timing(
            audio_path,
            song_hash=args.timing_hash or audio_path.stem,
            bpm_hint=args.bpm,
            sample_rate=args.beat_detect_sr,
        )

    bpm = args.bpm
    if bpm is None and timing_entry is not None:
        bpm = float(timing_entry["bpm"])
    if bpm is None or bpm <= 0:
        raise ValueError("A positive BPM is required. Provide --bpm or a timing metadata entry with bpm.")

    beat_offset_seconds = 0.0
    beat_offset_source = "none"
    if timing_entry is not None:
        beat_offset_seconds = float(timing_entry.get("first_downbeat_sec", 0.0) or 0.0)
        beat_offset_source = str(timing_entry.get("timing_source", "timing_metadata"))
    if args.beat_offset_seconds is not None:
        beat_offset_seconds = float(args.beat_offset_seconds)
        beat_offset_source = "manual_seconds"
    if args.beat_offset_beats is not None:
        beat_offset_seconds = float(args.beat_offset_beats) * 60.0 / float(bpm)
        beat_offset_source = "manual_beats"

    return {
        "bpm": float(bpm),
        "beat_offset_seconds": beat_offset_seconds,
        "beat_offset_source": beat_offset_source,
        "timing_entry": timing_entry,
    }


def _shift_notes(notes: list[Note], beat_offset: float, bpm: float) -> list[Note]:
    if beat_offset == 0.0:
        return notes
    shifted: list[Note] = []
    for note in notes:
        new_beat = note.beat + beat_offset
        shifted.append(
            Note(
                beat=new_beat,
                time_seconds=new_beat * 60.0 / bpm,
                x=note.x,
                y=note.y,
                color=note.color,
                cut_direction=note.cut_direction,
                angle_offset=note.angle_offset,
            )
        )
    return shifted


def _notes_per_bar(notes: list[Note], total_bars: int) -> list[int]:
    counts = [0] * max(1, total_bars)
    for note in notes:
        bar_idx = int(note.beat // 4.0)
        if 0 <= bar_idx < len(counts):
            counts[bar_idx] += 1
    return counts


def _longest_zero_run(values: list[int]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value == 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _chunk_means(values: list[int], chunk_size: int) -> list[float]:
    if not values:
        return [0.0]
    means: list[float] = []
    for start in range(0, len(values), chunk_size):
        chunk = values[start: start + chunk_size]
        means.append(statistics.fmean(chunk))
    return means or [0.0]


def _target_nps_for_difficulty(difficulty: str) -> float:
    targets = {
        "Easy": 1.0,
        "Normal": 1.8,
        "Hard": 2.8,
        "Expert": 3.6,
        "ExpertPlus": 4.8,
    }
    return targets.get(difficulty, 3.0)


def _score_candidate(
    notes: list[Note],
    *,
    bpm: float,
    total_bars: int,
    difficulty: str,
) -> tuple[float, dict[str, float | int]]:
    if not notes:
        return -1_000_000.0, {
            "note_count": 0,
            "score": -1_000_000.0,
            "active_bar_ratio": 0.0,
            "mean_notes_per_bar": 0.0,
            "section_density_std": 0.0,
            "section_density_range": 0.0,
            "half_density_delta": 0.0,
            "longest_silence_bars": total_bars,
            "longest_silence_ratio": 1.0,
            **{k: round(float(v), 6) for k, v in evaluate_standalone(notes, bpm).items()},
        }

    standalone = evaluate_standalone(notes, bpm)
    bar_counts = _notes_per_bar(notes, total_bars)
    active_bar_ratio = sum(1 for count in bar_counts if count > 0) / max(1, len(bar_counts))
    mean_notes_per_bar = statistics.fmean(bar_counts)
    section_means = _chunk_means(bar_counts, chunk_size=8)
    section_density_std = statistics.pstdev(section_means) if len(section_means) > 1 else 0.0
    section_density_range = max(section_means) - min(section_means) if section_means else 0.0
    midpoint = max(1, len(bar_counts) // 2)
    first_half = statistics.fmean(bar_counts[:midpoint])
    second_half = statistics.fmean(bar_counts[midpoint:]) if midpoint < len(bar_counts) else first_half
    half_density_delta = abs(first_half - second_half)
    longest_silence_bars = _longest_zero_run(bar_counts)
    longest_silence_ratio = longest_silence_bars / max(1, len(bar_counts))
    target_nps = _target_nps_for_difficulty(difficulty)
    nps = float(standalone["nps"])
    nps_error = abs(nps - target_nps) / max(1.0, target_nps)

    score = (
        2.2 * float(standalone["pattern_diversity"])
        + 3.2 * active_bar_ratio
        + 0.20 * section_density_std
        + 0.12 * section_density_range
        + 0.08 * half_density_delta
        + 0.30 * min(mean_notes_per_bar, 8.0)
        - 4.0 * float(standalone["parity_violation_rate"])
        - 3.2 * longest_silence_ratio
        - 2.4 * nps_error
    )

    summary: dict[str, float | int] = {
        "note_count": len(notes),
        "score": round(score, 6),
        "active_bar_ratio": round(active_bar_ratio, 6),
        "mean_notes_per_bar": round(mean_notes_per_bar, 6),
        "section_density_std": round(section_density_std, 6),
        "section_density_range": round(section_density_range, 6),
        "half_density_delta": round(half_density_delta, 6),
        "longest_silence_bars": longest_silence_bars,
        "longest_silence_ratio": round(longest_silence_ratio, 6),
        "target_nps": round(target_nps, 6),
        "nps_error": round(nps_error, 6),
        **{k: round(float(v), 6) for k, v in standalone.items()},
    }
    return score, summary


def _resolve_muq_model_source(args: argparse.Namespace, config: ModelConfig) -> str:
    if args.muq_model is None:
        return str(config.muq_model_name)
    model_path = args.muq_model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"MuQ model directory not found: {model_path}")
    if not model_path.is_dir():
        raise NotADirectoryError(f"MuQ model path is not a directory: {model_path}")
    return str(model_path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_dir = args.checkpoint.resolve()
    audio_path = args.audio.resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    song_name = args.song_name or audio_path.stem
    output_dir = _resolve_output_dir(repo_root, args.output_dir, song_name).resolve()
    resolved_timing = _resolve_generation_timing(args, audio_path)
    bpm = float(resolved_timing["bpm"])
    beat_offset_seconds = float(resolved_timing["beat_offset_seconds"])
    beat_offset_source = str(resolved_timing["beat_offset_source"])
    beat_offset_beats = beat_offset_seconds * bpm / 60.0

    logger.info("Loading checkpoint from %s", checkpoint_dir)
    model, config = _load_checkpoint(checkpoint_dir, device)
    if config.encoder_type != "muq_precomputed":
        raise ValueError(
            f"Checkpoint encoder_type={config.encoder_type!r} is not supported by this script."
        )

    muq_model_source = _resolve_muq_model_source(args, config)
    logger.info("Extracting MuQ features from %s", audio_path)
    embedder = MuQEmbedder(
        model_name=muq_model_source,
        device=device.type,
        cache_dir=args.muq_cache_dir,
        local_files_only=args.muq_local_only,
    )
    muq_features, timing = embedder.extract_file_windowed(
        audio_path,
        sample_rate=args.sample_rate,
        window_seconds=args.window_seconds,
        overlap_seconds=args.overlap_seconds,
    )
    beatgrid_features = interpolate_muq_to_beat_grid(
        muq_features,
        bpm=bpm,
        muq_hz=embedder.label_rate,
        beat_offset_seconds=beat_offset_seconds,
    )
    mel = torch.from_numpy(beatgrid_features)
    logger.info(
        "Feature shapes: raw=%s beatgrid=%s audio_seconds=%.2f bpm=%.3f beat_offset_seconds=%.3f source=%s",
        tuple(muq_features.shape),
        tuple(beatgrid_features.shape),
        timing["audio_seconds"],
        bpm,
        beat_offset_seconds,
        beat_offset_source,
    )

    logger.info("Generating notes for %s at %.3f BPM", song_name, bpm)
    notes: list[Note] = []
    used_seed = args.seed
    attempts = max(1, args.candidates, args.retry_empty + 1)
    total_bars = max(1, math.ceil((mel.shape[1] / 16.0) / 4.0))
    candidates: list[dict[str, object]] = []
    best_score = float("-inf")
    shifted_candidate_notes: dict[int, list[Note]] = {}
    for attempt_idx in range(attempts):
        candidate_seed = args.seed + (attempt_idx * args.seed_step)
        candidate_notes = generate_full_song(
            model,
            mel,
            args.difficulty,
            config,
            bpm=bpm,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=candidate_seed,
            inference_mode=args.inference_mode,
        )
        shifted_notes = _shift_notes(candidate_notes, beat_offset_beats, bpm)
        shifted_candidate_notes[candidate_seed] = shifted_notes
        candidate_score, candidate_summary = _score_candidate(
            candidate_notes,
            bpm=bpm,
            total_bars=total_bars,
            difficulty=args.difficulty,
        )
        candidates.append({
            "seed": candidate_seed,
            **candidate_summary,
        })
        logger.info(
            "Generation attempt %d/%d seed=%d notes=%d score=%.3f diversity=%.3f active_bars=%.3f section_std=%.3f silence=%.3f",
            attempt_idx + 1,
            attempts,
            candidate_seed,
            len(candidate_notes),
            candidate_score,
            float(candidate_summary["pattern_diversity"]),
            float(candidate_summary["active_bar_ratio"]),
            float(candidate_summary["section_density_std"]),
            float(candidate_summary["longest_silence_ratio"]),
        )
        if candidate_score > best_score:
            best_score = candidate_score
            notes = shifted_notes
            used_seed = candidate_seed

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.export_all_candidates:
        for candidate in candidates:
            seed = int(candidate["seed"])
            export_notes(
                shifted_candidate_notes[seed],
                bpm=bpm,
                song_name=f"{song_name} [seed {seed}]",
                audio_path=audio_path,
                output_dir=_candidate_output_dir(output_dir, seed),
                difficulty=args.difficulty,
            )
    debug_payload = {
        "selected_seed": used_seed,
        "selected_score": round(best_score, 6),
        "attempts": attempts,
        "bpm": bpm,
        "difficulty": args.difficulty,
        "inference_mode": args.inference_mode,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "export_all_candidates": bool(args.export_all_candidates),
        "winner_candidate_dir": str(_candidate_output_dir(output_dir, used_seed)) if args.export_all_candidates else None,
        "candidates": candidates,
    }
    (output_dir / "generation_candidates.json").write_text(
        json.dumps(debug_payload, indent=2),
        encoding="utf-8",
    )

    if args.export_all_candidates:
        logger.info(
            "Exporting %d notes to %s (seed=%d); all candidates also written under %s",
            len(notes), output_dir, used_seed, output_dir,
        )
    else:
        logger.info("Exporting %d notes to %s (seed=%d)", len(notes), output_dir, used_seed)
    export_notes(
        notes,
        bpm=bpm,
        song_name=song_name,
        audio_path=audio_path,
        output_dir=output_dir,
        difficulty=args.difficulty,
    )
    print(f"Generated map: {output_dir}")


if __name__ == "__main__":
    main()
