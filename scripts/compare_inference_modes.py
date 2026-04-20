from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path

import torch

from muq_beat_weaver.model.config import ModelConfig
from muq_beat_weaver.model.dataset import BeatSaberDataset
from muq_beat_weaver.model.evaluate import evaluate_map
from muq_beat_weaver.model.inference import generate_full_song
from muq_beat_weaver.model.transformer import BeatWeaverModel

logger = logging.getLogger(__name__)


def _spread_indices(indices: list[int], limit: int) -> list[int]:
    if limit <= 0 or len(indices) <= limit:
        return indices
    if limit == 1:
        return [indices[0]]
    selected_positions = {
        round(i * (len(indices) - 1) / (limit - 1))
        for i in range(limit)
    }
    return [indices[pos] for pos in sorted(selected_positions)]


def _aggregate(rows: list[dict[str, float | int | str]]) -> dict[str, dict[str, float | int]]:
    if not rows:
        return {}
    metric_names = [
        name for name, value in rows[0].items()
        if isinstance(value, (int, float)) and name not in {"sample_index"}
    ]
    summary: dict[str, dict[str, float | int]] = {}
    for name in metric_names:
        values = [float(row[name]) for row in rows]
        summary[name] = {
            "mean": round(statistics.fmean(values), 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
        }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare short single-window generation against long-song inference modes.",
    )
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "output" / "muq_precomputed_2000_45ep_base_bs8" / "checkpoints" / "best",
        help="Checkpoint directory containing model.pt and config.json.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
        help="Processed dataset directory used to build the validation split.",
    )
    parser.add_argument(
        "--audio-manifest",
        type=Path,
        required=True,
        help="Audio manifest JSON used by the dataset.",
    )
    parser.add_argument(
        "--per-group-limit",
        type=int,
        default=8,
        help="Maximum number of validation samples to evaluate per group.",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        choices=["Easy", "Normal", "Hard", "Expert", "ExpertPlus"],
        help="Optional difficulty filter applied after the validation split is loaded.",
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
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling filter.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling filter.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=1234,
        help="Base seed for deterministic evaluation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "output" / "inference_mode_comparison.json",
        help="Where to write the comparison JSON.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def _load_model(checkpoint_dir: Path, device: torch.device) -> tuple[BeatWeaverModel, ModelConfig]:
    config = ModelConfig.load(checkpoint_dir / "config.json")
    model = BeatWeaverModel(config)
    model.load_state_dict(
        torch.load(checkpoint_dir / "model.pt", map_location=device, weights_only=True),
    )
    model.to(device)
    model.eval()
    return model, config


def _group_indices(dataset: BeatSaberDataset, difficulty: str | None, max_audio_len: int) -> tuple[list[int], list[int]]:
    short_indices: list[int] = []
    long_indices: list[int] = []
    for idx, sample in enumerate(dataset.samples):
        if difficulty is not None and sample["difficulty"] != difficulty:
            continue
        total_frames = dataset.load_full_features(idx).shape[1]
        if total_frames <= max_audio_len:
            short_indices.append(idx)
        else:
            long_indices.append(idx)
    return short_indices, long_indices


def _evaluate_group(
    *,
    dataset: BeatSaberDataset,
    model: BeatWeaverModel,
    config: ModelConfig,
    indices: list[int],
    inference_mode: str,
    temperature: float,
    top_k: int,
    top_p: float,
    seed_base: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for ordinal, sample_idx in enumerate(indices):
        sample = dataset.samples[sample_idx]
        mel = torch.from_numpy(dataset.load_full_features(sample_idx))
        reference_notes = dataset.reference_notes(sample_idx)
        generated_notes = generate_full_song(
            model,
            mel,
            sample["difficulty"],
            config,
            bpm=sample["bpm"],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed_base + sample_idx,
            inference_mode=inference_mode,
        )
        metrics = evaluate_map(generated_notes, reference_notes, bpm=sample["bpm"])
        rows.append({
            "sample_index": sample_idx,
            "song_hash": sample["song_hash"],
            "difficulty": sample["difficulty"],
            "total_frames": int(mel.shape[1]),
            "reference_note_count": len(reference_notes),
            "generated_note_count": len(generated_notes),
            **{name: round(float(value), 6) for name, value in metrics.items()},
        })
        logger.info(
            "%s %d/%d sample=%d difficulty=%s generated=%d onset_f1=%.4f pattern_diversity=%.4f",
            inference_mode,
            ordinal + 1,
            len(indices),
            sample_idx,
            sample["difficulty"],
            len(generated_notes),
            metrics["onset_f1"],
            metrics["pattern_diversity"],
        )
    return rows


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    checkpoint_dir = args.checkpoint.resolve()
    model, config = _load_model(checkpoint_dir, device)
    dataset = BeatSaberDataset(
        processed_dir=args.processed_dir.resolve(),
        audio_manifest_path=args.audio_manifest.resolve(),
        config=config,
        split="val",
    )

    short_indices, long_indices = _group_indices(dataset, args.difficulty, config.max_audio_len)
    short_eval = _spread_indices(short_indices, args.per_group_limit)
    long_eval = _spread_indices(long_indices, args.per_group_limit)

    results: dict[str, object] = {
        "checkpoint": str(checkpoint_dir),
        "config": {
            "max_audio_len": config.max_audio_len,
            "max_seq_len": config.max_seq_len,
        },
        "selection": {
            "difficulty": args.difficulty,
            "per_group_limit": args.per_group_limit,
            "short_available": len(short_indices),
            "long_available": len(long_indices),
            "short_evaluated": short_eval,
            "long_evaluated": long_eval,
        },
        "groups": {},
    }

    short_rows = _evaluate_group(
        dataset=dataset,
        model=model,
        config=config,
        indices=short_eval,
        inference_mode="independent",
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed_base=args.seed_base,
    )
    long_independent_rows = _evaluate_group(
        dataset=dataset,
        model=model,
        config=config,
        indices=long_eval,
        inference_mode="independent",
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed_base=args.seed_base,
    )
    long_rolling_rows = _evaluate_group(
        dataset=dataset,
        model=model,
        config=config,
        indices=long_eval,
        inference_mode="rolling",
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed_base=args.seed_base,
    )

    results["groups"] = {
        "short_single_window": {
            "inference_mode": "independent",
            "aggregate": _aggregate(short_rows),
            "per_sample": short_rows,
        },
        "long_independent": {
            "inference_mode": "independent",
            "aggregate": _aggregate(long_independent_rows),
            "per_sample": long_independent_rows,
        },
        "long_rolling": {
            "inference_mode": "rolling",
            "aggregate": _aggregate(long_rolling_rows),
            "per_sample": long_rolling_rows,
        },
    }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote comparison to {output_path}")


if __name__ == "__main__":
    main()
