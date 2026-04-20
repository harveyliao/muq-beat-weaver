"""Post-training generation quality evaluation on validation samples."""

from __future__ import annotations

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


def _select_eval_indices(total_items: int, limit: int) -> list[int]:
    """Select a deterministic spread of indices across the validation set."""
    if total_items <= 0 or limit <= 0:
        return []
    if limit >= total_items:
        return list(range(total_items))
    if limit == 1:
        return [0]
    indices = {
        round(i * (total_items - 1) / (limit - 1))
        for i in range(limit)
    }
    return sorted(indices)


def _aggregate_metrics(
    per_sample: list[dict[str, float | int | str]],
) -> dict[str, dict[str, float]]:
    if not per_sample:
        return {}
    metric_names = sorted(
        key for key in per_sample[0]
        if isinstance(per_sample[0][key], (int, float))
        and key not in {"sample_index", "reference_note_count", "generated_note_count"}
    )
    aggregated: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = [float(sample[name]) for sample in per_sample]
        aggregated[name] = {
            "mean": round(statistics.fmean(values), 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
        }
    generated_counts = [int(sample["generated_note_count"]) for sample in per_sample]
    reference_counts = [int(sample["reference_note_count"]) for sample in per_sample]
    aggregated["generated_note_count"] = {
        "mean": round(statistics.fmean(generated_counts), 3),
        "min": min(generated_counts),
        "max": max(generated_counts),
    }
    aggregated["reference_note_count"] = {
        "mean": round(statistics.fmean(reference_counts), 3),
        "min": min(reference_counts),
        "max": max(reference_counts),
    }
    return aggregated


def evaluate_generation_checkpoint(
    checkpoint_dir: Path,
    config: ModelConfig,
    val_dataset: BeatSaberDataset,
    output_dir: Path,
    *,
    device: torch.device | None = None,
) -> Path | None:
    """Run generation-time quality evaluation and write a summary JSON."""
    eval_limit = min(config.generation_eval_samples, len(val_dataset))
    if eval_limit <= 0:
        logger.info("Skipping generation evaluation because generation_eval_samples <= 0")
        return None

    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    eval_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BeatWeaverModel(config)
    model.load_state_dict(
        torch.load(checkpoint_dir / "model.pt", map_location=eval_device, weights_only=True),
    )
    model.to(eval_device)
    model.eval()

    per_sample: list[dict[str, float | int | str]] = []
    selected_indices = _select_eval_indices(len(val_dataset), eval_limit)
    logger.info("Running generation evaluation on %d validation samples", len(selected_indices))

    for ordinal, sample_idx in enumerate(selected_indices, start=1):
        sample = val_dataset.samples[sample_idx]
        reference_notes = val_dataset.reference_notes(sample_idx)
        mel = torch.from_numpy(val_dataset.load_full_features(sample_idx))
        generated_notes = generate_full_song(
            model,
            mel,
            sample["difficulty"],
            config,
            bpm=sample["bpm"],
            temperature=config.generation_eval_temperature,
            top_k=config.generation_eval_top_k,
            top_p=config.generation_eval_top_p,
            seed=config.generation_eval_seed + sample_idx,
        )
        metrics = evaluate_map(generated_notes, reference_notes, bpm=sample["bpm"])
        per_sample.append({
            "sample_index": sample_idx,
            "song_hash": sample["song_hash"],
            "difficulty": sample["difficulty"],
            "characteristic": sample["characteristic"],
            "reference_note_count": len(reference_notes),
            "generated_note_count": len(generated_notes),
            **{name: round(float(value), 6) for name, value in metrics.items()},
        })
        logger.info(
            "Generation eval %d/%d: %s %s onset_f1=%.4f nps_acc=%.4f parity=%.4f",
            ordinal,
            len(selected_indices),
            sample["song_hash"],
            sample["difficulty"],
            metrics["onset_f1"],
            metrics["nps_accuracy"],
            metrics["parity_violation_rate"],
        )

    summary = {
        "checkpoint": str(checkpoint_dir),
        "samples_evaluated": len(per_sample),
        "selection": {
            "requested_samples": config.generation_eval_samples,
            "evaluated_indices": selected_indices,
            "temperature": config.generation_eval_temperature,
            "top_k": config.generation_eval_top_k,
            "top_p": config.generation_eval_top_p,
            "seed_base": config.generation_eval_seed,
        },
        "aggregate": _aggregate_metrics(per_sample),
        "per_sample": per_sample,
    }
    summary_path = output_dir / "generation_eval.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path
