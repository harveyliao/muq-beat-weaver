from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd

from muq_beat_weaver.model.config import ModelConfig
from muq_beat_weaver.model.dataset import prepare_dataset_corpus
from muq_beat_weaver.model.parquet import read_notes_parquet
from muq_beat_weaver.model.tokenizer import BAR, POS_BASE, POS_COUNT, difficulty_to_token, encode_beatmap
from muq_beat_weaver.schemas.normalized import DifficultyInfo, Note, NormalizedBeatmap, SongMetadata


def _default_data_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root.parent / "beat-weaver" / "data"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit whether corpus preprocessing creates empty or degenerate token targets.",
    )
    repo_root = Path(__file__).resolve().parents[1]
    data_root = _default_data_root()
    parser.add_argument("--config", type=Path, default=repo_root / "configs" / "muq_frozen_base_bs8_45ep.json")
    parser.add_argument("--audio-manifest", type=Path, default=data_root / "audio_manifest.json")
    parser.add_argument("--processed-dir", type=Path, default=data_root / "processed")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the audit summary as JSON.",
    )
    return parser


def _resolve_arg(path: Path, launch_cwd: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (launch_cwd / path).resolve()


def _make_beatmap(
    *,
    difficulty: str,
    characteristic: str,
    bpm: float,
    song_hash: str,
    notes: list[dict],
) -> NormalizedBeatmap:
    normalized_notes = [
        Note(
            beat=float(note["beat"]),
            time_seconds=float(note.get("time_seconds", note["beat"] * 60.0 / bpm)),
            x=int(note["x"]),
            y=int(note["y"]),
            color=int(note["color"]),
            cut_direction=int(note["cut_direction"]),
            angle_offset=int(note.get("angle_offset", 0)),
        )
        for note in notes
    ]
    return NormalizedBeatmap(
        metadata=SongMetadata(source="audit", source_id=song_hash, hash=song_hash, bpm=float(bpm)),
        difficulty_info=DifficultyInfo(
            characteristic=characteristic,
            difficulty=difficulty,
            difficulty_rank=0,
            note_jump_speed=0.0,
            note_jump_offset=0.0,
        ),
        notes=normalized_notes,
    )


def _leading_empty_bars(tokens: list[int]) -> int:
    bar_count = 0
    for token in tokens[2:]:
        if token == BAR:
            bar_count += 1
            continue
        if POS_BASE <= token < POS_BASE + POS_COUNT:
            return max(0, bar_count - 1)
        return bar_count
    return bar_count


def _summarize_lengths(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0, "max": 0}
    return {
        "count": len(lengths),
        "mean": round(statistics.fmean(lengths), 3),
        "median": float(statistics.median(lengths)),
        "min": min(lengths),
        "max": max(lengths),
    }


def main() -> None:
    args = build_parser().parse_args()
    launch_cwd = Path.cwd()
    config_path = _resolve_arg(args.config, launch_cwd)
    audio_manifest_path = _resolve_arg(args.audio_manifest, launch_cwd)
    processed_dir = _resolve_arg(args.processed_dir, launch_cwd)
    output_json = _resolve_arg(args.output_json, launch_cwd) if args.output_json is not None else None

    config = ModelConfig.load(config_path)
    config.encoder_type = "muq_precomputed"

    table = read_notes_parquet(processed_dir)
    df = table.to_pandas()
    if "angle_offset" not in df.columns:
        df["angle_offset"] = 0

    grouped = df.groupby(["song_hash", "difficulty", "characteristic"])
    raw_group_count = int(grouped.ngroups)
    raw_empty_encoded = 0
    raw_invalid_only = 0
    raw_unknown_difficulty = 0
    raw_valid_note_counts: list[int] = []
    raw_token_lengths: list[int] = []
    raw_leading_empty_bars: list[int] = []

    note_cols = ["beat", "time_seconds", "x", "y", "color", "cut_direction", "angle_offset", "bpm"]

    for (song_hash, difficulty, characteristic), group in grouped:
        try:
            difficulty_to_token(str(difficulty))
        except ValueError:
            raw_unknown_difficulty += 1
            continue
        note_dicts = group[note_cols].to_dict("records")
        bpm = float(note_dicts[0]["bpm"])
        valid_notes = [
            note for note in note_dicts
            if 0 <= int(note["x"]) < 4
            and 0 <= int(note["y"]) < 3
            and 0 <= int(note["cut_direction"]) < 9
            and int(note["color"]) in (0, 1)
        ]
        if not valid_notes:
            raw_invalid_only += 1
        beatmap = _make_beatmap(
            difficulty=difficulty,
            characteristic=characteristic,
            bpm=bpm,
            song_hash=song_hash,
            notes=valid_notes,
        )
        tokens = encode_beatmap(beatmap)
        if len(tokens) == 3:
            raw_empty_encoded += 1
        raw_valid_note_counts.append(len(valid_notes))
        raw_token_lengths.append(len(tokens))
        raw_leading_empty_bars.append(_leading_empty_bars(tokens))

    prepared = prepare_dataset_corpus(
        processed_dir,
        audio_manifest_path,
        config,
        include_splits=("train", "val", "test"),
    )
    training_samples = list(prepared.samples)
    training_empty_encoded = sum(1 for sample in training_samples if int(sample["full_token_count"]) == 3)
    training_token_lengths = [int(sample["full_token_count"]) for sample in training_samples]
    training_note_counts = [len(sample["notes"]) for sample in training_samples]
    training_leading_empty_bars = [
        _leading_empty_bars(sample["token_ids"][: sample["full_token_count"]])
        for sample in training_samples
    ]

    summary = {
        "config": str(config_path),
        "processed_dir": str(processed_dir),
        "audio_manifest": str(audio_manifest_path),
        "raw_grouped_maps": raw_group_count,
        "raw_valid_note_count": _summarize_lengths(raw_valid_note_counts),
        "raw_token_length": _summarize_lengths(raw_token_lengths),
        "raw_empty_encoded_maps": raw_empty_encoded,
        "raw_empty_encoded_ratio": round(raw_empty_encoded / max(1, raw_group_count), 6),
        "raw_invalid_only_maps": raw_invalid_only,
        "raw_invalid_only_ratio": round(raw_invalid_only / max(1, raw_group_count), 6),
        "raw_unknown_difficulty_maps": raw_unknown_difficulty,
        "raw_unknown_difficulty_ratio": round(raw_unknown_difficulty / max(1, raw_group_count), 6),
        "raw_maps_with_leading_empty_bar": sum(1 for x in raw_leading_empty_bars if x >= 1),
        "raw_maps_with_4plus_leading_empty_bars": sum(1 for x in raw_leading_empty_bars if x >= 4),
        "raw_leading_empty_bars": _summarize_lengths(raw_leading_empty_bars),
        "training_ready_samples": len(training_samples),
        "training_note_count": _summarize_lengths(training_note_counts),
        "training_token_length": _summarize_lengths(training_token_lengths),
        "training_empty_encoded_samples": training_empty_encoded,
        "training_empty_encoded_ratio": round(training_empty_encoded / max(1, len(training_samples)), 6),
        "training_samples_with_leading_empty_bar": sum(1 for x in training_leading_empty_bars if x >= 1),
        "training_samples_with_4plus_leading_empty_bars": sum(1 for x in training_leading_empty_bars if x >= 4),
        "training_leading_empty_bars": _summarize_lengths(training_leading_empty_bars),
        "split_sizes": {name: len(song_hashes) for name, song_hashes in prepared.split_hashes.items()},
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
