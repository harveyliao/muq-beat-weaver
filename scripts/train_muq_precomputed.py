from __future__ import annotations

import argparse
import os
from pathlib import Path

from muq_beat_weaver.model.config import ModelConfig
from muq_beat_weaver.model.dataset import build_train_val_datasets
from muq_beat_weaver.model.training import train


def _default_data_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root.parent / "beat-weaver" / "data"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a MuQ-precomputed model.")
    repo_root = Path(__file__).resolve().parents[1]
    data_root = _default_data_root()
    parser.add_argument("--config", type=Path, default=repo_root / "configs" / "muq_frozen_small.json")
    parser.add_argument("--audio-manifest", type=Path, default=data_root / "audio_manifest.json")
    parser.add_argument("--processed-dir", type=Path, default=data_root / "processed")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "output" / "muq_precomputed_2000_15ep")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    launch_cwd = Path.cwd()
    config_path = Path(args.config)
    audio_manifest_path = Path(args.audio_manifest)
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    resume_from = Path(args.resume_from) if args.resume_from is not None else None

    if not config_path.is_absolute():
        config_path = (launch_cwd / config_path).resolve()
    if not audio_manifest_path.is_absolute():
        audio_manifest_path = (launch_cwd / audio_manifest_path).resolve()
    if not processed_dir.is_absolute():
        processed_dir = (launch_cwd / processed_dir).resolve()
    if not output_dir.is_absolute():
        output_dir = (launch_cwd / output_dir).resolve()
    if resume_from is not None and not resume_from.is_absolute():
        resume_from = (launch_cwd / resume_from).resolve()

    os.chdir(audio_manifest_path.parent.parent)
    config = ModelConfig.load(config_path)
    config.encoder_type = "muq_precomputed"
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    config.muq_model_name = "OpenMuQ/MuQ-large-msd-iter"

    train_dataset, val_dataset = build_train_val_datasets(
        processed_dir,
        audio_manifest_path,
        config,
    )
    print(f"train_samples={len(train_dataset)} val_samples={len(val_dataset)}")
    best = train(
        config,
        train_dataset,
        val_dataset,
        output_dir,
        resume_from=resume_from,
    )
    print(f"best_checkpoint={best}")


if __name__ == "__main__":
    main()
