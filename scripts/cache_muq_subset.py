from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from muq_beat_weaver.model.audio import load_manifest
from muq_beat_weaver.model.muq_embeddings import (
    MuQEmbedder,
    _muq_cache_version_key,
    summarize_embedding,
)


def _default_data_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root.parent / "beat-weaver" / "data"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache MuQ embeddings for a subset of manifest songs.")
    data_root = _default_data_root()
    parser.add_argument("--audio-manifest", type=Path, default=data_root / "audio_manifest.json")
    parser.add_argument("--processed-dir", type=Path, default=data_root / "processed")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--model-name", type=str, default="OpenMuQ/MuQ-large-msd-iter")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--window-seconds", type=float, default=180.0)
    parser.add_argument("--overlap-seconds", type=float, default=30.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audio_manifest_path = Path(args.audio_manifest).resolve()
    processed_dir = Path(args.processed_dir).resolve()
    manifest_root = audio_manifest_path.parent.parent
    output_dir = processed_dir / "muq_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(audio_manifest_path)
    selected = sorted(manifest.items())
    if args.limit > 0:
        selected = selected[: args.limit]

    embedder = MuQEmbedder(model_name=args.model_name, device=args.device)
    version = _muq_cache_version_key(
        model_name=args.model_name,
        label_rate=embedder.label_rate,
        sample_rate=args.sample_rate,
        window_seconds=args.window_seconds,
        overlap_seconds=args.overlap_seconds,
    )
    (output_dir / "VERSION").write_text(version, encoding="utf-8")

    items: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    started_at = time.perf_counter()
    completed = 0
    skipped = 0
    failed = 0

    progress = tqdm(selected, desc="MuQ cache", unit="song")
    for song_hash, audio_path_str in progress:
        audio_path = Path(audio_path_str)
        if not audio_path.is_absolute():
            audio_path = manifest_root / audio_path
        embedding_path = output_dir / f"{song_hash}.npy"
        stats_path = output_dir / f"{song_hash}.json"
        if args.skip_existing and embedding_path.exists():
            skipped += 1
            progress.set_postfix(completed=completed, skipped=skipped, failed=failed)
            continue

        try:
            embedding, timing = embedder.extract_file_windowed(
                audio_path,
                sample_rate=args.sample_rate,
                window_seconds=args.window_seconds,
                overlap_seconds=args.overlap_seconds,
            )
        except Exception as exc:
            failed += 1
            errors.append(
                {
                    "song_hash": song_hash,
                    "audio_path": str(audio_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            progress.write(f"[skip] {song_hash}: {type(exc).__name__}: {exc}")
            progress.set_postfix(completed=completed, skipped=skipped, failed=failed)
            continue

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
        items.append(asdict(stats))
        completed += 1
        progress.set_postfix(completed=completed, skipped=skipped, failed=failed)

    summary = {
        "model_name": args.model_name,
        "device": str(embedder.device),
        "num_requested": len(selected),
        "num_completed": completed,
        "num_skipped": skipped,
        "num_failed": failed,
        "total_wall_seconds": time.perf_counter() - started_at,
        "items": items,
        "errors": errors,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("num_requested", "num_completed", "num_skipped", "num_failed", "total_wall_seconds")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
