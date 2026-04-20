from __future__ import annotations

import argparse
import os
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from _light_imports import load_audio_and_timing_modules

_IMPORTS = load_audio_and_timing_modules()
interpolate_muq_to_beat_grid = _IMPORTS["interpolate_muq_to_beat_grid"]
load_metadata_dict = _IMPORTS["load_metadata_dict"]
load_timing_metadata = _IMPORTS["load_timing_metadata"]
timing_fingerprint = _IMPORTS["timing_fingerprint"]


def _default_data_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root.parent / "beat-weaver" / "data"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build beat-grid MuQ cache from existing raw MuQ cache.")
    data_root = _default_data_root()
    parser.add_argument("--processed-dir", type=Path, default=data_root / "processed")
    parser.add_argument("--timing-metadata", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--max-workers", type=int, default=None)
    return parser


def _build_one_beatgrid(
    raw_path_str: str,
    out_path_str: str,
    sidecar_path_str: str,
    timing: dict,
    fingerprint: str,
) -> tuple[str, list[int]]:
    raw_path = Path(raw_path_str)
    out_path = Path(out_path_str)
    sidecar_path = Path(sidecar_path_str)
    raw = np.load(raw_path)
    beatgrid = interpolate_muq_to_beat_grid(
        raw,
        float(timing["bpm"]),
        beat_offset_seconds=float(timing.get("first_downbeat_sec", 0.0) or 0.0),
    )
    tmp_path = out_path.with_suffix(".npy.tmp")
    sidecar_tmp_path = sidecar_path.with_suffix(".json.tmp")
    try:
        with open(tmp_path, "wb") as f:
            np.save(f, beatgrid.astype(np.float32, copy=False))
        sidecar_tmp_path.write_text(
            json.dumps(
                {
                    "song_hash": raw_path.stem,
                    "timing_fingerprint": fingerprint,
                    "timing": timing,
                    "beatgrid_shape": list(beatgrid.shape),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp_path.replace(out_path)
        sidecar_tmp_path.replace(sidecar_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        if sidecar_tmp_path.exists():
            sidecar_tmp_path.unlink(missing_ok=True)
    return raw_path.stem, list(beatgrid.shape)


def main() -> None:
    args = build_parser().parse_args()
    processed_dir = Path(args.processed_dir)
    raw_dir = processed_dir / "muq_cache"
    beatgrid_dir = processed_dir / "muq_cache_beatgrid"
    beatgrid_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata_dict(processed_dir)
    timing_lookup = load_timing_metadata(
        processed_dir,
        timing_path=args.timing_metadata,
        metadata=metadata,
    )
    raw_files = sorted(raw_dir.glob("*.npy"))
    if args.limit is not None:
        raw_files = raw_files[: args.limit]

    started_at = time.perf_counter()
    completed = 0
    skipped = 0
    missing_timing = 0
    needs_review = 0
    rebuilt_due_to_timing_change = 0
    max_workers = args.max_workers or min(os.cpu_count() or 4, 8)

    (beatgrid_dir / "VERSION").write_text(
        "muq_beatgrid_cache_v2\nsource=muq_cache\ntiming=timing_metadata.json\ntransform=interpolate_muq_to_beat_grid_with_offset\n",
        encoding="utf-8",
    )

    progress = tqdm(raw_files, desc="MuQ beatgrid", unit="song")
    todo: list[tuple[str, str, str, dict, str]] = []
    for raw_path in progress:
        song_hash = raw_path.stem
        out_path = beatgrid_dir / raw_path.name
        sidecar_path = beatgrid_dir / f"{song_hash}.timing.json"
        timing = timing_lookup.get(song_hash)
        if timing is None or float(timing.get("bpm", 0.0) or 0.0) <= 0:
            missing_timing += 1
            progress.set_postfix(
                completed=completed,
                skipped=skipped,
                missing_timing=missing_timing,
                review=needs_review,
            )
            continue

        fingerprint = timing_fingerprint(timing)
        cached_fingerprint = None
        if sidecar_path.exists():
            cached_fingerprint = json.loads(sidecar_path.read_text(encoding="utf-8")).get("timing_fingerprint")
        if args.skip_existing and out_path.exists() and cached_fingerprint == fingerprint:
            skipped += 1
            progress.set_postfix(
                completed=completed,
                skipped=skipped,
                missing_timing=missing_timing,
                review=needs_review,
            )
            continue
        if out_path.exists() and cached_fingerprint != fingerprint:
            rebuilt_due_to_timing_change += 1
        if timing.get("needs_review"):
            needs_review += 1
        todo.append((str(raw_path), str(out_path), str(sidecar_path), timing, fingerprint))

    if todo:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_build_one_beatgrid, raw_path, out_path, sidecar_path, timing, fingerprint): raw_path
                for raw_path, out_path, sidecar_path, timing, fingerprint in todo
            }
            for future in as_completed(futures):
                future.result()
                completed += 1
                progress.set_postfix(
                    completed=completed,
                    skipped=skipped,
                    missing_timing=missing_timing,
                    review=needs_review,
                )
    progress.close()

    summary = {
        "num_requested": len(raw_files),
        "num_completed": completed,
        "num_skipped": skipped,
        "num_missing_timing": missing_timing,
        "num_needs_review": sum(1 for item in timing_lookup.values() if item.get("needs_review")),
        "num_rebuilt_due_to_timing_change": rebuilt_due_to_timing_change,
        "total_wall_seconds": time.perf_counter() - started_at,
    }
    (beatgrid_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
