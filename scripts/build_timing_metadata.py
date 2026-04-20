from __future__ import annotations

import argparse
import os
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from _light_imports import load_audio_and_timing_modules

_IMPORTS = load_audio_and_timing_modules()
load_manifest = _IMPORTS["load_manifest"]
estimate_song_timing = _IMPORTS["estimate_song_timing"]
load_metadata_dict = _IMPORTS["load_metadata_dict"]
load_timing_metadata = _IMPORTS["load_timing_metadata"]
save_timing_metadata = _IMPORTS["save_timing_metadata"]
timing_metadata_path = _IMPORTS["timing_metadata_path"]


def _default_data_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root.parent / "beat-weaver" / "data"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build timing metadata for corpus songs.")
    data_root = _default_data_root()
    parser.add_argument("--audio-manifest", type=Path, default=data_root / "audio_manifest.json")
    parser.add_argument("--processed-dir", type=Path, default=data_root / "processed")
    parser.add_argument("--timing-metadata", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--max-workers", type=int, default=None)
    return parser


def _estimate_one_song(
    song_hash: str,
    audio_path_str: str,
    bpm_hint: float,
    sample_rate: int,
) -> tuple[str, dict | None, str | None]:
    try:
        entry = estimate_song_timing(
            Path(audio_path_str),
            song_hash=song_hash,
            bpm_hint=bpm_hint if bpm_hint > 0 else None,
            sample_rate=sample_rate,
        )
        return song_hash, entry, None
    except Exception as exc:
        return song_hash, None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    args = build_parser().parse_args()
    audio_manifest_path = Path(args.audio_manifest).resolve()
    processed_dir = Path(args.processed_dir).resolve()
    manifest_root = audio_manifest_path.parent.parent
    metadata = load_metadata_dict(processed_dir)
    timing_path = timing_metadata_path(processed_dir, args.timing_metadata)
    timing_lookup = load_timing_metadata(processed_dir, timing_path=timing_path, metadata=metadata)

    manifest = load_manifest(audio_manifest_path)
    selected = sorted(manifest.items())
    if args.limit is not None:
        selected = selected[: args.limit]

    started_at = time.perf_counter()
    completed = 0
    skipped = 0
    failed = 0
    needs_review = 0
    errors: list[dict[str, str]] = []
    max_workers = args.max_workers or min(os.cpu_count() or 4, 8)

    todo: list[tuple[str, str, float]] = []
    progress = tqdm(selected, desc="Timing metadata", unit="song")
    for song_hash, audio_path_str in selected:
        if args.skip_existing and song_hash in timing_lookup:
            entry = timing_lookup[song_hash]
            if entry.get("timing_source") not in {"missing", "metadata_bpm_only"}:
                skipped += 1
                progress.update(1)
                continue

        audio_path = Path(audio_path_str)
        if not audio_path.is_absolute():
            audio_path = manifest_root / audio_path
        bpm_hint = float(metadata.get(song_hash, {}).get("bpm", 0.0) or 0.0)
        todo.append((song_hash, str(audio_path), bpm_hint))

    progress.set_postfix(completed=completed, skipped=skipped, failed=failed, review=needs_review)
    if todo:
        if max_workers <= 1:
            for song_hash, audio_path, bpm_hint in todo:
                result_hash, entry, error = _estimate_one_song(
                    song_hash,
                    audio_path,
                    bpm_hint,
                    args.sample_rate,
                )
                if entry is not None:
                    timing_lookup[result_hash] = entry
                    completed += 1
                    if entry.get("needs_review"):
                        needs_review += 1
                else:
                    failed += 1
                    errors.append(
                        {
                            "song_hash": song_hash,
                            "audio_path": audio_path,
                            "error": error or "unknown error",
                        }
                    )
                progress.update(1)
                progress.set_postfix(completed=completed, skipped=skipped, failed=failed, review=needs_review)
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_estimate_one_song, song_hash, audio_path, bpm_hint, args.sample_rate): (
                        song_hash,
                        audio_path,
                    )
                    for song_hash, audio_path, bpm_hint in todo
                }
                for future in as_completed(futures):
                    song_hash, audio_path = futures[future]
                    result_hash, entry, error = future.result()
                    if entry is not None:
                        timing_lookup[result_hash] = entry
                        completed += 1
                        if entry.get("needs_review"):
                            needs_review += 1
                    else:
                        failed += 1
                        errors.append(
                            {
                                "song_hash": song_hash,
                                "audio_path": audio_path,
                                "error": error or "unknown error",
                            }
                        )
                    progress.update(1)
                    progress.set_postfix(completed=completed, skipped=skipped, failed=failed, review=needs_review)
    progress.close()

    timing_path.parent.mkdir(parents=True, exist_ok=True)
    save_timing_metadata(timing_path, timing_lookup)
    summary = {
        "num_requested": len(selected),
        "num_completed": completed,
        "num_skipped": skipped,
        "num_failed": failed,
        "num_needs_review": sum(1 for item in timing_lookup.values() if item.get("needs_review")),
        "total_wall_seconds": time.perf_counter() - started_at,
        "timing_metadata": str(timing_path),
        "errors": errors,
    }
    timing_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
