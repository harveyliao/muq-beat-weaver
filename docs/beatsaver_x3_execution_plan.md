# BeatSaver X3 Expansion Plan

## Goal

Create a larger MuQ-precomputed training corpus from `../beat-weaver/data/raw/beatsaver`,
generate `muq_cache` and `muq_cache_beatgrid` for that larger corpus, and train the
current best `base_bs8` configuration on the expanded dataset.

## Why Use a New Processed Root

The current split logic in `muq_beat_weaver/model/dataset.py` regenerates
train/val/test deterministically from the full hash list of the processed corpus.
If the current `processed` directory is expanded in place, the split membership changes.

To keep the current benchmark reproducible while still allowing a new larger split,
the expanded corpus should live in a separate root:

- current corpus: `../beat-weaver/data/processed`
- expanded corpus: `../beat-weaver/data/processed_beatsaver_x3`

## Execution Steps

1. Process the raw BeatSaver corpus into a new processed root:
   - input: `../beat-weaver/data/raw/beatsaver`
   - output: `../beat-weaver/data/processed_beatsaver_x3`

2. Build a matching audio manifest for the same raw BeatSaver corpus:
   - input: `../beat-weaver/data/raw/beatsaver`
   - output: `../beat-weaver/data/processed_beatsaver_x3/audio_manifest.json`

3. Generate raw MuQ embeddings for the expanded corpus:
   - processed dir: `../beat-weaver/data/processed_beatsaver_x3`
   - manifest: `../beat-weaver/data/processed_beatsaver_x3/audio_manifest.json`
   - output cache dir: `../beat-weaver/data/processed_beatsaver_x3/muq_cache`

4. Convert raw MuQ embeddings to beat-grid embeddings:
   - processed dir: `../beat-weaver/data/processed_beatsaver_x3`
   - output cache dir: `../beat-weaver/data/processed_beatsaver_x3/muq_cache_beatgrid`

5. Train the current best configuration unchanged on the expanded corpus:
   - config: `configs/muq_frozen_base_bs8_45ep.json`
   - manifest: `../beat-weaver/data/processed_beatsaver_x3/audio_manifest.json`
   - processed dir: `../beat-weaver/data/processed_beatsaver_x3`
   - output dir: `output/muq_precomputed_beatsaver_x3_base_bs8`

## Commands

### Process raw BeatSaver maps

```powershell
Set-Location E:\github_repos\beat-weaver
.\.venv\Scripts\python.exe -m beat_weaver.cli process `
  --input data\raw\beatsaver `
  --output data\processed_beatsaver_x3
```

### Build the matching audio manifest

```powershell
Set-Location E:\github_repos\beat-weaver
.\.venv\Scripts\python.exe -m beat_weaver.cli build-manifest `
  --input data\raw\beatsaver `
  --output data\processed_beatsaver_x3\audio_manifest.json
```

### Generate raw MuQ cache

```powershell
Set-Location E:\github_repos\muq-beat-weaver
.\.venv\Scripts\python.exe scripts\cache_muq_subset.py `
  --audio-manifest ..\beat-weaver\data\processed_beatsaver_x3\audio_manifest.json `
  --processed-dir ..\beat-weaver\data\processed_beatsaver_x3 `
  --limit 0
```

### Build beat-grid MuQ cache

```powershell
Set-Location E:\github_repos\muq-beat-weaver
.\.venv\Scripts\python.exe scripts\build_muq_beatgrid_cache.py `
  --processed-dir ..\beat-weaver\data\processed_beatsaver_x3
```

### Train on the expanded corpus

```powershell
Set-Location E:\github_repos\muq-beat-weaver
.\.venv\Scripts\python.exe scripts\train_muq_precomputed.py `
  --config configs\muq_frozen_base_bs8_45ep.json `
  --audio-manifest ..\beat-weaver\data\processed_beatsaver_x3\audio_manifest.json `
  --processed-dir ..\beat-weaver\data\processed_beatsaver_x3 `
  --output-dir output\muq_precomputed_beatsaver_x3_base_bs8
```

## Notes

- This defines a new benchmark split from the larger corpus.
- The old `processed` corpus remains intact for continuity.
- If `cache_muq_subset.py --limit 0` does not mean “all songs”, the script should be patched
  before launching the raw MuQ cache step.
