# madmom Timing Experiment

## Goal

Evaluate whether `madmom` should be used during dataset prep to improve:

- BPM estimation
- first downbeat detection
- beat-grid alignment for `muq_cache_beatgrid`
- consistency between training-time note timing and inference-time export timing

The design target was:

- keep raw `muq_cache` unchanged
- rebuild only timing-dependent artifacts
- use one timing source of truth for preprocessing, training, and inference

## Code Changes

The experiment landed in these areas:

- `muq_beat_weaver/model/timing.py`
  - timing metadata schema
  - `madmom`-first timing extraction
  - `librosa` fallback
  - note rebasing onto a downbeat-relative beat grid
- `muq_beat_weaver/model/audio.py`
  - offset-aware beat-grid interpolation via `beat_offset_seconds`
- `muq_beat_weaver/model/dataset.py`
  - training samples now consume timing metadata
  - note beats are rebased to the same downbeat-relative grid as the beat-grid cache
  - pre-downbeat pickup notes are dropped and counted for review
- `scripts/build_timing_metadata.py`
  - corpus timing extraction
  - review flags
  - `--max-workers`
- `scripts/build_muq_beatgrid_cache.py`
  - rebuilds beat-grid cache from raw MuQ + timing metadata
  - writes per-song timing sidecars
  - skips unchanged songs using timing fingerprints
  - `--max-workers`
- `scripts/generate_from_audio.py`
  - can read timing metadata directly
  - keeps manual overrides
  - uses the same offset-aware interpolation path
- `scripts/_light_imports.py`
  - allows timing scripts to run in a lightweight Python 3.11 environment without pulling in the full training stack

## Environment Findings

### What did not work

Running `madmom` from the repo venv did not work:

- repo venv Python version: `3.13`
- `madmom` not installed there
- even if installed, `madmom 0.16.x` is fragile on newer Python / NumPy combinations

The first timing metadata build produced:

- `1000` processed songs
- `1000` `needs_review`
- timing sources:
  - `60362` `metadata_bpm_only`
  - `1000` `librosa_first_beat`

That showed the run never used `madmom`.

### What did work

A dedicated Python 3.11 timing environment worked:

- env location used during the experiment:
  - `E:\github_repos\madmom\.venv311_timing`
- confirmed:
  - `madmom` importable
  - `librosa`, `soundfile`, and `tqdm` available

Because the timing env intentionally omitted `torch`, the timing scripts needed a lightweight import bootstrap to avoid importing the training stack. That is why `scripts/_light_imports.py` exists.

### Windows-specific issue

On this machine, some multiprocessing and tempdir operations were blocked by filesystem / sandbox permissions:

- Windows multiprocessing pipes in the 3.11 timing env could fail
- writing outside the repo sandbox was blocked for the agent
- pytest tempdir cleanup sometimes failed in sandboxed runs

Because of that, the safest validation path was:

1. run timing extraction serially with `--max-workers 1`
2. confirm `madmom` is actually being used
3. scale up worker count after that

## Probe Result

A 10-song serial probe with the 3.11 timing env succeeded:

- `num_requested`: `10`
- `num_completed`: `10`
- `num_failed`: `0`
- `num_needs_review`: `2`

The probe timing-source distribution for processed songs showed `madmom` was actually active.

This was the key confirmation that the experiment worked technically.

## Recommended Workflow

### 1. Build timing metadata with Python 3.11

Run from the repo root:

```powershell
cd E:\github_repos\muq-beat-weaver
..\madmom\.venv311_timing\Scripts\python.exe scripts\build_timing_metadata.py `
  --processed-dir ..\beat-weaver\data\processed `
  --audio-manifest ..\beat-weaver\data\audio_manifest.json `
  --no-skip-existing `
  --max-workers 1
```

Start with `--max-workers 1` until the timing source distribution looks healthy.

Then inspect the result:

```powershell
@'
import json
from collections import Counter
p = r"E:\github_repos\beat-weaver\data\processed\timing_metadata.json"
data = json.load(open(p, encoding="utf-8"))
print(Counter(v["timing_source"] for v in data.values()))
print("needs_review", sum(1 for v in data.values() if v.get("needs_review")))
'@ | .\.venv\Scripts\python.exe -
```

Once `madmom` is showing up at scale, increase workers:

```powershell
..\madmom\.venv311_timing\Scripts\python.exe scripts\build_timing_metadata.py `
  --processed-dir ..\beat-weaver\data\processed `
  --audio-manifest ..\beat-weaver\data\audio_manifest.json `
  --no-skip-existing `
  --max-workers 8
```

### 2. Rebuild beat-grid cache

After timing metadata is good:

```powershell
.\.venv\Scripts\python.exe scripts\build_muq_beatgrid_cache.py `
  --processed-dir ..\beat-weaver\data\processed `
  --no-skip-existing `
  --max-workers 12
```

This step uses:

- raw `muq_cache`
- `timing_metadata.json`

and writes:

- `muq_cache_beatgrid`
- per-song timing sidecars
- summary output

### 3. Train as usual

No special training command changes are required.

The dataset path now automatically uses:

- timing-aware beat-grid cache when present
- timing metadata to rebase note beats onto the same downbeat-relative grid

## Practical Lessons

- `madmom` belongs in a dedicated timing environment, not the main repo venv.
- Raw MuQ embeddings should stay immutable.
- Timing-aware artifacts should be rebuilt independently.
- Serial validation is worth doing before parallelizing corpus-scale timing extraction.
- If a run reports nearly all songs as `needs_review`, the first thing to check is the timing source distribution.

## Status

At the end of the experiment:

- timing metadata support: implemented
- `madmom` timing path: implemented and validated on a serial probe
- beat-grid cache rebuild path: implemented
- training/inference timing unification: implemented
- test suite: passed
