# muq-beat-weaver

`muq-beat-weaver` is a research-focused Beat Saber map generation project for CSCI 4052U final project work. The core idea is to use pretrained MuQ audio embeddings as the neural audio representation, then train a transformer decoder to generate Beat Saber note sequences on a fixed beat grid. This repository contains the model-side experimentation, training, inference, timing alignment, and map export code.

This repo is not the full data ingestion pipeline by itself. It works together with a sibling `beat-weaver` repository that prepares raw Beat Saber datasets into the processed corpus used here for training.

For transparency on AI-assisted development in this course project, see [docs/ai-usage.md](docs/ai-usage.md).

## Repository Lineage And Attribution

- MuQ upstream model/code: [tencent-ailab/MuQ](https://github.com/tencent-ailab/MuQ)
- Original Beat Weaver project: [asfilion/beat-weaver](https://github.com/asfilion/beat-weaver)
- Data-preparation fork used for this project: [harveyliao/beat-weaver](https://github.com/harveyliao/beat-weaver)
  - [related section](https://github.com/harveyliao/beat-weaver#prepare-data-here-train-in-muq-beat-weaver)

In this project:

- `MuQ` provides the pretrained audio encoder used to derive music embeddings.
- `harveyliao/beat-weaver` is responsible for corpus preparation, including processing Beat Saber maps and building the audio manifest consumed by this repo.
- `asfilion/beat-weaver` is the original Beat Saber generation codebase that motivated the downstream symbolic generation workflow.

## Problem

The goal is to generate playable Beat Saber note charts directly from music audio.

This is difficult for traditional rule-based approaches because the system must infer:

- rhythmic structure from raw audio
- section-level density and variation
- note placement patterns that remain legal and playable
- alignment between continuous-time audio features and discrete symbolic chart events

The neural-network approach in this repo replaces handcrafted audio features with pretrained MuQ embeddings, then conditions an autoregressive transformer decoder on those embeddings to emit Beat Saber note tokens.

## Neural Network Design

### Audio representation

The audio side uses pretrained MuQ embeddings, typically from `OpenMuQ/MuQ-large-msd-iter`, rather than training an audio encoder from scratch in this repository. In practice, the main workflow uses precomputed MuQ features cached to disk for efficiency.

### Symbolic representation

Beat Saber maps are converted into a normalized schema and then tokenized into a constrained sequence:

- one difficulty token
- bar tokens
- 1/16-note positions on a 4/4 grid
- left-hand note tokens
- right-hand note tokens

The model learns only standard color-note generation. Bombs and obstacles are parsed from source data but are not modeled or exported by the current generator.

### Model architecture

The main model stack is:

1. MuQ embedding input or precomputed MuQ feature cache
2. linear adapter from encoder space to decoder space
3. transformer decoder with causal self-attention and cross-attention over audio memory
4. grammar-constrained autoregressive decoding during inference

This design keeps the pretrained audio representation separate from the Beat Saber sequence model, which makes experimentation cheaper and easier to iterate on.

### How model weights are obtained

- MuQ weights come from the pretrained upstream MuQ project.
- Decoder and adapter weights are trained in this repository using processed Beat Saber map data prepared by the `harveyliao/beat-weaver` fork.
- The main training entry point is `scripts/train_muq_precomputed.py`.

The current best-performing local workflow uses precomputed MuQ embeddings instead of end-to-end finetuning of the MuQ encoder.

## End-To-End Application Pipeline

The full application pipeline is:

1. Parse Beat Saber map folders from multiple schema versions into one normalized representation.
2. Convert note events into a fixed token vocabulary on a 1/16-note beat grid.
3. Build or load MuQ embeddings for each song.
4. Align audio features to beat positions using BPM and first-downbeat timing metadata.
5. Train a transformer decoder to predict token sequences from aligned audio features.
6. During inference, extract MuQ features from a new audio file, align them to the beat grid, generate candidate note charts, rerank them, and export the best candidate as a playable Beat Saber folder.

Application code interfaces with the neural components through the scripts in `scripts/` rather than the package CLI. The most important scripts are:

- `scripts/train_muq_precomputed.py`
- `scripts/generate_from_audio.py`
- `scripts/cache_muq_subset.py`
- `scripts/build_timing_metadata.py`
- `scripts/build_muq_beatgrid_cache.py`

## Software Architecture

Key modules:

- `muq_beat_weaver/schemas`: Beat Saber v2/v3/v4 schema detection and normalization
- `muq_beat_weaver/parsers`: folder and `Info.dat` parsing
- `muq_beat_weaver/model/tokenizer.py`: note-token encoding and decoding
- `muq_beat_weaver/model/dataset.py`: processed dataset loading, filtering, splitting, and caching
- `muq_beat_weaver/model/timing.py`: BPM/downbeat estimation and beat-grid rebasing
- `muq_beat_weaver/model/transformer.py`: model assembly
- `muq_beat_weaver/model/training.py`: training loop
- `muq_beat_weaver/model/inference.py`: constrained decoding and long-song generation
- `muq_beat_weaver/model/exporter.py`: playable Beat Saber map export

One important architectural decision is that the same timing metadata is reused in preprocessing, training, and inference. That reduces train/inference mismatch when mapping MuQ frame time to Beat Saber beat positions.

## Setup

### Requirements

- Python 3.11+
- `uv` for environment management
- a local GPU is recommended for training
- a processed dataset generated by the sibling `beat-weaver` fork

Install dependencies from the repo root:

```powershell
uv sync
```

### Pretrained output weights

If you want to run inference with the trained checkpoints used in this project, download the pretrained `output.zip` archive into the `output/` folder at the project root:

- Google Drive: [output.zip](https://drive.google.com/file/d/12PRc-fVwcnO1A5U5kR6vfw-jmlQM4am4/view?usp=drive_link)
- Access note: shared with anyone who has the link and in the Ontario Tech University organization

Instructions:

1. Download `output.zip` into `output/` under the project root.
2. Unzip `output.zip` inside `output/`.
3. After extraction, you should see these 3 items in `output/`:
   - `muq_precomputed_2000_45ep_base_bs8/`
   - `muq_precomputed_beatsaver_x3_base_bs8/`
   - `audit_training_targets.json`

Optional developer checks:

```powershell
uv run pytest
```

### Timing environment note

The timing pipeline supports a `madmom`-first path, but local experiments showed that `madmom` is more reliable in a separate Python 3.11 environment than in the main training environment. See [docs/madmom_timing_experiment.md](docs/madmom_timing_experiment.md) for details.

## Data Preparation

This repository assumes that raw Beat Saber content has already been processed by the `harveyliao/beat-weaver` fork.

- [related section](https://github.com/harveyliao/beat-weaver#prepare-data-here-train-in-muq-beat-weaver)

Typical division of responsibility:

- `harveyliao/beat-weaver`: process raw Beat Saber maps and build `audio_manifest.json`
- `muq-beat-weaver`: cache MuQ embeddings, build beat-grid-aligned features, train, and run inference

Example processed-data locations expected by default:

- `../beat-weaver/data/processed`
- `../beat-weaver/data/audio_manifest.json`

For the expanded BeatSaver experiment, see [docs/beatsaver_x3_execution_plan.md](docs/beatsaver_x3_execution_plan.md).

## Training

Main training command:

```powershell
uv run python scripts\train_muq_precomputed.py `
  --config configs\muq_frozen_base_bs8_45ep.json `
  --audio-manifest ..\beat-weaver\data\audio_manifest.json `
  --processed-dir ..\beat-weaver\data\processed `
  --output-dir output\muq_precomputed_run
```

If raw MuQ embeddings are not cached yet:

```powershell
uv run python scripts\cache_muq_subset.py `
  --audio-manifest ..\beat-weaver\data\audio_manifest.json `
  --processed-dir ..\beat-weaver\data\processed `
  --limit 0
```

If timing-aware beat-grid features need to be rebuilt:

```powershell
uv run python scripts\build_muq_beatgrid_cache.py `
  --processed-dir ..\beat-weaver\data\processed
```

## Inference

Generate a Beat Saber chart from a single song:

```powershell
uv run python scripts\generate_from_audio.py `
  --checkpoint output\muq_precomputed_beatsaver_x3_base_bs8\checkpoints\best `
  --muq-local-only `
  --audio "data\inference\RAYE,070 Shake - Escapism.wav" `
  --bpm 96 `
  --seed 4052 `
  --candidates 4 `
  --export-all-candidates `
  --inference-mode rolling `
  --song-name "RAYE,070 Shake - Escapism" `
  --difficulty Expert `
  --output-dir "C:\Users\Harve\BSManager\BSInstances\1.40.8\Beat Saber_Data\CustomWIPLevels\RAYE,070 Shake - Escapism" `
  --temperature 1.1 `
  --top-p 0.9 `
  --top-k 50
```

The inference script:

- extracts MuQ embeddings from the input song
- aligns embeddings to the beat grid
- generates multiple candidate maps
- reranks them with heuristics
- exports the best candidate as a playable Beat Saber level folder

## Results And Current Limitations

Based on the local research notes in `research/research.md`, the project currently shows:

- a workable training and inference pipeline using MuQ-precomputed features
- improved validation loss with larger processed corpora
- timing-aware preprocessing that better matches training and inference alignment
- grammar-constrained decoding that guarantees structurally valid token sequences

Current limitations:

- the generator only models color notes, not bombs or obstacles
- the representation is fixed to a 4/4, 1/16-note grid
- output quality is still inconsistent and can produce repetitive or sparse charts
- the package CLI is minimal; the real workflow is script-driven

## Demo Media

- [Video demo](https://drive.google.com/file/d/1USi5dRjUZ6W1N_6yTpE5NGEMMfjElXZE/view?usp=drive_link)

## References

- MuQ repository: [https://github.com/tencent-ailab/MuQ](https://github.com/tencent-ailab/MuQ)
- Original Beat Weaver repository: [https://github.com/asfilion/beat-weaver](https://github.com/asfilion/beat-weaver)
- Data-prep fork used by this project: [https://github.com/harveyliao/beat-weaver](https://github.com/harveyliao/beat-weaver)
- Local project analysis: [research/research.md](research/research.md)
- Course project requirements: [research/CSCI4052/Final Project.md](research/CSCI4052/Final%20Project.md)
