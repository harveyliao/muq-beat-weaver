# muq-beat-weaver Deep Repository Report

## 1. Executive Overview

`muq-beat-weaver` is a focused research repo for Beat Saber map generation where the main experiment is replacing or isolating the audio-side representation with MuQ embeddings while keeping the downstream symbolic representation, decoder, map export, and evaluation logic under local control.

The repo is not a full productized application. It is best understood as a model experimentation workspace with:

- a normalized Beat Saber parsing layer
- a compact token vocabulary for map notes
- a transformer decoder conditioned on audio features
- a training path built primarily around precomputed MuQ embeddings
- an inference/export path that turns generated notes back into playable v2 Beat Saber folders
- timing-alignment experiments intended to make training-time note coordinates and inference-time feature alignment use the same downbeat-relative grid

The package CLI is effectively a stub. The real operational surface is in `scripts/`, not `muq_beat_weaver.cli`.

## 2. What The Repo Actually Does

At a high level, the system does this:

1. Parse Beat Saber maps from v2, v3, or v4 into one normalized schema.
2. Represent only standard playable color notes for learning and generation.
3. Convert those notes into a constrained token sequence.
4. Feed audio-derived features, usually precomputed MuQ embeddings, into an encoder/adaptor stack.
5. Train an autoregressive decoder to emit the token sequence.
6. During inference, generate token sequences with a hard grammar mask, decode them back to notes, optionally rerank multiple candidates, and export a playable map folder.

The project scope is intentionally narrower than the sibling `beat-weaver` repo. The README explicitly says it does not carry the full Beat Saver download/extraction/parquet-writing pipeline. Instead, it assumes a processed corpus already exists, usually in `../beat-weaver/data/...`.

## 3. Repository Layout And Intent

### Core package

- `muq_beat_weaver/schemas`: schema detection and per-version Beat Saber parsers
- `muq_beat_weaver/parsers`: top-level folder parsing and Info.dat handling
- `muq_beat_weaver/model`: tokenizer, dataset, training, inference, evaluation, export, timing, MuQ embedding helpers

### Operational scripts

- `scripts/train_muq_precomputed.py`: main training entry point used by saved experiments
- `scripts/generate_from_audio.py`: main inference/export entry point
- `scripts/cache_muq_subset.py`: builds raw MuQ feature cache
- `scripts/build_timing_metadata.py`: estimates BPM/downbeat metadata
- `scripts/build_muq_beatgrid_cache.py`: converts raw MuQ features onto the beat grid
- `scripts/audit_training_targets.py`: inspects dataset/token targets
- `scripts/compare_inference_modes.py`: compares independent vs rolling windowed inference
- `scripts/analyze_tensorboard_run.py`: summarizes a run from TensorBoard/checkpoints/json outputs

### Experiment evidence

- `configs/`: model/training variants
- `docs/`: experiment notes and pipeline plans
- `output/`: completed run artifacts, checkpoints, summaries, generation evals
- `tests/`: unit tests that encode intended behavior

### Non-source clutter

The repo also contains local environments, caches, and outputs:

- `.venv`, `.venv312_madmom`, `.uv-cache`
- `.pytest_cache`, `.pytest_tmp`
- large `output/` checkpoint trees

These are evidence of active experimentation, but not part of the software design itself.

## 4. Beat Saber Parsing Layer

This part is relatively clean and self-contained.

### Normalized schema

`muq_beat_weaver/schemas/normalized.py` defines:

- `Note`
- `Bomb`
- `Obstacle`
- `DifficultyInfo`
- `SongMetadata`
- `NormalizedBeatmap`

Everything downstream is designed to consume this normalized representation instead of version-specific Beat Saber JSON.

### Version detection

`muq_beat_weaver/schemas/detection.py` detects:

- info format via `_version` or `version`
- beatmap format via `_notes`, `colorNotes`, or `colorNotesData`

Supported versions are:

- v2
- v3
- v4

### Version-specific specifics

#### v2

`schemas/v2.py` parses:

- `_notes`, mixing normal notes and bombs
- `_obstacles`, deriving height from obstacle type

#### v3

`schemas/v3.py` parses:

- `colorNotes`
- `bombNotes`
- `obstacles`

It preserves `angle_offset`, which matters because v3/v4 can encode note rotation.

#### v4

`schemas/v4.py` parses the dereferenced indexed format:

- note entries point into `colorNotesData`
- bomb entries point into `bombNotesData`
- obstacle entries point into `obstaclesData`

The tests explicitly confirm that multiple v4 note entries can reuse the same data row.

### Folder parsing behavior

`parsers/beatmap_parser.py`:

- finds `Info.dat` case-insensitively
- parses song metadata plus difficulty file references
- loads each difficulty file
- detects its version independently
- computes note/bomb/obstacle counts
- computes simple NPS from first/last note times
- logs and skips bad difficulty files instead of failing the whole folder

This means the parser is intentionally tolerant. It is built for noisy real-world corpora.

### Real-world compatibility detail

The tests explicitly cover the mixed case of:

- v2 `Info.dat`
- v3 difficulty file

That is an important specificity: the repo expects real Beat Saber datasets to contain mixed-format maps and handles that deliberately.

## 5. Tokenization Design

The tokenizer is one of the most important design decisions in the repo.

### Vocabulary

`muq_beat_weaver/model/tokenizer.py` defines a fixed 291-token vocabulary:

- special tokens: `PAD`, `START`, `END`
- one token per difficulty
- `BAR`
- 64 positional tokens for 1/16-note positions inside a 4-beat bar
- left-hand note tokens
- right-hand note tokens
- left/right empty placeholders

This means the symbolic target is not raw note events. It is a bar-structured grammar with paired left/right emissions per quantized position.

### Quantization

Notes are quantized to:

- 16 subdivisions per beat
- 4 beats per bar
- 64 positions per bar

This is rigid. The repo is structurally committed to a 4/4, 1/16-grid representation.

### What is preserved vs discarded

Preserved:

- beat location, after quantization
- note color
- `(x, y)` grid coordinate
- cut direction
- difficulty token

Discarded from generation target:

- bombs
- obstacles
- many richer Beat Saber semantics
- exact sub-grid timing finer than 1/16
- multiple notes per hand at the same position, beyond “keep first”

That last point is important. If several left-hand notes or several right-hand notes land on the same quantized position, duplicates are dropped during encoding.

### Implication

This tokenizer makes the learning problem dramatically simpler, but it also defines the ceiling of the model:

- no bomb generation
- no obstacle generation
- no richer event/channel data
- strict 4/4 bar framing
- one left token and one right token per grid position

The project is therefore a note-placement generator, not a full Beat Saber chart generator.

## 6. Model Architecture

### Top-level model

`muq_beat_weaver/model/model.py` wires together:

- encoder
- adapter
- decoder

### Encoder modes

`encoder.py` supports two modes:

- `encoder_type == "muq"`: live MuQ model loaded via `MuQ.from_pretrained(...)`
- anything else, practically `muq_precomputed`: pass precomputed features through unchanged except for shape normalization

In practice, the experiment history shows the repo is centered on `muq_precomputed`, not end-to-end MuQ finetuning.

### Adapter

`adapter.py` uses:

- a linear projection from encoder dim to decoder dim
- optional layer norm
- dropout

This is the bridge between MuQ output space and decoder input space.

### Decoder

`decoder.py` is a transformer decoder with two variants:

- RoPE-based custom decoder stack when `use_rope=True`
- fallback to standard `nn.TransformerDecoder` with sinusoidal position encoding otherwise

The default config uses RoPE.

The decoder performs:

- causal self-attention over tokens
- cross-attention into audio memory
- feedforward layers

### Parameter scale

Saved run artifacts show the main base model has about 26.0M trainable parameters.

## 7. Audio Features And MuQ

### Audio helpers

`audio.py` supports two feature families:

- mel spectrograms, with optional onset envelope
- MuQ embeddings interpolated to a beat grid

The current repo direction strongly favors MuQ.

### MuQ integration

`muq_embeddings.py` contains:

- `MuQEmbedder`
- raw embedding extraction from audio
- long-audio windowing with overlap
- midpoint-ownership merging of overlapped windows
- export helpers and stats

Notable specifics:

- default MuQ label rate assumed from model config or 25 Hz fallback
- long audio is windowed at 180s with 30s overlap by default
- raw window embeddings are merged by midpoint ownership, not averaging

That same midpoint-ownership pattern reappears elsewhere in the repo. It is a repeated design motif.

## 8. Timing Alignment Is A Major Secondary Theme

This repo is not just “MuQ + decoder”. Timing alignment became a substantial research track.

### Why timing exists

Raw MuQ frames are on a time axis. The token targets live on a beat grid. The repo tries to align those worlds more carefully using:

- BPM
- first downbeat offset
- timing metadata per song

### `timing.py`

This module defines `SongTiming` and supports:

- loading/saving `timing_metadata.json`
- default metadata-only timing fallback
- timing fingerprints
- rebasing note dictionaries to a downbeat-relative beat grid
- auto-estimating timing from audio

### Estimation strategy

The estimator is:

1. try `madmom` downbeat tracking
2. if that fails, fall back to `librosa`

The docs make clear that `madmom` support required extra engineering because of environment incompatibilities.

### Very important specificity

When notes occur before the detected first downbeat, they can be dropped. The dataset stores the count in `notes_dropped_before_downbeat`.

That means pickup notes are not preserved if they would sit before the inferred downbeat reference. This is intentional, and it materially changes targets.

### Alignment path reuse

Timing metadata is used in:

- dataset preparation
- MuQ beat-grid cache building
- inference-time feature alignment

This is one of the better architectural decisions in the repo: the same timing concept is reused in training and inference instead of duplicated ad hoc.

## 9. Dataset Pipeline

`model/dataset.py` is the largest and most operationally dense module in the repo.

### Inputs expected

The dataset assumes a processed corpus with:

- `metadata.json`
- one or more `notes_*.parquet` files or `notes.parquet`
- `audio_manifest.json`
- optionally `timing_metadata.json`
- feature caches such as `muq_cache` or `muq_cache_beatgrid`

This processed corpus usually lives outside this repo, in the sibling `beat-weaver` repo.

### Core behavior

Dataset preparation:

- loads metadata and audio manifest
- loads timing metadata, or synthesizes defaults
- reads note parquet
- groups by `(song_hash, difficulty, characteristic)`
- splits by song hash, not by difficulty rows
- filters by difficulty, characteristic, BPM, audio availability, and optional duration
- rebases notes using timing metadata
- pretokenizes all kept samples
- caches the prepared corpus as a pickle

### Splitting

Splits are deterministic 80/10/10 by song hash via a seeded permutation.

The docs explicitly note a consequence: if the processed corpus changes in place, split membership changes. That is why expanded corpora are expected to live in a separate processed root.

### Feature cache behavior

If `encoder_type` is MuQ-related:

- it prefers `muq_cache_beatgrid` if present
- otherwise falls back to raw `muq_cache`
- if only raw MuQ cache exists, it interpolates to the beat grid at load time

If `encoder_type` is not MuQ:

- it uses or computes mel features

### Prepared corpus cache

Prepared sample state is cached in `processed_dir/dataset_cache/prepared_<hash>.pkl`.

The cache key depends on:

- relevant config fields
- metadata/timing/parquet file size and mtime
- manifest presence
- feature cache version file
- number and freshness of `.npy` cache files

That is a pragmatic, file-state-based cache invalidation strategy rather than content hashing.

### Training-time windowing

For long feature sequences:

- train split randomly selects a bar-aligned window start
- val/test always use the prefix window
- notes are cropped to that window and rebased locally

Important specifics:

- one frame corresponds to one 1/16-beat subdivision in beat-grid features
- train windows are bar-aligned on 64-frame boundaries

This is tested explicitly.

### Source balancing

`build_weighted_sampler()` oversamples official maps to hit an `official_ratio`, and weights custom maps by BeatSaver score.

This is a useful but subtle repo-specific choice: data quality weighting is baked into sampling rather than loss weighting.

### Code quality note

`BeatSaberDataset.__init__` duplicates much of the logic already extracted into `prepare_dataset_corpus()`. The newer prepared-corpus path is cleaner, but the file still carries both forms.

## 10. Training Loop

Training is implemented in `model/training.py`.

### Core features

- AdamW optimizer
- mixed precision on CUDA
- gradient accumulation
- gradient clipping
- cosine schedule with warmup
- TensorBoard logging
- checkpointing of model, optimizer, scaler, scheduler, and training state
- early stopping on validation loss
- optional short-run profiling/debug mode

### Validation

Validation reports:

- `val_loss`
- token-level accuracy

### Extra loss term

There is an optional `color_balance_loss` that penalizes left/right note imbalance, though default configs leave `color_balance_weight` at `0.0`.

### Scheduler specifics

Warmup defaults to `warmup_ratio` if it is set; `warmup_steps` is only used when `warmup_ratio` is disabled.

This is tested.

### Platform-specific behavior

On Windows, training uses `num_workers=0` because the author explicitly notes DataLoader worker deadlocks with spawn-based multiprocessing and `persistent_workers`.

That is an important practical specificity of the current environment.

### Post-training behavior

After training, the code automatically runs generation-time evaluation on the validation set and writes `generation_eval.json`.

This is a strong signal that the repo values “can the model actually generate plausible maps?” more than teacher-forced token metrics alone.

## 11. Inference And Generation

This is the most behaviorally distinctive part of the repo.

### Grammar-constrained generation

`model/inference.py` does not let the decoder emit arbitrary token sequences. It applies a hard next-token grammar mask:

- `START -> difficulty`
- `difficulty -> BAR`
- `BAR -> POS | BAR | END`
- `POS -> LEFT`
- `LEFT -> RIGHT`
- `RIGHT -> POS(next only) | BAR | END`

This is a major design choice. Instead of learning sequence legality statistically, the repo enforces legality structurally at decode time.

### Consequence

Generated outputs are guaranteed to be syntactically valid under the repo’s internal grammar, but not necessarily musically good or semantically rich.

### Long-song inference

`generate_full_song()` supports:

- `independent`: generate each window independently, then stitch
- `rolling`: carry a trailing-bar token prefix from one window into the next

Both use overlap logic. Independent mode merges notes by midpoint ownership. Rolling mode keeps some trailing bars as prompt continuation.

### Candidate reranking

The main user-facing inference script, `scripts/generate_from_audio.py`, does not just generate once.

It:

- generates multiple candidates with different seeds
- scores them with a hand-designed heuristic
- exports the best one
- optionally exports all candidates

The candidate scoring considers:

- pattern diversity
- active-bar ratio
- section density variation
- longest silence
- parity violations
- notes per second vs target difficulty density

This is important: the repo has already discovered that raw single-sample generation quality is not good enough, so it layers heuristic reranking on top.

## 12. Export Path

`model/exporter.py` exports generated notes into a playable Beat Saber v2 folder.

### It writes

- `Info.dat`
- one difficulty `.dat`
- copied audio file as `song<suffix>`

### It does not export

- bombs
- obstacles
- richer v3/v4 features

So even if the input corpus contains those, the generated result is always a simplified v2 note-only map.

## 13. Evaluation Metrics

`model/evaluate.py` includes lightweight metrics:

- onset F1
- NPS accuracy
- beat alignment
- parity violation rate
- pattern diversity
- NPS

These metrics are practical and cheap, not meant as exhaustive chart-quality judgment.

There is one especially revealing quirk:

- generated beat alignment is always measured against the same 1/16 grid the model emits on

So in generation-eval outputs, `beat_alignment` is frequently exactly `0.0`. That does not mean the model has perfect rhythmic understanding. It mostly means the symbolic representation itself is already locked to the quantized grid.

## 14. Script-Level Workflows

### `scripts/train_muq_precomputed.py`

This is the main training script used by the saved runs.

It:

- loads a config
- forces `encoder_type = "muq_precomputed"`
- builds train/val datasets from the external processed corpus
- trains and writes checkpoints/output summaries

Notable oddity:

- it does `os.chdir(audio_manifest_path.parent.parent)` before dataset construction

That suggests some path assumptions or historical coupling with the sibling corpus layout.

### `scripts/generate_from_audio.py`

This is the main practical demo script.

It:

- loads a checkpoint
- extracts MuQ features from one audio file
- resolves BPM and downbeat offset from manual input, timing metadata, or auto estimation
- interpolates MuQ features to beat-grid features
- generates multiple candidate maps
- reranks them
- exports a v2 playable map

This is the closest thing to a real end-user flow in the repo.

### `scripts/build_timing_metadata.py`

Builds or refreshes `timing_metadata.json` across a corpus.

It:

- reads manifest and metadata
- optionally skips already-strong timing entries
- uses multiprocessing if requested
- writes timing metadata plus a summary JSON

### `scripts/build_muq_beatgrid_cache.py`

Not fully inspected line-by-line here, but docs and dataset behavior make its role clear:

- consume raw `muq_cache`
- use timing metadata
- output beat-grid-aligned MuQ features into `muq_cache_beatgrid`

The docs and pipeline script also indicate it writes per-song timing sidecars and can skip unchanged songs via timing fingerprints.

### `scripts/audit_training_targets.py`

This exists because target quality needs auditing independently of model loss. That is a good sign: the author is aware that bad preprocessing can silently poison training.

### `scripts/compare_inference_modes.py`

This reflects an active open question in the repo: whether independent windowing or rolling context works better for long songs.

### `scripts/run_beatsaver_x3_pipeline.ps1`

This is the clearest orchestration script in the repo. It ties the workspace to the sibling `beat-weaver` repo and runs a staged corpus-expansion flow:

- build a larger processed corpus externally
- build or refresh audio manifests
- cache raw MuQ
- build beat-grid MuQ cache
- train the base `bs8` model on the larger corpus

That script confirms the repo’s role in the broader system: model-side experimentation on top of an upstream data-prep pipeline, not corpus ingestion itself.

## 15. Configurations And Experiment Direction

The configs show the repo’s trajectory.

### Main branches

- `muq_frozen_base.json`: live MuQ encoder, frozen
- `muq_partial_unfreeze.json`: live MuQ with partial unfreeze intent
- `muq_frozen_*_bs8*.json`: precomputed MuQ, larger batch size, base/small decoder variants

### Strongest signal

The serious saved runs are all around:

- `encoder_type = muq_precomputed`
- `encoder_output_dim = 1024`
- decoder dim 512
- 6 decoder layers
- batch size 8

So the repo’s real baseline is not end-to-end MuQ training. It is “freeze MuQ upstream, cache embeddings, train the symbolic decoder efficiently.”

### Warmup comparison

There are two closely related 45-epoch configs:

- `muq_frozen_base_bs8_45ep.json` with `warmup_ratio = 0.1`
- `muq_frozen_base_bs8_45ep_wr005.json` with `warmup_ratio = 0.05`

That means warmup ratio itself became an experiment variable.

## 16. What The Saved Outputs Reveal

The output artifacts are unusually informative.

### Training scale

Example runs show:

- `muq_precomputed_2000_45ep_base_bs8`: 2952 train samples, 391 val samples, best val loss `1.9127`, about 10,153s total, 26.0M params
- `muq_precomputed_beatsaver_x3_base_bs8`: 13,700 train samples, 1,758 val samples, best val loss `1.64`, about 43,455s total

So the larger corpus improves validation loss materially.

### Preprocessing funnel is aggressive

`output/audit_training_targets.json` provides a particularly useful snapshot of the data pipeline’s selectivity. In that audit:

- 114,816 grouped map candidates were observed
- only 3,749 survived into training-ready samples
- 614 samples encoded as empty targets
- 613 samples contained only invalid notes after grid filtering
- 102,718 samples began with at least one empty bar

That matters because it explains why dataset quality and target auditing are first-class concerns in this repo: the raw corpus is noisy, pickup-heavy, and not directly suitable for training without heavy pruning/rebasing.

### But generation quality is still unstable

The generation eval JSONs show repeated patterns:

- many samples with `generated_note_count = 0`
- many samples with the same repeated counts like `654`
- low pattern diversity in many outputs
- decent or moderate NPS accuracy on some samples
- modest onset F1 overall

This strongly suggests the model often falls into stereotyped decode behaviors or terminates in repetitive regimes despite the grammar constraints.

### Beat alignment metric is misleadingly perfect

In generation evals, `beat_alignment` is frequently all zeros. As noted earlier, that mostly reflects the quantized token format, not superior rhythmic learning.

### Rolling observations from artifacts

The presence of `compare_inference_modes.py`, candidate reranking, and generation-eval outputs indicates that inference behavior is still a research problem, not a solved layer.

## 17. Tests: What The Repo Is Confident About

The tests are solid for data representation and helper logic.

They cover:

- gzip and json `.dat` reading
- parsing v2/v3/v4 maps
- mixed-format compatibility
- folder ignore rules for autosaves/backups
- tokenizer vocabulary, encoding, decoding, and token descriptions
- grammar validity in inference
- long-song generation helper behavior
- exporter behavior
- timing fallback and rebasing behavior
- dataset windowing rules
- warmup scheduling and generation-eval aggregation helpers
- MuQ embedding window merge helpers

### What tests do not cover deeply

They do not validate:

- actual convergence quality
- full training/inference on realistic corpora
- quality of candidate reranking
- correctness of `madmom` behavior on large corpora
- end-to-end map quality in gameplay terms

So the repo is well tested at the unit/helper level, but not strongly integration-tested.

## 18. Important Specificities And Caveats

### 1. The CLI is not the product

`muq_beat_weaver.cli` only defines placeholder subcommands and parses args; it does not dispatch them. Some wrapper scripts still point at that CLI surface, so the real working command paths are the explicit task scripts such as `train_muq_precomputed.py` and `generate_from_audio.py`.

### 2. The repo is tightly coupled to a sibling corpus repo

Many defaults point to:

- `../beat-weaver/data/processed`
- `../beat-weaver/data/audio_manifest.json`

So this repo is not self-contained operationally.

### 3. The symbolic target is deliberately simplified

The model learns only color notes. Bombs and obstacles are parsed but not modeled or exported.

### 4. The representation assumes 4/4 and 1/16 quantization

That is embedded in tokenizer structure, dataset windowing, and long-song merging assumptions.

### 5. Timing alignment can drop notes

Pickup notes before the inferred first downbeat may be removed.

### 6. Generated outputs are legal by grammar, not necessarily good

The grammar mask prevents malformed token sequences, but saved outputs show this does not prevent degenerate or empty maps.

### 7. Candidate reranking is compensating for model weaknesses

The inference script’s scoring/reranking layer is not cosmetic. It is an operational necessity.

### 8. Beat-grid cache presence changes behavior materially

If `muq_cache_beatgrid` exists, the dataset uses it. If not, it falls back to raw MuQ and interpolates on the fly. That means training behavior can differ depending on local artifact state.

### 9. There is some architectural duplication

`BeatSaberDataset.__init__` still duplicates a large amount of logic that also exists in `prepare_dataset_corpus()`.

### 10. Environment handling is part of the design

The docs make clear that:

- training env and timing env may need to differ
- `madmom` support is fragile
- Windows multiprocessing issues shaped implementation decisions

This is not incidental. It is part of how the repo actually works.

## 19. Overall Assessment

This is a serious research repo, not a polished library.

Its strongest qualities are:

- clear narrowing of scope around the MuQ-to-decoder boundary
- practical normalization of Beat Saber formats
- disciplined token grammar
- useful timing-alignment work
- script-driven workflows with saved empirical artifacts
- decent unit coverage for critical helper logic

Its main weaknesses or open problems are:

- quality ceiling from the simplified note-only representation
- unstable generation behavior despite decent training infrastructure
- operational coupling to external corpus layout
- some internal duplication and stubbed package CLI
- environment fragility around timing estimation

The repo is best described as:

“a MuQ-precomputed Beat Saber note-generation research harness with timing-aware preprocessing, grammar-constrained decoding, and extensive experiment residue showing active iteration on long-song inference and dataset alignment.”

## 20. Most Important Takeaways

If I had to summarize the repo in a few points:

- The real project is in `scripts/` plus `muq_beat_weaver/model`, not in the package CLI.
- The central bet is that precomputed MuQ embeddings can replace traditional audio features cleanly enough for Beat Saber note generation.
- The learned target is intentionally narrow: quantized left/right color note placements only.
- Timing alignment became important enough to form a second major pillar beside MuQ integration.
- Saved runs show better loss on larger corpora, but generation quality remains inconsistent and often degenerate.
- The repo is built for experimentation speed and architectural control, not for end-user polish.
