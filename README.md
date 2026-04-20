# muq-beat-weaver

MuQ-first experiment repo for swapping audio feature extraction and encoder components in the Beat Weaver pipeline.

## Scope

This repo is for rapid experimentation on the model-side pipeline boundary:

- waveform input
- pretrained MuQ encoder
- adapter / projection
- autoregressive decoder
- Beat Saber tokenization, parsing, export, and evaluation

It intentionally does not carry over the full Beat Saver download, extraction, and parquet-writing pipeline from eat-weaver.

## Initial status

The scaffold preserves the reusable tokenizer, decoder-side logic, parsers, schemas, and tests that matter for encoder experiments.

The first encoder target is MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").

## Related Docs

- [madmom timing experiment](E:\github_repos\muq-beat-weaver\docs\madmom_timing_experiment.md): notes on the downbeat-aware timing metadata pipeline, environment setup, and recommended rebuild flow for `timing_metadata.json` and `muq_cache_beatgrid`.

## run inference

example:

```
uv run python scripts\generate_from_audio.py `
    --checkpoint output\muq_precomputed_2000_45ep_base_bs8\checkpoints\best `
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
