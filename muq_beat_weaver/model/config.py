from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 291
    max_seq_len: int = 2048
    max_audio_len: int = 8192

    encoder_type: str = "muq_precomputed"
    encoder_input_type: str = "waveform"
    muq_model_name: str = "OpenMuQ/MuQ-large-msd-iter"
    target_sample_rate: int = 24000
    encoder_output_dim: int = 1024
    freeze_encoder: bool = True
    unfreeze_last_n_layers: int = 0

    decoder_layers: int = 6
    decoder_dim: int = 512
    decoder_heads: int = 8
    decoder_ff_dim: int = 2048

    batch_size: int = 2
    learning_rate: float = 1e-4
    encoder_learning_rate: float = 1e-5
    warmup_steps: int = 4000
    warmup_ratio: float | None = 0.1
    max_epochs: int = 100
    dropout: float = 0.1
    label_smoothing: float = 0.1
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 4
    early_stopping_patience: int = 10
    save_every_n_epochs: int = 0
    generation_eval_samples: int = 32
    generation_eval_temperature: float = 0.0
    generation_eval_top_k: int = 0
    generation_eval_top_p: float = 1.0
    generation_eval_seed: int = 1234

    official_ratio: float = 0.2
    min_difficulty: str = "Easy"
    characteristics: list[str] | None = None
    min_bpm: float = 0.0
    max_bpm: float = 9999.0
    max_audio_duration: float = 0.0

    use_rope: bool = True
    project_encoder_to_decoder_dim: bool = True
    adapter_layer_norm: bool = True

    n_mels: int = 80
    n_fft: int = 2048
    hop_length: int = 512
    sample_rate: int = 22050
    encoder_layers: int = 6
    encoder_dim: int = 512
    encoder_heads: int = 8
    encoder_ff_dim: int = 2048
    use_onset_features: bool = False
    use_conformer: bool = False
    conformer_kernel_size: int = 31
    color_balance_weight: float = 0.0
    density_loss_weight: float = 0.1

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ModelConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


