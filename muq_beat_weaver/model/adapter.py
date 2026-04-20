from __future__ import annotations

import torch
import torch.nn as nn

from muq_beat_weaver.model.config import ModelConfig


class EncoderAdapter(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        out_dim = config.decoder_dim if config.project_encoder_to_decoder_dim else config.encoder_output_dim
        self.proj = nn.Linear(config.encoder_output_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if config.adapter_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.norm(self.proj(memory)))

