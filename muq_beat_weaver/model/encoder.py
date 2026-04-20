from __future__ import annotations

import torch
import torch.nn as nn

from muq_beat_weaver.model.config import ModelConfig


class MuQEncoderWrapper(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        try:
            from muq import MuQ
        except ImportError as exc:
            raise ImportError("MuQ is required for encoder_type='muq'.") from exc
        self.model = MuQ.from_pretrained(config.muq_model_name)
        if config.freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, waveform: torch.Tensor, waveform_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        with torch.set_grad_enabled(any(p.requires_grad for p in self.model.parameters())):
            output = self.model(waveform.float(), attention_mask=waveform_mask, output_hidden_states=False)
        memory = output.last_hidden_state
        memory_mask = None
        if waveform_mask is not None:
            valid = waveform_mask.sum(dim=1)
            out_steps = memory.shape[1]
            scale = out_steps / waveform_mask.shape[1]
            out_lengths = torch.clamp((valid.float() * scale).ceil().long(), min=1, max=out_steps)
            positions = torch.arange(out_steps, device=memory.device).unsqueeze(0)
            memory_mask = positions < out_lengths.unsqueeze(1)
        return memory, memory_mask


class PrecomputedMuQEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    def forward(self, features: torch.Tensor, feature_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if features.dim() != 3:
            raise ValueError("Expected precomputed MuQ features with shape (batch, time, hidden) or (batch, hidden, time).")
        if features.shape[-1] == self.config.encoder_output_dim:
            memory = features
        elif features.shape[1] == self.config.encoder_output_dim:
            memory = features.transpose(1, 2)
        else:
            raise ValueError("Could not infer MuQ feature layout.")
        return memory, feature_mask

