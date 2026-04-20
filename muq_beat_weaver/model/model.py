from __future__ import annotations

import torch
import torch.nn as nn

from muq_beat_weaver.model.adapter import EncoderAdapter
from muq_beat_weaver.model.config import ModelConfig
from muq_beat_weaver.model.decoder import TokenDecoder
from muq_beat_weaver.model.encoder import MuQEncoderWrapper, PrecomputedMuQEncoder


class BeatWeaverModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        if config.encoder_type == 'muq':
            self.encoder = MuQEncoderWrapper(config)
        else:
            self.encoder = PrecomputedMuQEncoder(config)
        self.adapter = EncoderAdapter(config)
        self.decoder = TokenDecoder(config)

    def encode(
        self,
        audio: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        memory, memory_mask = self.encoder(audio, audio_mask)
        memory = self.adapter(memory)
        return memory, memory_mask

    def forward(self, audio: torch.Tensor, tokens: torch.Tensor, audio_mask: torch.Tensor | None = None, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        memory, memory_mask = self.encode(audio, audio_mask)
        return self.decoder(tokens, memory, token_mask, memory_mask)

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

