from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from muq_beat_weaver.model.config import ModelConfig


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 16384, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 16384):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_len = max_len

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        return freqs.cos().unsqueeze(0).unsqueeze(0), freqs.sin().unsqueeze(0).unsqueeze(0)


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, rope_cos: torch.Tensor | None = None, rope_sin: torch.Tensor | None = None, attn_mask: torch.Tensor | None = None, key_padding_mask: torch.Tensor | None = None, is_causal: bool = False) -> torch.Tensor:
        batch_size, query_len, _ = query.shape
        _, key_len, _ = key.shape
        q = self.q_proj(query).view(batch_size, query_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, key_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, key_len, self.nhead, self.head_dim).transpose(1, 2)
        if rope_cos is not None and rope_sin is not None:
            q = _apply_rotary_emb(q, rope_cos[:, :, :query_len], rope_sin[:, :, :query_len])
            k = _apply_rotary_emb(k, rope_cos[:, :, :key_len], rope_sin[:, :, :key_len])
        if key_padding_mask is not None and attn_mask is None:
            kpm = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = torch.zeros(batch_size, 1, query_len, key_len, device=query.device, dtype=query.dtype)
            attn_mask.masked_fill_(kpm, float('-inf'))
        elif key_padding_mask is not None and attn_mask is not None:
            kpm = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, query_len, key_len).clone()
            attn_mask.masked_fill_(kpm, float('-inf'))
            is_causal = False
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal, dropout_p=self.attn_dropout.p if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(batch_size, query_len, self.d_model)
        return self.out_proj(out)


class RoPEDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = RoPEMultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = RoPEMultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor, tgt_mask: torch.Tensor | None = None, tgt_key_padding_mask: torch.Tensor | None = None, memory_key_padding_mask: torch.Tensor | None = None, tgt_is_causal: bool = False) -> torch.Tensor:
        x = self.norm1(tgt)
        x = self.self_attn(x, x, x, rope_cos=rope_cos, rope_sin=rope_sin, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, is_causal=tgt_is_causal and tgt_mask is None)
        tgt = tgt + self.dropout1(x)
        x = self.norm2(tgt)
        x = self.cross_attn(x, memory, memory, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(x)
        x = self.norm3(tgt)
        x = self.linear2(self.dropout3(self.activation(self.linear1(x))))
        tgt = tgt + self.dropout4(x)
        return tgt


class TokenDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.use_rope = config.use_rope
        self.embedding = nn.Embedding(config.vocab_size, config.decoder_dim)
        self.output_proj = nn.Linear(config.decoder_dim, config.vocab_size)
        if config.use_rope:
            self.rope = RotaryPositionalEncoding(config.decoder_dim // config.decoder_heads, max_len=config.max_seq_len)
            self.dropout = nn.Dropout(config.dropout)
            self.layers = nn.ModuleList([
                RoPEDecoderLayer(config.decoder_dim, config.decoder_heads, config.decoder_ff_dim, config.dropout)
                for _ in range(config.decoder_layers)
            ])
        else:
            self.pos_enc = SinusoidalPositionalEncoding(config.decoder_dim, max_len=config.max_seq_len, dropout=config.dropout)
            decoder_layer = nn.TransformerDecoderLayer(d_model=config.decoder_dim, nhead=config.decoder_heads, dim_feedforward=config.decoder_ff_dim, dropout=config.dropout, batch_first=True)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)

    def forward(self, tokens: torch.Tensor, memory: torch.Tensor, token_mask: torch.Tensor | None = None, memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = tokens.size(1)
        x = self.embedding(tokens)
        tgt_key_padding_mask = ~token_mask if token_mask is not None else None
        memory_key_padding_mask = ~memory_mask if memory_mask is not None else None
        if self.use_rope:
            x = self.dropout(x)
            cos, sin = self.rope(seq_len, tokens.device)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=tokens.device)
            for layer in self.layers:
                x = layer(x, memory, cos, sin, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        else:
            x = self.pos_enc(x)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=tokens.device)
            x = self.decoder(x, memory, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, tgt_is_causal=True)
        return self.output_proj(x)

