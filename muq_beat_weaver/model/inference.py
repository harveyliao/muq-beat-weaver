"""Autoregressive generation with grammar-constrained decoding."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from muq_beat_weaver.model.config import ModelConfig
from muq_beat_weaver.model.tokenizer import (
    BAR,
    DIFF_EASY,
    DIFF_EXPERT_PLUS,
    END,
    LEFT_BASE,
    LEFT_COUNT,
    LEFT_EMPTY,
    POS_BASE,
    POS_COUNT,
    RIGHT_BASE,
    RIGHT_COUNT,
    RIGHT_EMPTY,
    START,
    VOCAB_SIZE,
    decode_tokens,
    difficulty_to_token,
)
from muq_beat_weaver.model.transformer import BeatWeaverModel
from muq_beat_weaver.schemas.normalized import Note


def _last_pos_in_bar(tokens: list[int]) -> int:
    """Infer the last POS offset active in the current bar."""
    last_pos = -1
    for token in tokens:
        if token == BAR:
            last_pos = -1
        elif POS_BASE <= token < POS_BASE + POS_COUNT:
            last_pos = token - POS_BASE
    return last_pos


def _extract_trailing_bar_prefix(
    tokens: list[int],
    *,
    keep_bars: int,
    max_prefix_tokens: int,
) -> tuple[list[int], float]:
    """Keep the last N bars of a generated sequence as a continuation prefix."""
    if keep_bars <= 0 or len(tokens) <= 2:
        return tokens[:2], 0.0

    body = [token for token in tokens[2:] if token != END]
    if not body:
        return tokens[:2], 0.0

    bar_positions = [idx for idx, token in enumerate(body) if token == BAR]
    if not bar_positions:
        return tokens[:2], 0.0

    bars_kept = min(keep_bars, len(bar_positions))
    start_body = bar_positions[-bars_kept]
    trailing = body[start_body:]
    while len(trailing) + 2 > max_prefix_tokens and bars_kept > 1:
        bars_kept -= 1
        start_body = bar_positions[-bars_kept]
        trailing = body[start_body:]

    if len(trailing) + 2 > max_prefix_tokens:
        return tokens[:2], 0.0

    return tokens[:2] + trailing, bars_kept * 4.0


def _build_grammar_mask(last_token: int, last_pos_in_bar: int = -1) -> torch.Tensor:
    """Build a boolean mask over the vocabulary for valid next tokens.

    Returns a tensor of shape (VOCAB_SIZE,) where True = allowed.

    Args:
        last_token: The most recently generated token.
        last_pos_in_bar: The last POS offset used in the current bar (-1 if none).
            Used to enforce strictly increasing positions within a bar,
            preventing multiple notes at the same beat.

    Grammar rules:
        START      鈫?DIFF_*
        DIFF_*     鈫?BAR
        BAR        鈫?POS_* | BAR | END
        POS_*      鈫?LEFT_* | LEFT_EMPTY
        LEFT_*     鈫?RIGHT_* | RIGHT_EMPTY
        RIGHT_*    鈫?POS_* (strictly >) | BAR | END
    """
    mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool)

    if last_token == START:
        # After START 鈫?only difficulty tokens
        mask[DIFF_EASY: DIFF_EXPERT_PLUS + 1] = True

    elif DIFF_EASY <= last_token <= DIFF_EXPERT_PLUS:
        # After DIFF 鈫?only BAR
        mask[BAR] = True

    elif last_token == BAR:
        # After BAR 鈫?POS, BAR, or END
        mask[POS_BASE: POS_BASE + POS_COUNT] = True
        mask[BAR] = True
        mask[END] = True

    elif POS_BASE <= last_token < POS_BASE + POS_COUNT:
        # After POS 鈫?LEFT note or LEFT_EMPTY
        mask[LEFT_EMPTY] = True
        mask[LEFT_BASE: LEFT_BASE + LEFT_COUNT] = True

    elif last_token == LEFT_EMPTY or (LEFT_BASE <= last_token < LEFT_BASE + LEFT_COUNT):
        # After LEFT 鈫?RIGHT note or RIGHT_EMPTY
        mask[RIGHT_EMPTY] = True
        mask[RIGHT_BASE: RIGHT_BASE + RIGHT_COUNT] = True

    elif last_token == RIGHT_EMPTY or (RIGHT_BASE <= last_token < RIGHT_BASE + RIGHT_COUNT):
        # After RIGHT 鈫?POS (strictly increasing), BAR, or END
        # Only allow POS tokens with offset > last_pos_in_bar
        min_next = last_pos_in_bar + 1
        if min_next < POS_COUNT:
            mask[POS_BASE + min_next: POS_BASE + POS_COUNT] = True
        mask[BAR] = True
        mask[END] = True

    else:
        # Unknown state 鈥?allow everything except PAD/START
        mask[2:] = True

    return mask


def _sample_with_filter(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """Sample a token from logits with temperature, top-k, and top-p filtering."""
    if temperature <= 0:
        return logits.argmax().item()

    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_val = values[-1]
        logits = torch.where(logits < min_val, torch.full_like(logits, float("-inf")), logits)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative prob above threshold
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = float("-inf")
        # Scatter back
        logits = torch.zeros_like(logits).scatter(0, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


@torch.no_grad()
def generate(
    model: BeatWeaverModel,
    mel_spectrogram: torch.Tensor,
    difficulty: str,
    config: ModelConfig,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int | None = None,
    mel_mask: torch.Tensor | None = None,
    initial_tokens: list[int] | None = None,
) -> list[int]:
    """Generate a token sequence autoregressively.

    Args:
        model: Trained BeatWeaverModel.
        mel_spectrogram: (n_mels, T_audio) 鈥?single spectrogram (no batch dim).
        difficulty: Difficulty name (e.g., "Expert").
        config: Model configuration.
        temperature: Sampling temperature (0 = greedy).
        top_k: Top-k filtering (0 = disabled).
        top_p: Top-p / nucleus filtering (1.0 = disabled).
        seed: Random seed for reproducibility.
        mel_mask: (T_audio,) 鈥?True for valid positions.
        initial_tokens: Optional token prefix starting with START and the
            requested difficulty token.

    Returns:
        List of token IDs including START and END.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    device = next(model.parameters()).device

    # Prepare mel: add batch dimension
    mel = mel_spectrogram.unsqueeze(0).to(device)  # (1, n_mels, T_audio)
    if mel_mask is not None:
        mel_mask = mel_mask.unsqueeze(0).to(device)  # (1, T_audio)

    # Encode audio once
    memory, memory_mask = model.encode(mel, mel_mask)

    diff_token = difficulty_to_token(difficulty)
    if initial_tokens is None:
        tokens = [START, diff_token]
    else:
        if len(initial_tokens) < 2 or initial_tokens[0] != START:
            raise ValueError("initial_tokens must begin with START and a difficulty token.")
        if initial_tokens[1] != diff_token:
            raise ValueError("initial_tokens difficulty does not match requested difficulty.")
        tokens = [token for token in initial_tokens if token != END]
        if len(tokens) > config.max_seq_len:
            raise ValueError("initial_tokens exceeds config.max_seq_len.")
    last_pos_in_bar = _last_pos_in_bar(tokens)

    for _ in range(config.max_seq_len - len(tokens)):
        # Prepare decoder input
        token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        token_mask = torch.ones(1, len(tokens), dtype=torch.bool, device=device)

        logits = model.decoder(token_tensor, memory, token_mask, memory_mask)
        # logits: (1, seq_len, vocab_size) 鈥?take last position
        next_logits = logits[0, -1]  # (vocab_size,)

        # Apply grammar mask (with position tracking for one-note-per-color-per-beat)
        grammar_mask = _build_grammar_mask(tokens[-1], last_pos_in_bar).to(device)
        next_logits[~grammar_mask] = float("-inf")

        # Sample
        next_token = _sample_with_filter(next_logits, temperature, top_k, top_p)
        tokens.append(next_token)

        # Update position tracking
        if next_token == BAR:
            last_pos_in_bar = -1  # Reset on new bar
        elif POS_BASE <= next_token < POS_BASE + POS_COUNT:
            last_pos_in_bar = next_token - POS_BASE

        if next_token == END:
            break

    return tokens


def generate_full_song(
    model: BeatWeaverModel,
    mel_spectrogram: torch.Tensor,
    difficulty: str,
    config: ModelConfig,
    bpm: float,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int | None = None,
    inference_mode: str = "independent",
) -> list[Note]:
    """Generate a complete Beat Saber map by processing audio in overlapping windows.

    For short audio that fits in a single window, this is equivalent to calling
    generate() + decode_tokens(). For longer audio, the mel is split into
    overlapping windows, each generating a token sequence that is decoded to
    notes and merged using midpoint ownership in the overlap zones.

    Args:
        model: Trained BeatWeaverModel.
        mel_spectrogram: (n_mels, T_audio) 鈥?full beat-aligned spectrogram.
        difficulty: Difficulty name (e.g., "Expert").
        config: Model configuration.
        bpm: Song BPM (needed for beat offset calculation and token decoding).
        temperature: Sampling temperature.
        top_k: Top-k filtering (0 = disabled).
        top_p: Top-p / nucleus filtering (1.0 = disabled).
        seed: Random seed for reproducibility.
        inference_mode: "independent" for stitched per-window generation or
            "rolling" to carry overlap token context forward across windows.

    Returns:
        List of Note objects spanning the full song, sorted by beat.
    """
    total_frames = mel_spectrogram.shape[1]
    max_len = config.max_audio_len

    if inference_mode not in {"independent", "rolling"}:
        raise ValueError(f"Unknown inference_mode: {inference_mode!r}")

    # Single window 鈥?generate directly
    if total_frames <= max_len:
        tokens = generate(
            model, mel_spectrogram, difficulty, config,
            temperature=temperature, top_k=top_k, top_p=top_p, seed=seed,
        )
        return decode_tokens(tokens, bpm)

    # Multi-window generation
    overlap = min(max_len // 4, 1024)
    stride = max_len - overlap

    # Compute window start positions
    starts: list[int] = []
    pos = 0
    while pos < total_frames:
        starts.append(pos)
        if pos + max_len >= total_frames:
            break
        pos += stride

    if inference_mode == "rolling":
        result: list[Note] = []
        overlap_beats = overlap / 16.0
        overlap_bars = max(1, int(overlap_beats // 4))
        rolling_prefix: list[int] | None = None
        prefix_beats = 0.0

        for i, start in enumerate(starts):
            end = start + max_len
            window_mel = mel_spectrogram[:, start:end]

            if window_mel.shape[1] < max_len:
                pad_size = max_len - window_mel.shape[1]
                window_mel = torch.nn.functional.pad(window_mel, (0, pad_size))

            window_seed = seed + i if seed is not None else None
            tokens = generate(
                model,
                window_mel,
                difficulty,
                config,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=window_seed,
                initial_tokens=rolling_prefix,
            )

            decoded_notes = decode_tokens(tokens, bpm)
            window_notes: list[Note] = []
            for note in decoded_notes:
                if i > 0 and note.beat < prefix_beats:
                    continue
                if i > 0:
                    note.beat -= prefix_beats
                    note.time_seconds = note.beat * 60.0 / bpm
                note.beat += start / 16.0
                note.time_seconds = note.beat * 60.0 / bpm
                window_notes.append(note)

            result.extend(window_notes)
            rolling_prefix, prefix_beats = _extract_trailing_bar_prefix(
                tokens,
                keep_bars=overlap_bars,
                max_prefix_tokens=max(2, config.max_seq_len // 2),
            )

        result.sort(key=lambda n: (n.beat, n.color))
        deduped: list[Note] = []
        seen = set()
        for note in result:
            key = (round(note.beat, 6), note.color, note.x, note.y, note.cut_direction)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(note)
        return deduped

    all_window_notes: list[tuple[int, list[Note]]] = []

    for i, start in enumerate(starts):
        end = start + max_len
        window_mel = mel_spectrogram[:, start:end]

        # Zero-pad last window if needed
        if window_mel.shape[1] < max_len:
            pad_size = max_len - window_mel.shape[1]
            window_mel = torch.nn.functional.pad(window_mel, (0, pad_size))

        # Use different seed per window for variety (if seed provided)
        window_seed = seed + i if seed is not None else None

        tokens = generate(
            model, window_mel, difficulty, config,
            temperature=temperature, top_k=top_k, top_p=top_p, seed=window_seed,
        )

        # Decode tokens 鈥?notes have beats relative to window start (bar 0)
        window_notes = decode_tokens(tokens, bpm)

        # Offset all beats by the window's start position in frames
        # Each frame = 1/16th note subdivision, beat = frame / 16
        beat_offset = start / 16.0
        for note in window_notes:
            note.beat += beat_offset
            note.time_seconds = note.beat * 60.0 / bpm

        all_window_notes.append((start, window_notes))

    # Merge with midpoint ownership 鈥?each window owns notes up to the
    # midpoint of its overlap with the next window, and from the midpoint
    # of its overlap with the previous window.
    result: list[Note] = []
    for i, (start, notes) in enumerate(all_window_notes):
        min_beat = 0.0
        max_beat = float("inf")

        if i > 0:
            prev_start = all_window_notes[i - 1][0]
            # Overlap zone between prev and current: [start, prev_start + max_len)
            overlap_mid_frame = (start + prev_start + max_len) / 2.0
            min_beat = overlap_mid_frame / 16.0

        if i < len(all_window_notes) - 1:
            next_start = all_window_notes[i + 1][0]
            # Overlap zone between current and next: [next_start, start + max_len)
            overlap_mid_frame = (next_start + start + max_len) / 2.0
            max_beat = overlap_mid_frame / 16.0

        result.extend(n for n in notes if min_beat <= n.beat < max_beat)

    result.sort(key=lambda n: (n.beat, n.color))
    return result


