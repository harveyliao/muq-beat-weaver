"""Tests for bar-aligned dataset windowing helpers."""

import pytest

from muq_beat_weaver.model.dataset import _select_window_start, _slice_notes_to_window


def test_select_window_start_is_bar_aligned_for_train():
    starts = set()
    for _ in range(20):
        start = _select_window_start(total_frames=512, window_frames=128, split="train")
        assert start % 64 == 0
        assert 0 <= start <= 384
        starts.add(start)

    assert len(starts) > 1


def test_select_window_start_uses_prefix_for_eval():
    assert _select_window_start(total_frames=512, window_frames=128, split="val") == 0
    assert _select_window_start(total_frames=512, window_frames=128, split="test") == 0


def test_slice_notes_to_window_rebases_beats():
    notes = [
        {"beat": 3.75, "time_seconds": 0.0, "x": 0, "y": 0, "color": 0, "cut_direction": 1},
        {"beat": 4.25, "time_seconds": 0.0, "x": 1, "y": 1, "color": 1, "cut_direction": 2},
        {"beat": 7.75, "time_seconds": 0.0, "x": 2, "y": 1, "color": 0, "cut_direction": 3},
        {"beat": 8.0, "time_seconds": 0.0, "x": 3, "y": 2, "color": 1, "cut_direction": 4},
    ]

    sliced = _slice_notes_to_window(notes, start_frame=64, end_frame=128, bpm=120.0)

    assert len(sliced) == 2
    assert sliced[0].beat == pytest.approx(0.25)
    assert sliced[1].beat == pytest.approx(3.75)
    assert sliced[0].time_seconds == pytest.approx(0.125)
    assert sliced[1].time_seconds == pytest.approx(1.875)
