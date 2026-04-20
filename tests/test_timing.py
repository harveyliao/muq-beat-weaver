import json
import uuid
from pathlib import Path

import numpy as np
import pytest

from muq_beat_weaver.model.audio import interpolate_muq_to_beat_grid
from muq_beat_weaver.model.dataset import _resolve_sample_bpm_and_notes
from muq_beat_weaver.model.timing import load_timing_metadata, resolve_single_song_timing


def test_load_timing_metadata_falls_back_to_metadata():
    processed_dir = Path("E:/github_repos/muq-beat-weaver/.tmp") / f"timing_test_{uuid.uuid4().hex}" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "metadata.json").write_text(
        json.dumps(
            [
                {"hash": "song_a", "bpm": 128.0},
                {"hash": "song_b", "bpm": 150.0},
            ]
        ),
        encoding="utf-8",
    )
    (processed_dir / "timing_metadata.json").write_text(
        json.dumps(
            {
                "song_a": {
                    "bpm": 127.5,
                    "first_downbeat_sec": 1.25,
                    "timing_source": "madmom",
                    "timing_confidence": "high",
                }
            }
        ),
        encoding="utf-8",
    )

    timing = load_timing_metadata(processed_dir)

    assert timing["song_a"]["timing_source"] == "madmom"
    assert timing["song_a"]["first_downbeat_sec"] == pytest.approx(1.25)
    assert timing["song_b"]["timing_source"] == "metadata_bpm_only"
    assert timing["song_b"]["bpm"] == pytest.approx(150.0)


def test_resolve_sample_bpm_and_notes_rebases_to_downbeat_grid():
    notes = [
        {"beat": 0.0, "time_seconds": 1.99, "x": 0, "y": 0, "color": 0, "cut_direction": 1, "bpm": 120.0},
        {"beat": 2.0, "time_seconds": 3.0, "x": 1, "y": 1, "color": 1, "cut_direction": 2, "bpm": 120.0},
        {"beat": 4.0, "time_seconds": 4.0, "x": 2, "y": 1, "color": 0, "cut_direction": 3, "bpm": 120.0},
    ]
    timing_lookup = {
        "song": {
            "bpm": 120.0,
            "first_downbeat_sec": 2.0,
            "timing_source": "madmom",
            "timing_confidence": "high",
            "needs_review": False,
        }
    }

    bpm, rebased, dropped = _resolve_sample_bpm_and_notes("song", notes, timing_lookup)

    assert bpm == pytest.approx(120.0)
    assert dropped == 0
    assert [note["beat"] for note in rebased] == pytest.approx([0.0, 2.0, 4.0])
    assert rebased[0]["time_seconds"] == pytest.approx(0.0)


def test_resolve_sample_bpm_and_notes_drops_pickup_notes_before_downbeat():
    notes = [
        {"beat": 0.0, "time_seconds": 1.0, "x": 0, "y": 0, "color": 0, "cut_direction": 1, "bpm": 120.0},
        {"beat": 4.0, "time_seconds": 2.0, "x": 1, "y": 1, "color": 1, "cut_direction": 2, "bpm": 120.0},
    ]
    timing_lookup = {
        "song": {
            "bpm": 120.0,
            "first_downbeat_sec": 1.5,
            "timing_source": "madmom",
            "timing_confidence": "high",
            "needs_review": False,
        }
    }

    _, rebased, dropped = _resolve_sample_bpm_and_notes("song", notes, timing_lookup)

    assert dropped == 1
    assert len(rebased) == 1
    assert rebased[0]["beat"] == pytest.approx(1.0)


def test_interpolate_muq_to_beat_grid_respects_beat_offset_seconds():
    raw = np.arange(100, dtype=np.float32).reshape(100, 1)

    aligned = interpolate_muq_to_beat_grid(raw, bpm=120.0, muq_hz=25.0, beat_offset_seconds=1.0)

    # 100 frames at 25 Hz = 4 seconds total. After a 1 second offset, 3 seconds remain.
    # At 120 BPM and 16 subdivisions/beat, that is 32 subdivisions/second.
    assert aligned.shape == (1, 96)


def test_resolve_single_song_timing_accepts_single_entry_payload():
    resolved = resolve_single_song_timing(
        {"bpm": 100.0, "first_downbeat_sec": 3.5, "timing_source": "manual"},
        timing_hash="song",
    )
    assert resolved["bpm"] == pytest.approx(100.0)
    assert resolved["first_downbeat_sec"] == pytest.approx(3.5)
