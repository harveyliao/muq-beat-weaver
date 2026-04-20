"""Tests for MuQ embedding export helpers."""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from muq_beat_weaver.model.muq_embeddings import (
    _build_window_plan,
    _merge_window_embeddings,
    _muq_cache_version_key,
    find_audio_files_in_subfolders,
    summarize_embedding,
)


def test_find_audio_files_in_subfolders_respects_sorted_limit(tmp_path):
    root = tmp_path / "official"
    for name in ("c_track", "a_track", "b_track"):
        folder = root / name
        folder.mkdir(parents=True)
        (folder / "song.wav").write_bytes(b"wav")

    found = find_audio_files_in_subfolders(root, limit=2)

    assert [path.parent.name for path in found] == ["a_track", "b_track"]
    assert all(path.name == "song.wav" for path in found)


def test_summarize_embedding_reports_shape_bytes_and_nan_flag(tmp_path):
    embedding = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32)
    stats = summarize_embedding(
        embedding,
        audio_path=tmp_path / "song.wav",
        embedding_path=tmp_path / "song.npy",
        sample_rate=24000,
        audio_seconds=1.5,
        load_audio_seconds=0.1,
        inference_seconds=0.2,
        save_seconds=0.05,
    )

    assert stats.embedding_shape == [2, 2]
    assert stats.embedding_bytes == embedding.nbytes
    assert stats.contains_nan is False
    assert stats.total_seconds == pytest.approx(0.35)
    assert stats.mean_abs == pytest.approx(1.0)


def test_build_window_plan_uses_overlap_until_end():
    windows = _build_window_plan(400.0, window_seconds=180.0, overlap_seconds=30.0)

    assert windows == [
        (0.0, 180.0),
        (150.0, 330.0),
        (300.0, 400.0),
    ]


def test_merge_window_embeddings_uses_midpoint_ownership():
    first = np.arange(10, dtype=np.float32).reshape(5, 2)
    second = (100 + np.arange(10, dtype=np.float32)).reshape(5, 2)

    merged = _merge_window_embeddings(
        [first, second],
        [(0.0, 4.0), (3.0, 7.0)],
        frame_hz=1.0,
    )

    expected = np.concatenate([first[:4], second[1:]], axis=0)
    np.testing.assert_array_equal(merged, expected)


def test_muq_cache_version_key_records_windowing_strategy():
    version = _muq_cache_version_key(
        model_name="OpenMuQ/MuQ-large-msd-iter",
        label_rate=25.0,
        sample_rate=24000,
        window_seconds=180.0,
        overlap_seconds=30.0,
    )

    assert "muq_cache_v2" in version
    assert "window_seconds=180.000" in version
    assert "overlap_seconds=30.000" in version
    assert "merge=midpoint_ownership" in version


