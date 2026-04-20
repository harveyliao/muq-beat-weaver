"""Tests for training schedule resolution and experiment evaluation helpers."""

from muq_beat_weaver.model.config import ModelConfig
from muq_beat_weaver.model.experiment_eval import _aggregate_metrics, _select_eval_indices
from muq_beat_weaver.model.training import _resolve_warmup_steps


class TestResolveWarmupSteps:
    def test_uses_ratio_by_default(self):
        config = ModelConfig(warmup_steps=4000, warmup_ratio=0.1)
        assert _resolve_warmup_steps(config, total_steps=1000) == 100

    def test_falls_back_to_explicit_steps_when_ratio_disabled(self):
        config = ModelConfig(warmup_steps=250, warmup_ratio=None)
        assert _resolve_warmup_steps(config, total_steps=1000) == 250

    def test_clamps_to_total_steps_minus_one(self):
        config = ModelConfig(warmup_steps=4000, warmup_ratio=None)
        assert _resolve_warmup_steps(config, total_steps=100) == 99


class TestSelectEvalIndices:
    def test_returns_all_indices_when_limit_exceeds_dataset(self):
        assert _select_eval_indices(total_items=4, limit=10) == [0, 1, 2, 3]

    def test_spreads_indices_across_dataset(self):
        assert _select_eval_indices(total_items=10, limit=4) == [0, 3, 6, 9]


class TestAggregateMetrics:
    def test_aggregates_means_and_ranges(self):
        per_sample = [
            {
                "sample_index": 0,
                "characteristic": "Standard",
                "reference_note_count": 10,
                "generated_note_count": 8,
                "onset_f1": 0.5,
                "nps_accuracy": 0.8,
            },
            {
                "sample_index": 1,
                "characteristic": "Standard",
                "reference_note_count": 20,
                "generated_note_count": 12,
                "onset_f1": 0.75,
                "nps_accuracy": 0.6,
            },
        ]
        summary = _aggregate_metrics(per_sample)
        assert summary["onset_f1"] == {"mean": 0.625, "min": 0.5, "max": 0.75}
        assert summary["nps_accuracy"] == {"mean": 0.7, "min": 0.6, "max": 0.8}
        assert summary["generated_note_count"] == {"mean": 10.0, "min": 8, "max": 12}
        assert summary["reference_note_count"] == {"mean": 15.0, "min": 10, "max": 20}
