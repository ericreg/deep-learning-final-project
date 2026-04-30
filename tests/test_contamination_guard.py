import pytest

from unlearning_demo.data import assert_no_benchmark_training


def test_contamination_guard_accepts_cyber_corpora_payload():
    assert_no_benchmark_training(
        {"domain": "cyber", "source": {"wmdp_benchmark_used_for_training": False}}
    )


def test_contamination_guard_rejects_benchmark_training():
    with pytest.raises(ValueError):
        assert_no_benchmark_training(
            {"domain": "cyber", "source": {"wmdp_benchmark_used_for_training": True}}
        )


def test_contamination_guard_rejects_non_cyber_payload():
    with pytest.raises(ValueError):
        assert_no_benchmark_training(
            {"domain": "other", "source": {"wmdp_benchmark_used_for_training": False}}
        )
