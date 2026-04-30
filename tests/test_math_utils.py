from unlearning_demo.math_utils import generic_logits, soft_label_cross_entropy, softmax


def test_generic_logits_penalize_reinforced_increases_only():
    out = generic_logits([1.0, 2.0, 3.0], [2.0, 1.0, 3.5], alpha=2.0)
    assert out == [-1.0, 2.0, 2.0]


def test_soft_label_cross_entropy_is_positive():
    loss = soft_label_cross_entropy([0.0, 1.0], softmax([1.0, 0.0]))
    assert loss > 0
