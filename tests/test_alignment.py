from unlearning_demo.token_alignment import align_translated_to_original


def test_alignment_preserves_equal_tokens_and_maps_replacement():
    mapping = align_translated_to_original(
        original_ids=[10, 20, 30, 40],
        translated_ids=[10, 99, 40],
    )
    assert mapping[0] == 0
    assert mapping[-1] == 2
    assert all(0 <= idx <= 2 for idx in mapping)
