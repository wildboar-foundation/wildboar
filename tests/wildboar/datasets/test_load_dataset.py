import pytest

from wildboar.datasets import load_datasets


@pytest.mark.parametrize("merge_train_test, length", [(True, 2), (False, 4)])
def test_load_datasets_merge_train_test(merge_train_test, length):
    lengths = [
        len(ret)
        for _, ret in load_datasets(
            "wildboar/ucr-tiny", merge_train_test=merge_train_test
        )
    ]
    assert lengths == [length] * 5
