import pytest

from wildboar.datasets import _split_repo_bundle


def test_split_repo_bundle_valid():
    assert _split_repo_bundle("repo/bundle:tag") == ("repo", "bundle", "tag")
    assert _split_repo_bundle("repo/bundle") == ("repo", "bundle", None)
    assert _split_repo_bundle("repo/bundle-with-hyphen:tag") == (
        "repo",
        "bundle-with-hyphen",
        "tag",
    )
    with pytest.raises(ValueError):
        _split_repo_bundle("repo/bundle_with_underscore:tag")
    with pytest.raises(ValueError):
        _split_repo_bundle("repo/bundle:")


def test_split_repo_bundle_edge_cases():
    assert _split_repo_bundle("") == (None, None, None)
    with pytest.raises(ValueError):
        _split_repo_bundle(":")
    with pytest.raises(ValueError):
        _split_repo_bundle("repo/")
    with pytest.raises(ValueError):
        _split_repo_bundle(":/bundle")


def test_split_repo_bundle_invalid():
    with pytest.raises(ValueError):
        _split_repo_bundle("invalid_repo_bundle_format")
