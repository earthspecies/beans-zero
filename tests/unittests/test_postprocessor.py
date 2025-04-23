import doctest
from beans_zero import post_processor


def test_postprocessor_doctests() -> None:
    """Run doctests in the utils module."""
    failures, _ = doctest.testmod(post_processor)
    assert failures == 0
