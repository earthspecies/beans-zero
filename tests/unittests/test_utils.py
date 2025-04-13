import doctest
from beans_zero import utils


def test_postprocessor_doctests() -> None:
    """Run doctests in the utils module."""
    failures, _ = doctest.testmod(utils)
    assert failures == 0
